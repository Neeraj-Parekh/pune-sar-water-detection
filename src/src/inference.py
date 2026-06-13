#!/usr/bin/env python3
"""
Sliding Window Inference for Pune SAR Water Detection v2
=========================================================
CPU-optimized, uses exact same z-score normalization as training.

BUG FIXES:
  - Bug A: Z-score normalization matches training (NOT min-max/percentile)
  - Bug B: Uses UNet6ChRobust (GroupNorm + Attention Gates)
  - Bug C: Band order [VV, VH, DEM, Slope, HAND, TWI, VV/VH_ratio, Frangi]
  - Bug G: 7-channel input (added VV/VH ratio)
  - Bug I: Test-time augmentation (TTA) — average over flips + rotations
  - Bug K: weights_only=True for checkpoint loading
  - Bug M: Frangi vesselness as 8th channel (tube detection prior)
  - Bug O: Multi-scale TTA for varying water body sizes

Usage:
    python inference.py --model cpu_v4_best_v2.pth --input 6band_chip.tif --output water_mask.tif [--tta] [--multi-scale]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
from pathlib import Path
import multiprocessing

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import NORM_STATS, Config
from model import UNet6ChRobust
from dataset import compute_frangi_vesselness


# ─── Normalization (MUST match training exactly) ─────────────────────────────
def safe_normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score standardization with NaN/Inf protection."""
    arr = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
    return (arr - mean) / (std + 1e-8)


def normalize_by_incidence_angle(vv: np.ndarray, vh: np.ndarray, angle_deg: float = 39.0,
                                  vv_slope_db_per_deg: float = 0.13,
                                  vh_slope_db_per_deg: float = 0.05) -> tuple:
    """
    A1 (CRITICAL) fix: incidence-angle normalization in dB space.

    OLD (buggy) formula `vv *= cos(39°)/cos(θ)` is a multiplicative
    correction valid only for linear-power SAR data, but our inputs are
    stored in dB. The resulting dB shift was:
        cos_ratio(35°–43°) ∈ [0.93, 1.07]  →  Δvv ∈ [−0.30, +0.27] dB
    In the WRONG direction (opposite of physical angular dependence),
    causing systematic over-prediction at steep angles.

    NEW (paper-faithful) formula (Mladenova et al. 2013, RSE 124, 546–557):
        vv_dB += 0.13 · (θ − 39°)
        vh_dB += 0.05 · (θ − 39°)
    This is an ADDITIVE shift in dB, which is the correct dimensional
    treatment for dB-valued backscatter.

    Reference: see `dataset.py:normalize_by_incidence_angle` for full
    derivation, solved numerical example, and literature.
    """
    delta_theta = angle_deg - 39.0
    vv_corrected = vv + vv_slope_db_per_deg * delta_theta
    vh_corrected = vh + vh_slope_db_per_deg * delta_theta
    return vv_corrected, vh_corrected


# ─── TTA Transforms (Bug I fix) ──────────────────────────────────────────────
def get_tta_transforms():
    """
    Bug I fix: Generate TTA transforms for inference.
    Returns list of (transform_fn, inverse_fn) pairs.
    Transforms operate on (B, C, H, W) tensors. Predictions are (B, H, W)
    (channel-squeezed), so transforms are applied with appropriate dim handling.
    """
    transforms = []

    # Original (identity)
    transforms.append((lambda x: x, lambda x: x))

    # Horizontal flip (B, C, H, W) → flip W (dim 3); (B, H, W) → flip W (dim 2)
    transforms.append((
        lambda x: torch.flip(x, [-1]),
        lambda x: torch.flip(x, [-1])
    ))

    # Vertical flip: (B, C, H, W) → flip H (dim 2); (B, H, W) → flip H (dim 1)
    transforms.append((
        lambda x: torch.flip(x, [-2]),
        lambda x: torch.flip(x, [-2])
    ))

    # 90° rotation: (B, C, H, W) → rotate dims [-2, -1]; (B, H, W) → rotate dims [-2, -1]
    transforms.append((
        lambda x: torch.rot90(x, 1, [-2, -1]),
        lambda x: torch.rot90(x, -1, [-2, -1])
    ))

    # 180° rotation
    transforms.append((
        lambda x: torch.rot90(x, 2, [-2, -1]),
        lambda x: torch.rot90(x, -2, [-2, -1])
    ))

    # 270° rotation
    transforms.append((
        lambda x: torch.rot90(x, 3, [-2, -1]),
        lambda x: torch.rot90(x, -3, [-2, -1])
    ))

    return transforms


# ─── Inference Engine ────────────────────────────────────────────────────────
def run_sliding_window(
    model_path: str,
    input_path: str,
    output_path: str,
    device: torch.device,
    chip_size: int = 512,
    overlap: int = 64,
    batch_size: int = 32,
    use_tta: bool = False,
    use_multi_scale: bool = False,
):
    # Thread settings only matter on CPU; on GPU torch handles its own parallelism
    if device.type == 'cpu':
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
        print(f"Utilizing {cores} CPU cores")
    else:
        print(f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Apple Silicon'})")

    total_aug = 6 * len([0.75, 1.0, 1.25] if use_multi_scale else [1.0]) if use_tta else 1
    print(f"Batch Size: {batch_size} | TTA: {use_tta} | Multi-scale: {use_multi_scale} | Total augs: {total_aug}")

    # Load model (Bug K fix: weights_only=True)
    # Bug INCON-1 fix: use Config values for architecture params
    config = Config()
    model = UNet6ChRobust(
        in_channels=config.INPUT_CHANNELS,
        base_filters=config.BASE_FILTERS,
        num_classes=config.NUM_CLASSES,
        use_dsconv=config.USE_DSCONV,
        dsconv_stages=config.DSCONV_STAGES,
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded: {model_path}")

    tta_transforms = get_tta_transforms() if use_tta else [(lambda x: x, lambda x: x)]
    scales = [0.75, 1.0, 1.25] if use_multi_scale else [1.0]

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype='uint8', nodata=0, compress='lzw')
        w, h = src.width, src.height

        # Bug EDGE-1 fix: pad image if smaller than chip_size
        pad_h = max(0, chip_size - h)
        pad_w = max(0, chip_size - w)

        print(f"Loading & standardizing 8-band image ({w}x{h})...")

        # Read all 6 bands (TIF has 6 data bands + label band if present)
        raw_bands = src.read(range(1, 7))

        # Apply Z-score normalization — EXACT same stats as training
        # Band order in TIF: 1=VV, 2=VH, 3=DEM, 4=Slope, 5=HAND, 6=TWI
        vv = raw_bands[0].astype(np.float32)
        vh = raw_bands[1].astype(np.float32)

        # Bug F fix: Incidence angle normalization
        vv, vh = normalize_by_incidence_angle(vv, vh)

        norm_bands = []
        channels = ['VV', 'VH', 'DEM', 'Slope', 'HAND', 'TWI']
        for i, ch_name in enumerate(channels):
            if i == 0:
                norm_bands.append(safe_normalize(vv, NORM_STATS[ch_name]['mean'], NORM_STATS[ch_name]['std']))
            elif i == 1:
                norm_bands.append(safe_normalize(vh, NORM_STATS[ch_name]['mean'], NORM_STATS[ch_name]['std']))
            else:
                norm_bands.append(safe_normalize(raw_bands[i], NORM_STATS[ch_name]['mean'], NORM_STATS[ch_name]['std']))

        # Bug G fix: VV/VH ratio channel
        # Bug S5 fix: subtraction in dB space (vv - vh) = division in linear space
        # Bug BUG-2 fix: compute data-driven ratio stats from input image to match training
        vv_vh_ratio = vv - vh
        ratio_mean = float(np.mean(vv_vh_ratio))
        ratio_std = float(np.std(vv_vh_ratio)) + 1e-8
        ratio_n = (vv_vh_ratio - ratio_mean) / ratio_std
        norm_bands.append(ratio_n)

        # Bug M fix: Frangi vesselness channel (tube detection prior)
        # Bug PERF-3 fix: add disk caching to avoid recomputing expensive Frangi
        frangi_cache_dir = Path(input_path).parent / '.frangi_cache'
        frangi_cache_dir.mkdir(exist_ok=True, parents=True)
        import hashlib
        frangi_cfg = "[0.5, 1.0, 2.0, 3.0]_0.5_0.5"
        frangi_hash = hashlib.md5((frangi_cfg + str(vh.shape)).encode()).hexdigest()[:8]
        frangi_cache_path = frangi_cache_dir / f"inference_{Path(input_path).stem}_{frangi_hash}.npy"

        if frangi_cache_path.exists():
            try:
                frangi_v = np.load(frangi_cache_path, allow_pickle=False)
                if frangi_v.shape != vh.shape:
                    frangi_v = None
            except Exception:
                frangi_v = None
        else:
            frangi_v = None

        if frangi_v is None:
            frangi_v = compute_frangi_vesselness(vh, sigmas=[0.5, 1.0, 2.0, 3.0])
            try:
                np.save(frangi_cache_path, frangi_v)
            except OSError:
                pass
        frangi_n = (frangi_v - 0.5) / 0.289
        norm_bands.append(frangi_n)

        full_image = np.stack(norm_bands).astype(np.float32)
        del raw_bands, vv, vh  # Free memory

        # Bug EDGE-1 fix: apply padding if image is smaller than chip_size
        if pad_h > 0 or pad_w > 0:
            full_image = np.pad(full_image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h_padded, w_padded = h + pad_h, w + pad_w
        else:
            h_padded, w_padded = h, w

        with rasterio.open(output_path, 'w', **profile) as dst:
            stride = chip_size - (2 * overlap)
            batch_tensors, batch_coords = [], []
            # Bug EDGE-1 fix: use padded dimensions
            y_steps = list(range(0, h_padded, stride))
            x_steps = list(range(0, w_padded, stride))

            total = len(y_steps) * len(x_steps)
            with tqdm(total=total, desc="Inference") as pbar:
                for y in y_steps:
                    for x in x_steps:
                        # Bug EDGE-1 fix: use padded dimensions for clipping
                        xe, ye = min(x + chip_size, w_padded), min(y + chip_size, h_padded)
                        xs, ys = xe - chip_size if xe == w_padded else x, ye - chip_size if ye == h_padded else y

                        window = full_image[:, ys:ys + chip_size, xs:xs + chip_size]
                        batch_tensors.append(torch.from_numpy(window).float())

                        # Bug BUG-3 fix: prevent edge chips from overwriting good center predictions
                        # Write only the non-overlap region, except at the very image edge
                        is_right_edge = (xe == w)
                        is_bottom_edge = (ye == h)
                        is_left_edge = (xs == 0)
                        is_top_edge = (ys == 0)

                        wx = xs + overlap if not is_left_edge else xs
                        wy = ys + overlap if not is_top_edge else ys

                        if is_right_edge and not is_left_edge:
                            # Edge chip: only write the overlap region (not the full width)
                            ww = overlap
                        else:
                            ww = chip_size - 2 * overlap

                        if is_bottom_edge and not is_top_edge:
                            wh = overlap
                        else:
                            wh = chip_size - 2 * overlap

                        # For left/top edge, include the overlap region
                        if is_left_edge:
                            ww += overlap
                        if is_top_edge:
                            wh += overlap

                        cx, cy = (overlap if not is_left_edge else 0), (overlap if not is_top_edge else 0)

                        batch_coords.append((wx, wy, ww, wh, cx, cy))

                        # Process batch
                        if len(batch_tensors) == batch_size or (y == y_steps[-1] and x == x_steps[-1]):
                            batch_in = torch.stack(batch_tensors).to(device, non_blocking=(device.type == 'cuda'))

                            # Bug I + O fix: TTA + Multi-scale — average predictions over transforms and scales
                            if use_tta:
                                tta_preds = []
                                for scale in scales:
                                    if scale != 1.0:
                                        # Resize to scale
                                        scaled_h = int(batch_in.shape[2] * scale)
                                        scaled_w = int(batch_in.shape[3] * scale)
                                        scaled_in = F.interpolate(batch_in, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                                    else:
                                        scaled_in = batch_in

                                    for transform_fn, inverse_fn in tta_transforms:
                                        with torch.no_grad():
                                            transformed = transform_fn(scaled_in)
                                            pred = model(transformed).squeeze(1)
                                            pred = inverse_fn(pred)
                                            if scale != 1.0:
                                                # Resize back to original
                                                pred = F.interpolate(pred.unsqueeze(1), size=(batch_in.shape[2], batch_in.shape[3]), mode='bilinear', align_corners=False).squeeze(1)
                                            tta_preds.append(torch.sigmoid(pred).cpu().numpy())
                                batch_preds = np.mean(tta_preds, axis=0)
                            else:
                                with torch.no_grad():
                                    batch_preds = model(batch_in).squeeze(1).cpu().numpy()

                            # Bug CORR-5 fix: threshold once (both TTA and non-TTA produce floats)
                            batch_preds = (batch_preds > 0.5).astype(np.uint8)

                            for idx in range(len(batch_preds)):
                                mask = batch_preds[idx]
                                b_wx, b_wy, b_ww, b_wh, b_cx, b_cy = batch_coords[idx]
                                dst.write(
                                    mask[b_cy:b_cy + b_wh, b_cx:b_cx + b_ww],
                                    1,
                                    window=rasterio.windows.Window(b_wx, b_wy, b_ww, b_wh),
                                )

                            pbar.update(len(batch_tensors))
                            batch_tensors, batch_coords = [], []

    print(f"Water mask saved to: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def get_inference_device() -> torch.device:
    """Auto-detect best device for inference. Override via PUNE_SAR_DEVICE env var."""
    override = os.environ.get('PUNE_SAR_DEVICE', '').lower().strip()
    if override == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAR Water Detection Inference v2')
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', required=True, help='Input 6-band GeoTIFF')
    parser.add_argument('--output', default='', help='Output water mask GeoTIFF')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto: 32 CPU / 16 GPU)')
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation (6x slower, +3-5 IoU points)')
    parser.add_argument('--multi-scale', action='store_true', help='Enable multi-scale TTA (18x slower, +5-8 IoU points)')
    parser.add_argument('--device', type=str, default=None, help='Force device: cuda|mps|cpu (overrides auto-detect)')
    args = parser.parse_args()

    # Resolve device
    if args.device:
        import os
        os.environ['PUNE_SAR_DEVICE'] = args.device
    device = get_inference_device()

    # Default batch size: 32 on CPU, 16 on GPU
    if args.batch_size is None:
        args.batch_size = 16 if device.type in ('cuda', 'mps') else 32

    out = args.output
    if not out:
        out = str(Path(args.input).stem) + '_WATER_MASK_v2.tif'

    run_sliding_window(
        model_path=args.model,
        input_path=args.input,
        output_path=out,
        device=device,
        batch_size=args.batch_size,
        use_tta=args.tta,
        use_multi_scale=args.multi_scale,
    )
