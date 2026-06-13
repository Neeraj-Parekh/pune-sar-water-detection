#!/usr/bin/env python3
"""
Method B: Ablation + Spatial Overlay
=====================================
Runs inference twice (with/without DEM) and computes error difference
maps overlaid with curvature classes.

Shows: "DEM removal disproportionately fixes errors in high-curvature terrain."

Usage:
    python ablation_overlay.py [--model cpu_v4_best.pth] [--n-chips 20]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from config import Config, NORM_STATS
from model_v4_simple import UNet6Ch
from dataset import compute_frangi_vesselness, safe_normalize


# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent
CHIPS_DIR = Path(__file__).parent / 'chips'
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model(model_path, device):
    """Load trained U-Net model (original 6-channel v4 architecture)."""
    model = UNet6Ch(in_channels=6, base_filters=64)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def prepare_chip(npy_path, terrain_zeroed=False):
    """
    Load a numpy chip and prepare it for inference (6-channel model).
    
    Args:
        npy_path: Path to .npy file (513x513x7)
        terrain_zeroed: If True, zero out DEM/Slope/HAND/TWI channels
    
    Returns:
        tensor: (1, 6, H, W) normalized input tensor
        label: (H, W) ground truth label
    """
    data = np.load(npy_path)  # (513, 513, 7)
    vv = data[:, :, 0]
    vh = data[:, :, 1]
    dem = data[:, :, 2]
    slope = data[:, :, 3]
    hand = data[:, :, 4]
    twi = data[:, :, 5]
    label = data[:, :, 6]  # Last band is label

    # Zero terrain channels if ablating
    if terrain_zeroed:
        dem = np.zeros_like(dem)
        slope = np.zeros_like(slope)
        hand = np.zeros_like(hand)
        twi = np.zeros_like(twi)

    # Normalize SAR channels
    vv_norm = safe_normalize(vv, NORM_STATS['VV']['mean'], NORM_STATS['VV']['std'])
    vh_norm = safe_normalize(vh, NORM_STATS['VH']['mean'], NORM_STATS['VH']['std'])

    # Normalize terrain channels
    dem_norm = safe_normalize(dem, NORM_STATS['DEM']['mean'], NORM_STATS['DEM']['std'])
    slope_norm = safe_normalize(slope, NORM_STATS['Slope']['mean'], NORM_STATS['Slope']['std'])
    hand_norm = safe_normalize(hand, NORM_STATS['HAND']['mean'], NORM_STATS['HAND']['std'])
    twi_norm = safe_normalize(twi, NORM_STATS['TWI']['mean'], NORM_STATS['TWI']['std'])

    # Stack channels: [VV, VH, DEM, Slope, HAND, TWI] (6 channels)
    channels = np.stack([
        vv_norm, vh_norm, dem_norm, slope_norm, hand_norm, twi_norm
    ], axis=0)  # (6, H, W)

    # Crop to 512x512 (model requires divisible by 16)
    channels = channels[:, :512, :512]
    label = label[:512, :512]

    # Convert to tensor
    tensor = torch.from_numpy(channels).float().unsqueeze(0)  # (1, 6, H, W)

    return tensor, label


def compute_curvature(dem, cell_size=10.0):
    """
    Compute terrain curvature from DEM using finite differences.
    
    Returns:
        curvature: (H, W) array of curvature values
        curvature_class: (H, W) array of curvature classes
            0 = flat, 1 = ridge (convex), 2 = valley (concave)
    """
    # Compute second derivatives
    dzdx = ndimage.sobel(dem, axis=1) / cell_size
    dzdy = ndimage.sobel(dem, axis=0) / cell_size
    d2zdx2 = ndimage.sobel(dzdx, axis=1) / cell_size
    d2zdy2 = ndimage.sobel(dzdy, axis=0) / cell_size

    # Curvature (profile curvature approximation)
    curvature = -(d2zdx2 + d2zdy2)

    # Classify curvature
    curvature_class = np.zeros_like(curvature, dtype=int)
    curvature_class[curvature > 0.1] = 1  # Ridge (convex)
    curvature_class[curvature < -0.1] = 2  # Valley (concave)

    return curvature, curvature_class


def run_inference(model, tensor, device):
    """Run model inference on a single chip."""
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
    return prob


def compute_error_map(pred_prob, label, threshold=0.5):
    """Compute binary error map (1 = error, 0 = correct)."""
    pred_binary = (pred_prob > threshold).astype(int)
    label_binary = (label > 0.5).astype(int)
    error = (pred_binary != label_binary).astype(float)
    return error, pred_binary, label_binary


def classify_errors(pred_binary, label_binary):
    """
    Classify errors into FP and FN.
    
    Returns:
        fp_mask: (H, W) True where false positive
        fn_mask: (H, W) True where false negative
    """
    label_binary = (label_binary > 0.5).astype(int)
    pred_binary = (pred_binary > 0.5).astype(int)
    fp_mask = (pred_binary == 1) & (label_binary == 0)
    fn_mask = (pred_binary == 0) & (label_binary == 1)
    return fp_mask, fn_mask


def main():
    parser = argparse.ArgumentParser(description='Ablation Overlay Analysis')
    parser.add_argument('--model', type=str, default=str(RESULTS_DIR / 'cpu_v4_best.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--n-chips', type=int, default=20,
                        help='Number of chips to analyze')
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)

    # Find chips
    chip_files = sorted(CHIPS_DIR.glob('*.npy'))[:args.n_chips]
    print(f"Analyzing {len(chip_files)} chips...")

    # Results storage
    results = []

    for i, chip_path in enumerate(chip_files):
        chip_name = chip_path.stem
        print(f"  [{i+1}/{len(chip_files)}] {chip_name}")

        # Load chip (full model)
        tensor_full, label = prepare_chip(chip_path, terrain_zeroed=False)
        prob_full = run_inference(model, tensor_full, device)

        # Load chip (DEM ablated)
        tensor_ablated, _ = prepare_chip(chip_path, terrain_zeroed=True)
        prob_ablated = run_inference(model, tensor_ablated, device)

        # Compute error maps
        error_full, pred_full, label_bin = compute_error_map(prob_full, label)
        error_ablated, pred_ablated, _ = compute_error_map(prob_ablated, label)

        # Error difference: positive = removing DEM fixes errors
        error_diff = error_full - error_ablated

        # Compute curvature from DEM (crop to 512 to match model output)
        dem = np.load(chip_path)[:512, :512, 2]
        curvature, curvature_class = compute_curvature(dem)

        # Analyze error difference by curvature class
        class_names = ['flat', 'ridge', 'valley']
        class_stats = []

        for cls_id, cls_name in enumerate(class_names):
            mask = curvature_class == cls_id
            if mask.sum() > 0:
                mean_diff = error_diff[mask].mean()
                std_diff = error_diff[mask].std()
                n_pixels = mask.sum()

                # Also compute FP/FN counts
                fp_full, fn_full = classify_errors(pred_full, label_bin)
                fp_ablated, fn_ablated = classify_errors(pred_ablated, label_bin)

                fp_change = fp_ablated[mask].sum() - fp_full[mask].sum()
                fn_change = fn_ablated[mask].sum() - fn_full[mask].sum()

                class_stats.append({
                    'class': cls_name,
                    'n_pixels': int(n_pixels),
                    'mean_error_diff': float(mean_diff),
                    'std_error_diff': float(std_diff),
                    'fp_change': int(fp_change),
                    'fn_change': int(fn_change),
                })

        results.append({
            'chip': chip_name,
            'curvature_classes': class_stats,
            'overall_error_diff_mean': float(error_diff.mean()),
            'overall_error_diff_std': float(error_diff.std()),
        })

    # Aggregate results
    print("\n" + "="*70)
    print("ABLATION OVERLAY RESULTS")
    print("="*70)

    # Aggregate by curvature class
    agg = {}
    for r in results:
        for cs in r['curvature_classes']:
            cls = cs['class']
            if cls not in agg:
                agg[cls] = {'mean_diffs': [], 'fp_changes': [], 'fn_changes': []}
            agg[cls]['mean_diffs'].append(cs['mean_error_diff'])
            agg[cls]['fp_changes'].append(cs['fp_change'])
            agg[cls]['fn_changes'].append(cs['fn_change'])

    print(f"\n{'Curvature Class':<15} {'Mean Error Diff':>16} {'FP Change':>12} {'FN Change':>12} {'Interpretation':>20}")
    print("-"*75)

    for cls in ['flat', 'ridge', 'valley']:
        if cls in agg:
            mean_diff = np.mean(agg[cls]['mean_diffs'])
            fp_change = np.sum(agg[cls]['fp_changes'])
            fn_change = np.sum(agg[cls]['fn_changes'])

            if mean_diff > 0.01:
                interp = "DEM removal reduces errors"
            elif mean_diff < -0.01:
                interp = "DEM helps (adds errors)"
            else:
                interp = "No significant effect"

            print(f"{cls:<15} {mean_diff:>+16.4f} {fp_change:>+12d} {fn_change:>+12d} {interp:>20}")

    # Generate visualization
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    classes = ['flat', 'ridge', 'valley']
    mean_diffs = [np.mean(agg[c]['mean_diffs']) if c in agg else 0 for c in classes]
    fp_changes = [np.sum(agg[c]['fp_changes']) if c in agg else 0 for c in classes]

    # Plot 1: Error difference by curvature
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = axes[0].bar(classes, mean_diffs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Mean Error Difference\n(Full - Ablated)', fontsize=10)
    axes[0].set_title('DEM Effect by Curvature Class', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, mean_diffs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:+.4f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: FP change by curvature
    bars2 = axes[1].bar(classes, fp_changes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('False Positive Change\n(Ablated - Full)', fontsize=10)
    axes[1].set_title('FP Pixels Changed by DEM Removal', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, fp_changes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f'{val:+d}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Example error diff map
    if len(results) > 0:
        # Load first chip for visualization
        chip_path = chip_files[0]
        tensor_full, label = prepare_chip(chip_path, terrain_zeroed=False)
        prob_full = run_inference(model, tensor_full, device)
        tensor_ablated, _ = prepare_chip(chip_path, terrain_zeroed=True)
        prob_ablated = run_inference(model, tensor_ablated, device)

        error_full, _, _ = compute_error_map(prob_full, label)
        error_ablated, _, _ = compute_error_map(prob_ablated, label)
        error_diff = error_full - error_ablated

        dem = np.load(chip_path)[:, :, 2]
        _, curvature_class = compute_curvature(dem)

        # Show error diff with curvature overlay
        im = axes[2].imshow(error_diff, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].contour(curvature_class, levels=[0.5, 1.5], colors=['black', 'red'],
                        linewidths=0.5, alpha=0.5)
        axes[2].set_title('Error Difference Map\n(Red=DEM removal reduces errors, Blue=DEM removal increases errors)', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_ablation_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig_ablation_overlay.png'}")

    # Save results
    output_results = {
        'aggregate': {
            cls: {
                'mean_error_diff': float(np.mean(agg[cls]['mean_diffs'])) if cls in agg else 0,
                'std_error_diff': float(np.std(agg[cls]['mean_diffs'])) if cls in agg else 0,
                'total_fp_change': int(np.sum(agg[cls]['fp_changes'])) if cls in agg else 0,
                'total_fn_change': int(np.sum(agg[cls]['fn_changes'])) if cls in agg else 0,
            }
            for cls in ['flat', 'ridge', 'valley']
        },
        'per_chip': results,
        'n_chips': len(chip_files),
    }

    with open(OUTPUT_DIR / 'ablation_overlay_results.json', 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"Results saved: {OUTPUT_DIR / 'ablation_overlay_results.json'}")


if __name__ == '__main__':
    main()
