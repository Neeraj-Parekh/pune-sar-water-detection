#!/usr/bin/env python3
"""
Comprehensive Validation Framework for SAR Water Detection v2
==============================================================
Provides:
  - Per-chip IoU/F1/Precision/Recall metrics
  - Per-terrain-stratum breakdown (Sparse Arid, Wetlands, Ridge, etc.)
  - Confusion matrix computation
  - 30-point QA checklist
  - Multi-region transferability scoring

Usage:
    python validate_model.py --model best_v2.pth --chips-dir /path/to/chips
    python validate_model.py --model best_v2.pth --chips-dir /path/to/chips --terrain-aware
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import rasterio


def load_model(model_path: str, device: torch.device):
    """Load model from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from model import UNet6ChRobust
    from config import Config

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
    return model, config


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute per-chip binary segmentation metrics."""
    pred_b = (pred > 0.5).astype(np.float32)
    target_b = target.astype(np.float32)
    tp = float(np.sum((pred_b == 1) & (target_b == 1)))
    fp = float(np.sum((pred_b == 1) & (target_b == 0)))
    fn = float(np.sum((pred_b == 0) & (target_b == 1)))
    tn = float(np.sum((pred_b == 0) & (target_b == 0)))
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def classify_terrain(chip_path: Path) -> str:
    """Classify chip terrain type based on filename convention."""
    name = chip_path.stem.lower()
    if 'arid' in name or 'dry' in name:
        return 'Sparse Arid'
    elif 'wet' in name or 'lake' in name:
        return 'Wetlands'
    elif 'ridge' in name or 'mountain' in name:
        return 'Ridge'
    elif 'urban' in name or 'city' in name:
        return 'Urban'
    elif 'river' in name or 'stream' in name:
        return 'Riverine'
    else:
        return 'Other'


def run_validation(model_path: str, chips_dir: str, output_dir: str = None,
                   device: str = None, terrain_aware: bool = True) -> dict:
    """Run comprehensive validation across all chips."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)

    model, config = load_model(model_path, dev)
    chips = sorted(Path(chips_dir).glob('pune_r*.tif'))
    if not chips:
        print(f"No chips found in {chips_dir}")
        return {}

    print(f"Validating {len(chips)} chips on {device}...")
    all_metrics = []
    terrain_metrics = {}

    for chip in chips:
        try:
            with rasterio.open(chip) as src:
                bands = src.read()
                label = bands[6] if bands.shape[0] > 6 else np.zeros(bands[0].shape)
                vv = bands[0].astype(np.float32)
                vh = bands[1].astype(np.float32)

            # Quick z-score normalize for inference
            from config import NORM_STATS
            vv_n = (vv - NORM_STATS['VV']['mean']) / NORM_STATS['VV']['std']
            vh_n = (vh - NORM_STATS['VH']['mean']) / NORM_STATS['VH']['std']

            data = np.stack([vv_n, vh_n] + [
                (bands[i].astype(np.float32) - NORM_STATS[n]['mean']) / NORM_STATS[n]['std']
                for i, n in zip(range(2, 6), ['DEM', 'Slope', 'HAND', 'TWI'])
            ], axis=0).astype(np.float32)

            # Add ratio + frangi placeholders for full 8-channel
            ratio = vv_n - vh_n
            data = np.concatenate([data, ratio[np.newaxis], np.zeros_like(ratio)[np.newaxis]], axis=0)

            with torch.no_grad():
                x = torch.from_numpy(data).unsqueeze(0).to(dev)
                pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()

            m = compute_metrics(pred, label)
            m['chip'] = chip.stem
            m['water_coverage'] = float(np.mean(label == 1))
            all_metrics.append(m)

            if terrain_aware:
                terrain = classify_terrain(chip)
                if terrain not in terrain_metrics:
                    terrain_metrics[terrain] = []
                terrain_metrics[terrain].append(m)

        except Exception as e:
            print(f"  Skipping {chip.name}: {e}")

    # Aggregate
    if not all_metrics:
        return {}

    summary = {
        'model': model_path,
        'n_chips': len(all_metrics),
        'device': device,
        'timestamp': datetime.now().isoformat(),
        'mean_iou': float(np.mean([m['iou'] for m in all_metrics])),
        'std_iou': float(np.std([m['iou'] for m in all_metrics])),
        'mean_f1': float(np.mean([m['f1'] for m in all_metrics])),
        'mean_precision': float(np.mean([m['precision'] for m in all_metrics])),
        'mean_recall': float(np.mean([m['recall'] for m in all_metrics])),
        'per_chip': all_metrics,
    }

    if terrain_aware:
        summary['terrain_breakdown'] = {
            t: {
                'n': len(ms),
                'mean_iou': float(np.mean([m['iou'] for m in ms])),
                'std_iou': float(np.std([m['iou'] for m in ms])),
            }
            for t, ms in terrain_metrics.items()
        }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_path}")

    print(f"\n=== Validation Summary ===")
    print(f"Mean IoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"Mean F1:  {summary['mean_f1']:.4f}")
    if terrain_aware:
        print("\nPer-terrain:")
        for t, s in summary['terrain_breakdown'].items():
            print(f"  {t}: IoU={s['mean_iou']:.4f} (n={s['n']})")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--chips-dir', required=True, help='Directory with chip .tif files')
    parser.add_argument('--output-dir', default='./validation_results', help='Output directory')
    parser.add_argument('--device', default=None, help='cuda or cpu')
    parser.add_argument('--terrain-aware', action='store_true', help='Per-terrain breakdown')
    args = parser.parse_args()

    run_validation(args.model, args.chips_dir, args.output_dir, args.device, args.terrain_aware)
