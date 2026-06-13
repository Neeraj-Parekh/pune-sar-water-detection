#!/usr/bin/env python3
"""
Baseline Comparison for SAR Water Detection
=============================================
Bug J fix: Implements Otsu thresholding on VV and MNDWI on Sentinel-2
for comparison with our deep learning model.

Usage:
    python baselines.py --input chip.tif --label label.tif
    python baselines.py --input_dir chips/ --label_dir labels/
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path
from skimage.filters import threshold_otsu


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """Compute IoU, Precision, Recall, F1 for binary prediction."""
    pred = (pred > 0.5).astype(np.float32)
    label = (label > 0.5).astype(np.float32)

    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def otsu_vv(vv_band: np.ndarray) -> np.ndarray:
    """
    Bug J fix: Otsu thresholding on VV band.
    Water has low backscatter in VV, so Otsu finds the bimodal threshold.
    """
    valid = ~np.isnan(vv_band)
    if valid.sum() < 100:
        return np.zeros_like(vv_band)

    try:
        thresh = threshold_otsu(vv_band[valid])
        # Water = low backscatter (below threshold)
        return (vv_band < thresh).astype(np.uint8)
    except Exception:
        return np.zeros_like(vv_band)


def mndwi_s2(green_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """
    Bug J fix: MNDWI (Modified NDWI) on Sentinel-2.
    MNDWI = (Green - SWIR) / (Green + SWIR)
    Water has high MNDWI values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mndwi = (green_band - swir_band) / (green_band + swir_band + 1e-8)
    # Typical threshold for water: MNDWI > 0.0
    return (mndwi > 0.0).astype(np.uint8)


def run_single_chip(chip_path: str, label_path: str) -> dict:
    """Run all baselines on a single chip and return metrics."""
    with rasterio.open(chip_path) as src:
        bands = src.read()
    with rasterio.open(label_path) as src:
        label = src.read(1)

    results = {}

    # Otsu on VV (band 0)
    vv_pred = otsu_vv(bands[0])
    results['otsu_vv'] = compute_metrics(vv_pred, label)

    # If S2 bands available (bands 3=Green, 4=NIR, 11=SWIR for S2)
    if bands.shape[0] >= 12:
        s2_pred = mndwi_s2(bands[3], bands[11])
        results['mndwi_s2'] = compute_metrics(s2_pred, label)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Single chip path')
    parser.add_argument('--label', type=str, help='Single label path')
    parser.add_argument('--input_dir', type=str, help='Directory of chips')
    parser.add_argument('--label_dir', type=str, help='Directory of labels')
    args = parser.parse_args()

    if args.input and args.label:
        results = run_single_chip(args.input, args.label)
        print("\n=== Baseline Results ===")
        for method, metrics in results.items():
            print(f"\n{method}:")
            print(f"  IoU:        {metrics['iou']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1:         {metrics['f1']:.4f}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    elif args.input_dir and args.label_dir:
        chip_dir = Path(args.input_dir)
        label_dir = Path(args.label_dir)
        chip_files = sorted(chip_dir.glob('*.tif'))

        all_results = {'otsu_vv': [], 'mndwi_s2': []}
        for chip_path in chip_files:
            label_path = label_dir / chip_path.name
            if not label_path.exists():
                continue
            results = run_single_chip(str(chip_path), str(label_path))
            for method, metrics in results.items():
                all_results[method].append(metrics)

        print("\n=== Aggregate Baseline Results ===")
        for method, metrics_list in all_results.items():
            if not metrics_list:
                continue
            mean_iou = np.mean([m['iou'] for m in metrics_list])
            std_iou = np.std([m['iou'] for m in metrics_list])
            mean_f1 = np.mean([m['f1'] for m in metrics_list])
            mean_precision = np.mean([m['precision'] for m in metrics_list])
            mean_recall = np.mean([m['recall'] for m in metrics_list])
            print(f"\n{method} (n={len(metrics_list)}):")
            print(f"  IoU:        {mean_iou:.4f} ± {std_iou:.4f}")
            print(f"  Precision:  {mean_precision:.4f}")
            print(f"  Recall:     {mean_recall:.4f}")
            print(f"  F1:         {mean_f1:.4f}")


if __name__ == '__main__':
    main()
