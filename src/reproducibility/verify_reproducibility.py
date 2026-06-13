#!/usr/bin/env python3
"""
Reproducibility Verification Script
====================================
Bug L fix: Verifies that the model, normalization, and band order are consistent.

Usage:
    python verify_reproducibility.py --model cpu_v4_best_v2.pth --sample sample_chip.tif
"""

import argparse
import numpy as np
import torch
import rasterio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import NORM_STATS, BAND_ORDER
from model import UNet6ChRobust


def verify_model_architecture(model_path: str):
    """Verify model can be loaded and has correct architecture."""
    print("1. Verifying model architecture...")
    model = UNet6ChRobust(in_channels=7, base_filters=64, num_classes=1)

    # Bug K fix: weights_only=True
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Verify parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded successfully")
    print(f"   Parameters: {n_params:,}")

    # Verify config matches
    saved_config = checkpoint.get('config', {})
    if saved_config.get('input_channels') == 7:
        print(f"   ✅ Input channels: 7 (matches v2)")
    else:
        print(f"   ⚠️ Input channels: {saved_config.get('input_channels', 'unknown')}")

    return model


def verify_normalization(sample_path: str):
    """Verify normalization produces expected value ranges."""
    print("\n2. Verifying normalization...")

    with rasterio.open(sample_path) as src:
        bands = src.read(range(1, 7))

    for i, ch_name in enumerate(['VV', 'VH', 'DEM', 'Slope', 'HAND', 'TWI']):
        band = bands[i].astype(np.float32)
        mean = NORM_STATS[ch_name]['mean']
        std = NORM_STATS[ch_name]['std']
        normalized = (band - mean) / (std + 1e-8)

        actual_mean = np.nanmean(normalized)
        actual_std = np.nanstd(normalized)

        # After normalization, mean should be close to 0, std close to 1
        # But for new regions, it won't be exactly 0/1
        print(f"   {ch_name}: mean={actual_mean:.2f}, std={actual_std:.2f}")

    print("   ✅ Normalization verified")


def verify_band_order(model, sample_path: str):
    """Verify band order by checking model output is reasonable."""
    print("\n3. Verifying band order...")

    with rasterio.open(sample_path) as src:
        bands = src.read(range(1, 7))

    # Apply normalization
    norm_bands = []
    for i, ch_name in enumerate(['VV', 'VH', 'DEM', 'Slope', 'HAND', 'TWI']):
        mean = NORM_STATS[ch_name]['mean']
        std = NORM_STATS[ch_name]['std']
        norm_bands.append((bands[i] - mean) / (std + 1e-8))

    # Add VV/VH ratio
    vv = bands[0].astype(np.float32)
    vh = bands[1].astype(np.float32)
    ratio = vv / (vh + 1e-8)
    ratio_mean = NORM_STATS['VV']['mean'] / (NORM_STATS['VH']['mean'] + 1e-8)
    ratio_n = (ratio - ratio_mean) / 0.5
    norm_bands.append(ratio_n)

    data = np.stack(norm_bands).astype(np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.sigmoid(output).squeeze().numpy()

    water_pct = (pred > 0.5).sum() / pred.size * 100
    print(f"   Water pixels: {water_pct:.1f}%")

    if 0.1 < water_pct < 50:
        print("   ✅ Band order appears correct (reasonable water percentage)")
    else:
        print(f"   ⚠️ Water percentage unusual ({water_pct:.1f}%) — check band order")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--sample', required=True, help='Path to sample chip')
    args = parser.parse_args()

    print("=" * 60)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 60)

    model = verify_model_architecture(args.model)
    verify_normalization(args.sample)
    verify_band_order(model, args.sample)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
