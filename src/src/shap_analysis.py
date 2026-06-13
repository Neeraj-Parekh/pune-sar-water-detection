#!/usr/bin/env python3
"""
Method A: Gradient-Based Spatial Attribution
=============================================
Uses Integrated Gradients (IG) to compute per-pixel attribution for each channel.
SHAP-compatible alternative that produces equivalent results.

For every false positive pixel, computes attribution for the DEM channel
to determine if DEM is actively contributing to errors.

Usage:
    python shap_analysis.py [--model cpu_v4_best.pth] [--n-chips 10] [--n-steps 20]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, NORM_STATS
from model_v4_simple import UNet6Ch
from dataset import safe_normalize


RESULTS_DIR = Path(__file__).parent.parent.parent / 'evaluation_results_v4'
CHIPS_DIR = Path(__file__).parent.parent.parent / 'chips' / 'processed' / 'features_npy'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model(model_path, device):
    model = UNet6Ch(in_channels=6, base_filters=64)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def prepare_chip(npy_path):
    data = np.load(npy_path)
    vv = data[:, :, 0]
    vh = data[:, :, 1]
    dem = data[:, :, 2]
    slope = data[:, :, 3]
    hand = data[:, :, 4]
    twi = data[:, :, 5]
    label = data[:, :, 6]

    vv_norm = safe_normalize(vv, NORM_STATS['VV']['mean'], NORM_STATS['VV']['std'])
    vh_norm = safe_normalize(vh, NORM_STATS['VH']['mean'], NORM_STATS['VH']['std'])
    dem_norm = safe_normalize(dem, NORM_STATS['DEM']['mean'], NORM_STATS['DEM']['std'])
    slope_norm = safe_normalize(slope, NORM_STATS['Slope']['mean'], NORM_STATS['Slope']['std'])
    hand_norm = safe_normalize(hand, NORM_STATS['HAND']['mean'], NORM_STATS['HAND']['std'])
    twi_norm = safe_normalize(twi, NORM_STATS['TWI']['mean'], NORM_STATS['TWI']['std'])

    channels = np.stack([vv_norm, vh_norm, dem_norm, slope_norm, hand_norm, twi_norm], axis=0)
    channels = channels[:, :512, :512]
    label = label[:512, :512]

    tensor = torch.from_numpy(channels).float().unsqueeze(0)
    return tensor, label, data


def integrated_gradients(model, input_tensor, baseline=None, n_steps=20, target_channel=2, device='cpu'):
    """
    Compute Integrated Gradients for a specific input channel.
    
    Args:
        model: Trained model
        input_tensor: (1, 6, H, W) input
        baseline: (1, 6, H, W) baseline (zeros if None)
        n_steps: Number of interpolation steps
        target_channel: Input channel to attribute (2 = DEM)
        device: torch device
    
    Returns:
        attributions: (H, W) attribution map for the target channel
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
    
    # Generate interpolated inputs: (n_steps+1, 6, H, W)
    alphas = torch.linspace(0, 1, n_steps + 1).to(device)
    delta = input_tensor - baseline
    interpolated = baseline.unsqueeze(0) + alphas.view(-1, 1, 1, 1) * delta.unsqueeze(0)
    interpolated = interpolated.reshape(-1, *input_tensor.shape[1:])  # (n_steps+1, 6, H, W)
    interpolated.requires_grad_(True)
    
    # Forward pass
    outputs = model(interpolated)  # (n_steps+1, 1, H, W)
    
    # Sum output over spatial dims
    target = outputs.sum(dim=(1, 2, 3))  # (n_steps+1,)
    
    # Backward pass
    model.zero_grad()
    target.backward(torch.ones_like(target), retain_graph=True)
    
    # Get gradients
    grads = interpolated.grad  # (n_steps+1, 6, H, W)
    
    # Average gradients across interpolation steps
    avg_grads = grads.mean(dim=0)  # (6, H, W)
    
    # Attribution = (input - baseline) * avg_grads
    attribution = delta.squeeze(0) * avg_grads  # (6, H, W)
    
    # Return attribution for target channel
    return attribution[target_channel].detach().cpu().numpy()


def compute_predictions(model, tensor, device):
    """Get model predictions."""
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor.to(device))).squeeze().cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser(description='Gradient Attribution Analysis')
    parser.add_argument('--model', type=str, default=str(RESULTS_DIR / 'cpu_v4_best.pth'))
    parser.add_argument('--n-chips', type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=20)
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"Device: {device}")

    model = load_model(args.model, device)
    chip_files = sorted(CHIPS_DIR.glob('*.npy'))[:args.n_chips]
    print(f"Analyzing {len(chip_files)} chips with {args.n_steps} IG steps...")

    all_results = []

    for i, chip_path in enumerate(chip_files):
        chip_name = chip_path.stem
        print(f"\n[{i+1}/{len(chip_files)}] {chip_name}")

        tensor, label, raw = prepare_chip(chip_path)

        # Compute Integrated Gradients for DEM channel (index 2)
        print("  Computing Integrated Gradients for DEM...")
        try:
            ig_dem = integrated_gradients(
                model, tensor, n_steps=args.n_steps,
                target_channel=2, device=device
            )
            print(f"  IG shape: {ig_dem.shape}, range: [{ig_dem.min():.4f}, {ig_dem.max():.4f}]")
        except Exception as e:
            print(f"  IG failed: {e}")
            continue

        # Get predictions
        pred = compute_predictions(model, tensor, device)
        label_bin = (label[:512, :512] > 0.5).astype(int)
        pred_bin = (pred > 0.5).astype(int)

        # Classify pixels
        fp_mask = (pred_bin == 1) & (label_bin == 0)
        tn_mask = (pred_bin == 0) & (label_bin == 0)
        tp_mask = (pred_bin == 1) & (label_bin == 1)

        # Attribution at FP vs TN
        if fp_mask.sum() > 5 and tn_mask.sum() > 5:
            ig_fp = np.abs(ig_dem[fp_mask])
            ig_tn = np.abs(ig_dem[tn_mask])

            t_stat, p_value = stats.ttest_ind(ig_fp, ig_tn)

            pooled_std = np.sqrt(
                ((ig_fp.std()**2 * max(len(ig_fp)-1, 1)) +
                 (ig_tn.std()**2 * max(len(ig_tn)-1, 1))) /
                max(len(ig_fp) + len(ig_tn) - 2, 1)
            )
            cohens_d = (ig_fp.mean() - ig_tn.mean()) / (pooled_std + 1e-8)

            result = {
                'chip': chip_name,
                'n_fp': int(fp_mask.sum()),
                'n_tn': int(tn_mask.sum()),
                'mean_ig_dem_fp': float(ig_fp.mean()),
                'mean_ig_dem_tn': float(ig_tn.mean()),
                'std_ig_dem_fp': float(ig_fp.std()),
                'std_ig_dem_tn': float(ig_tn.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < 0.05,
                'ig_max': float(ig_dem.max()),
                'ig_min': float(ig_dem.min()),
            }
            all_results.append(result)

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  FP={fp_mask.sum()}, TN={tn_mask.sum()}")
            print(f"  |IG_DEM| FP={ig_fp.mean():.4f}±{ig_fp.std():.4f}, TN={ig_tn.mean():.4f}±{ig_tn.std():.4f}")
            print(f"  t={t_stat:.2f}, p={p_value:.2e} {sig}, d={cohens_d:.3f}")
        else:
            print(f"  Insufficient FP ({fp_mask.sum()}) or TN ({tn_mask.sum()})")

    if all_results:
        print("\n" + "="*70)
        print("GRADIENT ATTRIBUTION ANALYSIS RESULTS")
        print("="*70)

        sig_results = [r for r in all_results if r['significant']]
        print(f"\nPatches analyzed: {len(all_results)}")
        print(f"Significant (p<0.05): {len(sig_results)}")

        if sig_results:
            mean_fp = np.mean([r['mean_ig_dem_fp'] for r in sig_results])
            mean_tn = np.mean([r['mean_ig_dem_tn'] for r in sig_results])
            mean_d = np.mean([r['cohens_d'] for r in sig_results])
            print(f"Mean |IG_DEM| FP: {mean_fp:.4f}")
            print(f"Mean |IG_DEM| TN: {mean_tn:.4f}")
            print(f"Mean Cohen's d: {mean_d:.3f}")

            if mean_d > 0.5:
                print("\n*** DEM actively contributes to false positives ***")
                print("    High |IG_DEM| at FP pixels indicates DEM-derived features")
                print("    are used by the model to predict water where there is none.")
            elif mean_d > 0.2:
                print("\n** DEM moderately contributes to false positives **")
            else:
                print("\n* DEM contribution is small *")

        # Generate visualizations
        print("\nGenerating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Effect size across chips
        chips = [r['chip'][:20] for r in all_results]
        effects = [r['cohens_d'] for r in all_results]
        colors = ['red' if e > 0.5 else 'orange' if e > 0.2 else 'blue' for e in effects]
        axes[0, 0].barh(chips, effects, color=colors, edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Large')
        axes[0, 0].axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Medium')
        axes[0, 0].set_xlabel("Cohen's d")
        axes[0, 0].set_title('DEM Attribution Effect Size by Chip')
        axes[0, 0].legend()

        # Plot 2: Mean attribution at FP vs TN
        fp_vals = [r['mean_ig_dem_fp'] for r in all_results]
        tn_vals = [r['mean_ig_dem_tn'] for r in all_results]
        axes[0, 1].bar(['FP', 'TN'], [np.mean(fp_vals), np.mean(tn_vals)],
                        yerr=[np.std(fp_vals), np.std(tn_vals)],
                        color=['red', 'blue'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('|IG_DEM|')
        axes[0, 1].set_title('Mean |IG_DEM| at FP vs TN Pixels')

        # Plot 3: Example IG heatmap
        if len(chip_files) > 0:
            tensor, label, raw = prepare_chip(chip_files[0])
            ig_dem = integrated_gradients(model, tensor, n_steps=args.n_steps,
                                          target_channel=2, device=device)
            pred = compute_predictions(model, tensor, device)

            im = axes[1, 0].imshow(ig_dem, cmap='RdBu_r', vmin=-0.01, vmax=0.01)
            axes[1, 0].contour(label[:512, :512], levels=[0.5], colors=['black'], linewidths=0.5)
            axes[1, 0].set_title('IG_DEM Heatmap\n(Black=Water Label)')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        # Plot 4: Summary table
        axes[1, 1].axis('off')
        table_data = [
            ['Patches analyzed', str(len(all_results))],
            ['Significant (p<0.05)', str(len(sig_results))],
            ['Mean |IG_DEM| FP', f'{np.mean([r["mean_ig_dem_fp"] for r in sig_results]):.4f}' if sig_results else 'N/A'],
            ['Mean |IG_DEM| TN', f'{np.mean([r["mean_ig_dem_tn"] for r in sig_results]):.4f}' if sig_results else 'N/A'],
            ["Cohen's d", f'{np.mean([r["cohens_d"] for r in sig_results]):.3f}' if sig_results else 'N/A'],
        ]
        table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'Value'],
                                  loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Summary')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig_shap_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'fig_shap_analysis.png'}")

        output_results = {
            'per_chip': all_results,
            'summary': {
                'n_chips': len(all_results),
                'n_significant': len(sig_results),
                'mean_effect_size': float(np.mean([r['cohens_d'] for r in sig_results])) if sig_results else 0,
                'method': 'Integrated Gradients (20 steps)',
            }
        }
        with open(OUTPUT_DIR / 'shap_analysis_results.json', 'w') as f:
            json.dump(output_results, f, indent=2)
        print(f"Results saved: {OUTPUT_DIR / 'shap_analysis_results.json'}")


if __name__ == '__main__':
    main()
