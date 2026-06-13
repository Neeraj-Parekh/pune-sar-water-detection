#!/usr/bin/env python3
"""
Method C: Controlled Experiment Visualization
==============================================
Demonstrates that terrain features impair transfer in flat regions
but help in Pune (training domain) — the "crossover" pattern.

Reviewer 2's challenge: "Could the problem be general SAR confusion
in dry alluvial terrain, rather than specifically DEM-induced error?"

This script shows: SAR-only vs terrain-fused IoU per region,
identifying where terrain helps vs hurts — the causal evidence.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent.parent / 'evaluation_results_v4'
OUTPUT_DIR = Path(__file__).parent.parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load existing evaluation results."""
    with open(RESULTS_DIR / 'geographic_analysis.json') as f:
        geo = json.load(f)
    with open(RESULTS_DIR / 'allIndia_results.json') as f:
        terrain = json.load(f)
    with open(RESULTS_DIR / 'allIndia_SARonly.json') as f:
        sar_only = json.load(f)
    return geo, terrain, sar_only


def compute_crossover(geo_data):
    """
    Compute per-region terrain impact.
    Positive delta = terrain hurts (SAR-only better).
    Negative delta = terrain helps (terrain-fused better).
    """
    regions = []
    for r in geo_data['regional_analysis']:
        terrain_iou = r['terrain_iou']
        sar_iou = r['sar_only_iou']
        delta = sar_iou - terrain_iou  # positive = terrain hurts
        regions.append({
            'region': r['region'],
            'n_chips': r['n_chips'],
            'terrain_iou': terrain_iou,
            'sar_iou': sar_iou,
            'delta': delta,
            'improvement': r['improvement'],
            'physics': r['physics']
        })
    return regions


def plot_crossover(regions, output_path):
    """
    Scatter plot: SAR-only IoU vs terrain-fused IoU per region.
    Points above y=x line = terrain hurts.
    Points below y=x line = terrain helps.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Crossover line (y = x)
    ax.plot([0, 0.8], [0, 0.8], 'k--', alpha=0.5, label='y = x (no terrain effect)')

    # Color code by terrain type
    colors = {
        'Central (Deccan/Plains)': '#2ecc71',  # Green = terrain helps
        'East (Deltas/Estuaries)': '#3498db',
        'South (Coastal/Backwaters)': '#9b59b6',
        'West (Reservoirs/Rivers)': '#e67e22',
        'North (Himalaya/Kashmir)': '#e74c3c',  # Red = terrain hurts badly
        'Other': '#95a5a6'
    }

    for r in regions:
        color = colors.get(r['region'], '#95a5a6')
        size = r['n_chips'] * 20  # Scale by number of chips
        ax.scatter(r['sar_iou'], r['terrain_iou'], c=color, s=size,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(r['region'].split('(')[0].strip(),
                    (r['sar_iou'], r['terrain_iou']),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, fontweight='bold')

    # Shade regions
    ax.fill_between([0, 0.8], [0, 0.8], [0.8, 0.8],
                    alpha=0.05, color='red', label='Terrain hurts (above y=x)')
    ax.fill_between([0, 0.8], [0, 0], [0, 0.8],
                    alpha=0.05, color='green', label='Terrain helps (below y=x)')

    ax.set_xlabel('SAR-Only IoU (terrain channels zeroed)', fontsize=12)
    ax.set_ylabel('Terrain-Fused IoU (6-channel)', fontsize=12)
    ax.set_title('Crossover: Terrain Features Help in Pune\nbut Hurt in Flat Regions', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(-0.02, 0.82)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_comparison(regions, output_path):
    """
    Bar chart: SAR-only vs terrain-fused IoU per region.
    Shows the crossover pattern clearly.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    region_names = [r['region'].split('(')[0].strip() for r in regions]
    terrain_ious = [r['terrain_iou'] for r in regions]
    sar_ious = [r['sar_iou'] for r in regions]

    x = np.arange(len(region_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, terrain_ious, width, label='Terrain-Fused (6-Ch)',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, sar_ious, width, label='SAR-Only (VV+VH)',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add crossover arrows
    for i, r in enumerate(regions):
        if r['delta'] > 0.1:  # Terrain hurts significantly
            ax.annotate('', xy=(i + width/2, r['sar_iou']),
                        xytext=(i - width/2, r['terrain_iou']),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            ax.text(i, max(r['sar_iou'], r['terrain_iou']) + 0.02,
                    f"+{r['delta']:.2f}", ha='center', fontsize=8, color='red')

    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Terrain Features Help in Training Domain (Pune)\nbut Hurt in Cross-Region Transfer', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(regions):
    """Print statistical summary."""
    print("\n" + "="*70)
    print("CONTROLLED EXPERIMENT: SAR-Only vs Terrain-Fused IoU")
    print("="*70)

    print(f"\n{'Region':<30} {'Terrain IoU':>12} {'SAR-Only IoU':>14} {'Δ IoU':>10} {'Effect':>10}")
    print("-"*70)

    for r in regions:
        effect = "HELPS" if r['delta'] < 0 else "HURTS"
        print(f"{r['region']:<30} {r['terrain_iou']:>12.4f} {r['sar_iou']:>14.4f} {r['delta']:>+10.4f} {effect:>10}")

    # Count crossover regions
    helps = sum(1 for r in regions if r['delta'] < 0)
    hurts = sum(1 for r in regions if r['delta'] > 0)
    print(f"\nCrossover: {helps} regions where terrain helps, {hurts} regions where terrain hurts")
    print("This demonstrates that terrain features are domain-specific confounders,")


def main():
    """Main entry point."""
    print("Loading data...")
    geo, terrain, sar_only = load_data()

    print("Computing crossover analysis...")
    regions = compute_crossover(geo)

    print_summary(regions)

    print("\nGenerating figures...")
    plot_crossover(regions, OUTPUT_DIR / 'fig_crossover_scatter.png')
    plot_bar_comparison(regions, OUTPUT_DIR / 'fig_crossover_bars.png')

    # Save results to JSON
    results = {
        'regions': regions,
        'summary': {
            'total_regions': len(regions),
            'terrain_helps': sum(1 for r in regions if r['delta'] < 0),
            'terrain_hurts': sum(1 for r in regions if r['delta'] > 0),
            'max_hurt': max(regions, key=lambda r: r['delta'])['region'],
            'max_help': min(regions, key=lambda r: r['delta'])['region']
        }
    }
    with open(OUTPUT_DIR / 'controlled_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT_DIR / 'controlled_experiment_results.json'}")


if __name__ == '__main__':
    main()
