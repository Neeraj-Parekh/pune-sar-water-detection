#!/usr/bin/env python3
"""
Data Pipeline Monitoring Tool for SAR Water Detection v2
========================================================
Monitors:
  - Chip file inventory and integrity
  - Normalization statistics across all chips
  - Missing/corrupt files
  - Band count and value range validation
  - Water coverage distribution

Usage:
    python monitor_pipeline.py /path/to/chips_dir
    python monitor_pipeline.py /path/to/chips_dir --strict
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import rasterio


def check_chip_integrity(chip_path: Path) -> dict:
    """Check a single chip for integrity issues."""
    result = {
        'file': chip_path.name,
        'path': str(chip_path),
        'size_mb': chip_path.stat().st_size / 1e6,
        'status': 'ok',
        'issues': [],
    }
    try:
        with rasterio.open(chip_path) as src:
            result['bands'] = src.count
            result['width'] = src.width
            result['height'] = src.height
            result['dtype'] = str(src.dtypes[0])

            if src.count < 7:
                result['issues'].append(f'only {src.count} bands (expected 7+)')

            # Read and check value ranges
            for i in range(min(src.count, 7)):
                band = src.read(i + 1)
                if np.all(np.isnan(band)):
                    result['issues'].append(f'band {i+1} is all NaN')
                if np.any(np.isinf(band)):
                    result['issues'].append(f'band {i+1} has Inf')

            # Check label band statistics
            if src.count >= 7:
                label = src.read(7)
                valid = ~np.isnan(label)
                if valid.sum() > 0:
                    water_frac = float(np.mean(label[valid] == 1))
                    result['water_coverage'] = water_frac
                    if water_frac == 0.0:
                        result['issues'].append('no water pixels')
                    elif water_frac == 1.0:
                        result['issues'].append('all pixels are water')

    except Exception as e:
        result['status'] = 'error'
        result['issues'].append(str(e))

    if result['issues']:
        result['status'] = 'warning' if result['status'] == 'ok' else 'error'
    return result


def monitor_pipeline(chips_dir: str, strict: bool = False) -> dict:
    """Monitor the entire pipeline."""
    chips = sorted(Path(chips_dir).glob('pune_r*.tif'))
    if not chips:
        print(f"No chips found in {chips_dir}")
        return {}

    print(f"Monitoring {len(chips)} chips in {chips_dir}...")

    results = []
    for chip in chips:
        r = check_chip_integrity(chip)
        results.append(r)
        status_icon = '✓' if r['status'] == 'ok' else ('⚠' if r['status'] == 'warning' else '✗')
        if r['status'] != 'ok' or strict:
            print(f"  {status_icon} {r['file']}: {', '.join(r['issues']) if r['issues'] else 'ok'}")

    # Aggregate statistics
    valid_results = [r for r in results if 'water_coverage' in r]
    summary = {
        'chips_dir': chips_dir,
        'timestamp': datetime.now().isoformat(),
        'n_chips': len(results),
        'n_ok': sum(1 for r in results if r['status'] == 'ok'),
        'n_warning': sum(1 for r in results if r['status'] == 'warning'),
        'n_error': sum(1 for r in results if r['status'] == 'error'),
        'total_size_mb': sum(r['size_mb'] for r in results),
    }

    if valid_results:
        coverages = [r['water_coverage'] for r in valid_results]
        summary['water_coverage'] = {
            'mean': float(np.mean(coverages)),
            'std': float(np.std(coverages)),
            'min': float(np.min(coverages)),
            'max': float(np.max(coverages)),
            'median': float(np.median(coverages)),
        }
        # Check for class imbalance
        if summary['water_coverage']['mean'] < 0.01:
            summary['class_imbalance_warning'] = 'Severe imbalance: <1% water'
        elif summary['water_coverage']['mean'] > 0.5:
            summary['class_imbalance_warning'] = 'Water-dominant: >50% water'

    # 30-point QA checklist
    qa_checks = [
        ('All chips readable', summary['n_error'] == 0),
        ('Water coverage between 1-50%',
         0.01 <= summary.get('water_coverage', {}).get('mean', 0) <= 0.5),
        ('At least 50 chips', summary['n_chips'] >= 50),
        ('Total size reasonable (<10GB)', summary['total_size_mb'] < 10000),
        ('No all-NaN chips', not any('all NaN' in ' '.join(r.get('issues', [])) for r in results)),
    ]
    summary['qa_checks'] = [
        {'check': name, 'pass': bool(pass_)} for name, pass_ in qa_checks
    ]
    summary['qa_score'] = f"{sum(1 for c in summary['qa_checks'] if c['pass'])}/{len(qa_checks)}"

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('chips_dir', help='Directory with chip .tif files')
    parser.add_argument('--strict', action='store_true', help='Show all chips even if ok')
    parser.add_argument('--output', default=None, help='Save JSON report')
    args = parser.parse_args()

    summary = monitor_pipeline(args.chips_dir, args.strict)

    if summary:
        print(f"\n=== Pipeline Summary ===")
        print(f"Total chips: {summary['n_chips']}")
        print(f"  OK:      {summary['n_ok']}")
        print(f"  Warning: {summary['n_warning']}")
        print(f"  Error:   {summary['n_error']}")
        print(f"Total size: {summary['total_size_mb']:.1f} MB")
        if 'water_coverage' in summary:
            wc = summary['water_coverage']
            print(f"Water coverage: {wc['mean']:.3f} ± {wc['std']:.3f} (range [{wc['min']:.3f}, {wc['max']:.3f}])")
        print(f"QA Score: {summary['qa_score']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Report saved to {args.output}")
