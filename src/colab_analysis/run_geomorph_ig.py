#!/usr/bin/env python3
"""
Runs Integrated Gradients analysis on specific geomorphological classes
using the updated shap_analysis.py, and compiles a summary report.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Geomorphological mapping of prefixes
geomorph_groups = {
    "Floods": [
        "gold_enhanced_assam_flood",
        "gold_enhanced_bihar_flood"
    ],
    "Rivers": [
        "gold_enhanced_ganges_patna",
        "gold_enhanced_ganges_varanasi",
        "gold_enhanced_godavari_rajahmundry",
        "gold_enhanced_kosi_river",
        "gold_enhanced_yamuna_delhi"
    ],
    "Himalayan": [
        "gold_enhanced_himalayan_river"
    ],
    "Coastal/Deltas": [
        "gold_enhanced_mumbai_harbor",
        "gold_enhanced_sundarbans"
    ]
}

def main():
    results_compiled = []
    
    print("==========================================================")
    print("GEOMORPHOLOGICAL INTEGRATED GRADIENS RUNNER")
    print("==========================================================\n")
    
    for category, prefixes in geomorph_groups.items():
        print(f"\n>>> Running Group: {category}")
        print("-" * 50)
        for prefix in prefixes:
            suffix = prefix.replace("gold_enhanced_", "")
            print(f"Executing prefix filter: {prefix} (suffix: {suffix})...")
            
            # Execute shap_analysis.py via subprocess
            cmd = [
                sys.executable,
                str(BASE_DIR / "shap_analysis.py"),
                "--chip-prefix", prefix,
                "--n-chips", "20",
                "--n-steps", "30",
                "--output-suffix", suffix
            ]
            
            subprocess.run(cmd, check=True)
            
            # Read generated json output
            json_path = FIGURES_DIR / f"shap_analysis_results_{suffix}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    
                    for chip_res in data.get('per_chip', []):
                        results_compiled.append({
                            'Category': category,
                            'Chip': chip_res['chip'],
                            'FPs': chip_res['n_fp'],
                            'TNs': chip_res['n_tn'],
                            'IG_DEM_FP': chip_res['mean_ig_dem_fp'],
                            'IG_DEM_TN': chip_res['mean_ig_dem_tn'],
                            'Cohens_d': chip_res['cohens_d'],
                            'Significant': "Yes" if chip_res['significant'] else "No"
                        })
                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
            else:
                print(f"  Note: No output JSON found at {json_path} (likely insufficient FPs/TNs).")

    # Generate Markdown Table Report
    if results_compiled:
        report_lines = [
            "# Integrated Gradients DEM Attribution by Geomorphological Category\n",
            "| Category | Chip | FPs | TNs | Mean |IG_DEM| FP | Mean |IG_DEM| TN | Cohen's d | Significant |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        for r in results_compiled:
            line = f"| {r['Category']} | {r['Chip']} | {r['FPs']} | {r['TNs']} | {r['IG_DEM_FP']:.4f} | {r['IG_DEM_TN']:.4f} | {r['Cohens_d']:.3f} | {r['Significant']} |"
            report_lines.append(line)
            
        report_content = "\n".join(report_lines)
        
        # Save Report
        report_path = FIGURES_DIR / "geomorph_ig_summary.md"
        with open(report_path, "w") as f:
            f.write(report_content)
            
        print("\n" + "="*70)
        print("CONSOLIDATED GEOMORPHOLOGICAL SUMMARY REPORT")
        print("="*70)
        print(report_content)
        print("\nReport saved to:", report_path)
    else:
        print("\nNo results were compiled (insufficient false positive data across all run chips).")

if __name__ == "__main__":
    main()
