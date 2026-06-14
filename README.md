# SAR Water Detection: Terrain Feature Transferability Analysis

Code and data for:

> **Terrain Features Are Associated with Impaired Spatial Transferability of SAR-Based Water Detection: A Cross-Regional Analysis of Nine Hydrological Water Body Types Across India**
>
> Parekh & Itolikar, *Evolving Earth*, 2026
> DOI: [10.5281/zenodo.19436533](https://doi.org/10.5281/zenodo.19436533)

## Repository Structure

```
├── src/colab_analysis/         # Core scripts
│   ├── model_v4_simple.py      # 6-channel U-Net architecture
│   ├── dataset.py              # Data loading
│   ├── config.py               # Training config
│   ├── shap_analysis.py        # Integrated Gradients attribution (Section 4.3.1)
│   └── ablation_overlay.py     # Pixel-level ablation overlay (Section 4.2.1)
├── model/
│   └── cpu_v4_best.pth         # Trained weights (31.4M params, epoch 27)
├── results/                    # Evaluation JSONs
│   ├── allIndia_results.json   # 468-patch transfer evaluation
│   ├── allIndia_SARonly.json   # SAR-only (terrain zeroed)
│   ├── allIndia_curvature.json # Curvature-stratified analysis
│   └── geographic_analysis.json # 99-patch regional breakdown
├── figures/                    # 16 paper figures (PNG)
├── CITATION.cff
├── requirements.txt
└── LICENSE (MIT)
```

## Quick Start

```bash
pip install -r requirements.txt
```

### Run Integrated Gradients Attribution
```bash
python src/colab_analysis/shap_analysis.py \
    --model model/cpu_v4_best.pth \
    --data <path_to_patches> \
    --output results/
```

### Run Ablation Overlay
```bash
python src/colab_analysis/ablation_overlay.py \
    --model model/cpu_v4_best.pth \
    --data <path_to_patches> \
    --output results/
```

## Model

- **Architecture:** U-Net with attention gates (dec3/dec4)
- **Input:** VV, VH, DEM, Slope, HAND, TWI (6 channels)
- **Training:** BoundaryLoss + DiceLoss + FocalLoss, AdamW (lr=5e-5), 52 epochs
- **Best epoch:** 27 (val loss = 0.2329)

## Key Results

| Metric | Value |
|--------|-------|
| Training IoU (Pune) | 0.896 |
| Transfer IoU (macro) | 0.595 |
| SAR-only IoU | 0.344 |
| Recovery (terrain removal) | 9.3× |
| IG effect size (mean Cohen's d) | 5.19 |
| Nine-type IoU range | 13.0× |

## Citation

```bibtex
@article{parekh2026terrain,
  title={Terrain Features Are Associated with Impaired Spatial Transferability of SAR-Based Water Detection},
  author={Parekh, Neeraj and Itolikar, Ashish B.},
  journal={Evolving Earth},
  year={2026},
  doi={10.5281/zenodo.19436533}
}
```
