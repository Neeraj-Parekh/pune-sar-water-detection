# SAR Water Detection: Terrain Feature Transferability Analysis

Code and data for the paper:

> **Terrain Features Are Associated with Impaired Spatial Transferability of SAR-Based Water Detection: A Cross-Regional Analysis of Nine Hydrological Water Body Types Across India**
>
> Neeraj Parekh and Ashish B. Itolikar
> *Evolving Earth*, 2026
> DOI: [10.5281/zenodo.19436533](https://doi.org/10.5281/zenodo.19436533)

## Repository Structure

```
├── src/                          # Source code
│   ├── model_v4_simple.py        # 6-channel U-Net architecture
│   ├── dataset.py                # Data loading utilities
│   ├── config.py                 # Training configuration
│   ├── train_pune_unet6ch_cpu_v4_robust.py  # Training script
│   ├── inference.py              # Inference script
│   └── colab_analysis/           # Colab-compatible analysis scripts
│       ├── shap_analysis.py      # Integrated Gradients attribution
│       ├── ablation_overlay.py   # Pixel-level ablation overlay
│       └── elsevier_config.py    # Figure formatting (Okabe-Ito, Arial)
├── model/                        # Trained model weights
│   └── cpu_v4_best.pth           # Best checkpoint (31.4M params, epoch 27)
├── results/                      # Evaluation JSONs
│   ├── allIndia_results.json     # Full 468-patch evaluation
│   ├── allIndia_SARonly.json     # SAR-only (terrain zeroed)
│   ├── allIndia_curvature.json   # Curvature-stratified analysis
│   ├── geographic_analysis.json  # Regional breakdown (99 patches)
│   └── benchmark_comparison.json # Comparison with literature
├── figures/                      # Paper figures (PNG)
├── chips_sample/                 # Sample chips (download from Google Drive)
├── reproducibility/              # Reproducibility scripts
├── CITATION.cff                  # How to cite
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Inference

```bash
python src/inference.py \
    --model model/cpu_v4_best.pth \
    --input <path_to_sar_6band.tif> \
    --output <output_dir>
```

### 3. Reproduce Key Results

```bash
# Evaluate on all-India test set
python src/evaluate.py --model model/cpu_v4_best.pth --data data/

# Run curvature analysis
python src/curvature_analysis.py --results results/

# Run Integrated Gradients attribution
python src/colab_analysis/shap_analysis.py --model model/cpu_v4_best.pth --data data/
```

## Model Architecture

- **Architecture:** 6-channel U-Net with attention gates at dec3/dec4
- **Input channels:** VV, VH, DEM, Slope, HAND, TWI
- **Output:** Binary water mask
- **Parameters:** 31.4M
- **Training:** BoundaryLoss + DiceLoss + FocalLoss, AdamW (lr=5e-5), 52 epochs

## Key Results

| Metric | Value |
|--------|-------|
| Training IoU (Pune) | 0.896 |
| Transfer IoU (macro, 9 regions) | 0.595 |
| SAR-only IoU | 0.344 |
| Recovery with terrain removal | 9.3× |
| Terrain curvature OR reversal | 6.3× |
| Nine-type IoU range | 13.0× |

## Data

- **Training:** Pune Metropolitan Region (6,348 km²), 224 patches
- **Testing:** 468 patches across 5 independently curated datasets covering India
- **Sample chips:** Download from [Google Drive](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

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

## License

MIT License
