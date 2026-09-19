# Terrain Features and SAR Water Detection Transferability Across India

Code and results for:

> **Influence of Terrain Features on the Spatial Transferability of Sentinel-1 SAR Water Body Detection Across India: A Geomorphological Analysis**
>
> Neeraj Parekh¹ and Ashish B. Itolikar²
>
> ¹ Department of Electronics and Telecommunication Engineering, MIT Academy of Engineering, Alandi, Pune, India
> ² Department of Humanities and Engineering Sciences, MIT Academy of Engineering, Alandi, Pune, India
>
> Manuscript submitted to the *Journal of the Indian Society of Remote Sensing*.

The full code and results archive, the trained model weights, and the 99-chip geographic stress-test set are deposited on Zenodo: **[10.5281/zenodo.22834407](https://doi.org/10.5281/zenodo.22834407)**.

## Summary

A six-channel U-Net (VV, VH, DEM, Slope, HAND, TWI) trained on Sentinel-1 GRD patches over the Pune region achieves a held-out IoU of 0.896. When the same model is applied to patches drawn from across India, performance depends strongly on how the terrain channels are derived:

| Evaluation | Mean IoU |
| --- | --- |
| Pune held-out test set | 0.896 |
| 468-patch cross-dataset transfer | 0.595 (1.51× degradation) |
| 99-chip geographic stress set, terrain channels zeroed (SAR-only) | 0.544 |
| 99-chip geographic stress set, HAND/TWI rebuilt with the training recipes | 0.460 |
| 99-chip geographic stress set, independently derived terrain products | 0.001–0.02 |

The manuscript draws two conclusions from this pattern: terrain-fused inference is fragile to the choice of terrain product and recipe, while SAR-only inference is a stable fallback; and geomorphological stratification of the test patches reveals a large, structured performance range that a single aggregate score conceals.

## Repository Structure

```
├── src/colab_analysis/          # Analysis scripts
│   ├── unet6ch_model.py         # Six-channel U-Net architecture
│   ├── dataset.py               # Normalisation and filtering helpers
│   ├── config.py                # Shared configuration and Pune statistics
│   ├── shap_analysis.py         # Integrated Gradients attribution (Section 4.3.1)
│   └── ablation_overlay.py      # Pixel-level terrain ablation overlay (Section 4.2.1)
├── model/
│   └── pune_unet6ch_weights.pth # Trained weights (31.4 M parameters, 376 MB)
├── results/                     # Evaluation outputs (JSON), see results/README.md
├── figures/                     # The 16 manuscript figures (PNG)
├── CITATION.cff
├── LICENSE                      # MIT
├── requirements.txt
└── README.md
```

The model weights are stored with Git LFS; a plain `git clone` fetches them automatically.

## Quick Start

```bash
pip install -r requirements.txt
```

Both analysis scripts expect the evaluation chips as `.npy` arrays of shape 513 × 513 × 7 (six input channels followed by the label channel); the 99-chip stress set in the Zenodo archive uses this format. Copy or link the chips into `src/colab_analysis/chips/`, or pass `--chip-dir`.

### Integrated Gradients attribution (Section 4.3.1)

```bash
python src/colab_analysis/shap_analysis.py \
    --model model/pune_unet6ch_weights.pth \
    --chip-dir <path_to_chips> \
    --n-chips 10
```

### Pixel-level terrain ablation overlay (Section 4.2.1)

```bash
python src/colab_analysis/ablation_overlay.py \
    --model model/pune_unet6ch_weights.pth \
    --n-chips 20
```

This script reads chips from `src/colab_analysis/chips/` by default.

## Model and Training

- **Architecture:** U-Net with GroupNorm and attention gates on decoder stages 3 and 4; six input channels (VV, VH, DEM, Slope, HAND, TWI); 31.4 M parameters.
- **Optimisation:** Adam (learning rate 5 × 10⁻⁵, batch size 8, up to 60 epochs, 5 warm-up epochs followed by exponential decay, no weight decay, gradient clipping at norm 1.0); best checkpoint at epoch 52 with validation IoU 0.8955.
- **Loss:** Focal BCE (α = 0.25, γ = 2.0) weighted 0.6 plus Dice weighted 0.4.
- **Normalisation:** per-channel z-scores using Pune training statistics (VV −9.08 ± 4.14, VH −16.33 ± 3.97, DEM 666.74 ± 145.62, Slope 8.05 ± 8.88, HAND 46.61 ± 76.19, TWI 10.55 ± 2.39).

A note on channel conventions: the exporter that produced the training patches stored the HAND band before Slope, whereas the training loader read band 3 as Slope and band 4 as HAND. The released model is the model as trained under this convention, and the manuscript reports this together with a convention-sensitivity re-evaluation of the stress test. Readers reproducing the evaluation should follow `results/README.md` for which outputs use the legacy convention and which use the corrected one.

## Data

Sentinel-1 GRD (IW, DV) backscatter with terrain layers derived from SRTM and MERIT Hydro; water labels follow the JRC Global Surface Water occurrence layer, with a VH threshold rule applied during label fusion. Patches were exported with Google Earth Engine at 513 × 513 pixels (approximately 5 km per side), with the central 512 × 512 pixels fed to the network. The 99-chip geographic stress set used in the convention re-evaluation is included in the Zenodo archive.

## Citation

```bibtex
@misc{parekh2026punesar,
  author       = {Parekh, Neeraj and Itolikar, Ashish B.},
  title        = {Influence of Terrain Features on the Spatial Transferability of
                  Sentinel-1 SAR Water Body Detection Across India:
                  A Geomorphological Analysis},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22834407},
  url          = {https://doi.org/10.5281/zenodo.22834407}
}
```

## License

Released under the MIT License; see [LICENSE](LICENSE).
