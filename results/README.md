# Evaluation Outputs

This directory holds the JSON outputs of the model evaluations discussed in the
manuscript. The 99-chip geographic stress test was run more than once: the
first evaluation used the band conventions of the original analysis scripts,
and a later re-evaluation applied corrected conventions and rebuilt the HAND
and TWI layers with the training recipes. The manuscript reports the
re-evaluated numbers (the convention-sensitivity and geographic-stratification
tables); the files below are preserved so that both stages of the analysis can
be inspected.

| File | Contents | Status |
| --- | --- | --- |
| `training_state.json` | Pune training run: best checkpoint at epoch 52, validation IoU 0.8955 | Reported (training details) |
| `benchmark_comparison.json` | Cross-study comparison; this study: train 0.8955, transfer 0.595, degradation 1.51× | Reported (cross-study table) |
| `allIndia_curvature.json` | Curvature-stratified odds-ratio analysis of false positives | Reported (curvature tables) |
| `geographic_analysis.json` | 99-chip regional breakdown of the first stress-test evaluation | Superseded (legacy conventions) |
| `allIndia_results.json` | First 99-chip six-channel evaluation (mean IoU 0.037 as published at the time) | Superseded (legacy conventions) |
| `allIndia_SARonly.json` | First SAR-only evaluation of the same chips (mean IoU 0.344) | Superseded (legacy conventions) |
| `evaluation_metrics.json` | 45-chip Pune hold-out under the earlier evaluation conventions (IoU 0.524) | Superseded (legacy conventions) |
| `tif_inference_results.json` | 27-chip GeoTIFF inference experiment (IoU 0.077) | Exploratory |

The convention-sensitivity re-evaluation of the 99-chip stress set (SAR-only
0.544; HAND/TWI rebuilt with the training recipes 0.460; independently derived
terrain products 0.001–0.02) is included in the Zenodo archive
([10.5281/zenodo.22834407](https://doi.org/10.5281/zenodo.22834407)) together
with the evaluation scripts and the chip set itself, so those headline numbers
can be reproduced directly from the archive.
