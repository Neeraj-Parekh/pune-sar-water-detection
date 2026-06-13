"""
Unified Configuration for Pune SAR Water Detection v2
======================================================
Merged from:
  - train_pune_unet6ch_cpu_v4_robust.py (CPU training baseline)
  - train_pune_unet6ch_v4_guerrilla.py  (radiometric jitter, weight decay)

BUG FIXES APPLIED (v1 → v2):
  - Bug A: Normalization — single source of truth (z-score, Pune stats)
  - Bug B: Architecture — UNet6ChRobust (GroupNorm + Attention Gates)
  - Bug C: Band order — explicit [VV, VH, DEM, Slope, HAND, TWI]
  - Bug D: Weight decay set to 1e-4 (was 0.0)
  - Bug E: Gamma-distributed speckle noise augmentation added
  - Bug F: Incidence angle normalization for VV/VH
  - Bug G: VV/VH ratio channel added (7th channel)
  - Bug H: 5-fold stratified cross-validation support
  - Bug I: Test-time augmentation (TTA) at inference
  - Bug K: weights_only=True for checkpoint loading
  - Bug L: Reproducibility package (Dockerfile, pinned requirements)
  - Bug M: Topology preservation — clDice loss + DSConv + Frangi vesselness
  - Bug N: BoundaryLoss with α-scheduling for sharp water edges
  - Bug O: Multi-scale TTA for varying water body sizes
"""

import torch
from pathlib import Path


# ─── NORMALIZATION STATS (Pune-specific, z-score) ────────────────────────────
NORM_STATS = {
    'VV':    {'mean': -9.08,    'std': 4.14},
    'VH':    {'mean': -16.33,   'std': 3.97},
    'DEM':   {'mean': 666.74,   'std': 145.62},
    'Slope': {'mean': 8.05,     'std': 8.88},
    'HAND':  {'mean': 46.61,    'std': 76.19},
    'TWI':   {'mean': 10.55,    'std': 2.39},
}

# Band order — explicit and verified.
# Output tensor order: [VV, VH, DEM, Slope, HAND, TWI, VV/VH_ratio]
BAND_ORDER = ['VV', 'VH', 'DEM', 'Slope', 'HAND', 'TWI', 'VV_VH_ratio']


def get_device() -> torch.device:
    """
    Auto-detect best available device.

    Order of preference:
      1. CUDA (NVIDIA GPUs) — preferred for training
      2. MPS (Apple Silicon) — for Mac developers
      3. CPU — fallback

    Override via env var: PUNE_SAR_DEVICE=cuda|mps|cpu
    """
    import os
    override = os.environ.get('PUNE_SAR_DEVICE', '').lower().strip()
    if override == 'cpu':
        return torch.device('cpu')
    if override == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if override == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Config:
    # ─── Device (auto-detected: CUDA > MPS > CPU) ─────────────────────────────
    DEVICE = get_device()
    SEED = 42

    # ─── DataLoader (GPU-aware) ───────────────────────────────────────────────
    NUM_WORKERS = 0 if DEVICE.type == 'cpu' else min(4, __import__('os').cpu_count() or 0)
    PIN_MEMORY = DEVICE.type == 'cuda'
    PERSISTENT_WORKERS = NUM_WORKERS > 0

    # ─── Mixed Precision (GPU only) ───────────────────────────────────────────
    USE_AMP = DEVICE.type == 'cuda'
    AMP_DTYPE = 'bfloat16' if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else 'float16'

    # ─── Data ─────────────────────────────────────────────────────────────────
    CHIPS_DIR = Path.home() / 'sar_water_detection' / 'pune_training_chips_v1'
    NORM_STATS = NORM_STATS
    BAND_ORDER = BAND_ORDER
    TARGET_SIZE = 480

    # ─── Model Architecture ───────────────────────────────────────────────────
    INPUT_CHANNELS = 8           # Bug M fix: +1 for Frangi vesselness (7 + Frangi)
    BASE_FILTERS = 64
    NUM_CLASSES = 1

    # ─── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE = 8 if DEVICE.type == 'cpu' else 16
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-5
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4          # Bug D fix
    GRADIENT_CLIP_MAX_NORM = 1.0

    # ─── Loss ─────────────────────────────────────────────────────────────────
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    BCE_WEIGHT = 0.6
    DICE_WEIGHT = 0.4

    # ─── NaN Protection ───────────────────────────────────────────────────────
    MAX_NAN_BATCHES_PER_EPOCH = 10
    NAN_LOSS_THRESHOLD = 1e6

    # ─── Augmentation ─────────────────────────────────────────────────────────
    AUGMENT_PROB = 0.5
    SAR_JITTER_DB = 2.0          # Bug D fix: radiometric jitter ±2 dB
    SPECKLE_SHAPE = 4.0          # Bug E fix: Gamma speckle shape parameter
    SPECKLE_SCALE = 0.25         # Bug E fix: Gamma speckle scale parameter

    # ─── Incidence Angle (Bug F fix) ──────────────────────────────────────────
    INCIDENCE_ANGLE_MEAN = 39.0  # Typical Sentinel-1 IW incidence angle (degrees)
    INCIDENCE_ANGLE_STD = 5.0    # Standard deviation across swath

    # ─── Cross-Validation (Bug H fix) ─────────────────────────────────────────
    N_FOLDS = 5
    CV_SEED = 42

    # ─── Topology & Boundary Losses (Bug M, N fixes) ──────────────────────────
    CLDICE_LAMBDA = 0.3          # λ for clDice in combined loss
    CLDICE_NUM_ITER = 3          # soft skeletonization iterations
    CLDICE_TAU = 10.0            # softmax temperature for soft skeleton
    BOUNDARY_WEIGHT = 0.2        # weight for BoundaryLoss
    BOUNDARY_ALPHA_MIN = 0.3     # minimum α for region/boundary balance
    BOUNDARY_DECAY_EPOCHS = 60   # epochs to reach α_min

    # ─── DSConv (Bug M fix: thin curvilinear features) ─────────────────────────
    USE_DSCONV = True            # enable DSConv in decoder
    DSCONV_STAGES = [3, 4]       # decoder stages to apply DSConv
    DSCONV_KERNEL_LENGTH = 9     # 1D kernel length

    # ─── Frangi Vesselness (Bug M fix: tube detection prior) ───────────────────
    USE_FRANGI = True            # add Frangi as 8th input channel
    # A3 (HIGH) fix: sigma range broadened for 10 m Sentinel-1.
    # OLD sigmas [0.5, 1.0, 2.0, 3.0] (in pixel units) cover only 5–30 m
    # features. With Sentinel-1 IW GRD at 10 m ground sampling distance
    # (GSD), 1 sigma pixel = 10 m, so this range misses most rivers
    # (10–100 m wide) and small reservoirs (50–500 m). Pune-area
    # rivers (Mula, Mutha, Bhima) are typically 30–80 m wide;
    # canals 5–15 m; lakes 100–1000 m.
    #
    # NEW range [1.0, 2.0, 4.0, 8.0] covers 10–80 m features, which
    # matches the dominant water-body widths in the Pune AOI.
    # Reference: Frangi et al. 1998 recommends log-spaced sigmas
    # covering the expected feature scale range. For 10 m GSD, the
    # canonical sigma range is [GSD, 4·GSD, 8·GSD] = [10, 40, 80] m
    # in physical units, or [1, 4, 8] in pixel units.
    FRANGI_SIGMAS = [1.0, 2.0, 4.0, 8.0]
    FRANGI_ALPHA = 0.5
    FRANGI_BETA = 0.5

    # ─── Multi-Scale TTA (Bug O fix) ───────────────────────────────────────────
    TTA_SCALES = [0.75, 1.0, 1.25]
    TTA_ENABLED = False          # default off, enable via CLI flag

    # ─── Validation & Checkpointing ───────────────────────────────────────────
    VAL_INTERVAL = 1
    CHECKPOINT_EVERY_EPOCH = True
    EARLY_STOPPING_PATIENCE = 25

    # ─── Paths ────────────────────────────────────────────────────────────────
    OUTPUT_DIR = Path.home() / 'sar_water_detection' / f'results_{DEVICE.type}_v4_robust_v2'
    MODEL_CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    BEST_MODEL_PATH = OUTPUT_DIR / 'best_v2.pth'
    LATEST_CHECKPOINT_PATH = OUTPUT_DIR / 'latest_checkpoint_v2.pth'
    TRAINING_STATE_PATH = OUTPUT_DIR / 'training_state_v2.json'

    def log_device_info(self) -> str:
        """Return a one-line summary of the active device for logging."""
        if self.DEVICE.type == 'cuda':
            return (f"Device: CUDA ({torch.cuda.get_device_name(0)}) | "
                    f"AMP: {self.USE_AMP} ({self.AMP_DTYPE}) | "
                    f"Workers: {self.NUM_WORKERS} | "
                    f"Pin memory: {self.PIN_MEMORY} | "
                    f"Batch: {self.BATCH_SIZE}")
        if self.DEVICE.type == 'mps':
            return f"Device: MPS (Apple Silicon) | Workers: {self.NUM_WORKERS} | Batch: {self.BATCH_SIZE}"
        return f"Device: CPU | Workers: {self.NUM_WORKERS} | Batch: {self.BATCH_SIZE}"
