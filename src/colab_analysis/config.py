"""
Shared configuration for the Pune SAR water-detection analysis scripts.

The normalisation statistics below are the per-channel means and standard
deviations of the Pune training patches; the released model was trained on
inputs z-scored with these values. The scripts in this directory (Integrated
Gradients attribution and the pixel-level terrain ablation overlay) apply the
same statistics so that inference matches training.
"""

import os
import torch
from pathlib import Path


# Per-channel z-score statistics of the Pune training patches.
NORM_STATS = {
    'VV':    {'mean': -9.08,  'std': 4.14},
    'VH':    {'mean': -16.33, 'std': 3.97},
    'DEM':   {'mean': 666.74, 'std': 145.62},
    'Slope': {'mean': 8.05,   'std': 8.88},
    'HAND':  {'mean': 46.61,  'std': 76.19},
    'TWI':   {'mean': 10.55,  'std': 2.39},
}

# Channel order of the network input tensors.
BAND_ORDER = ['VV', 'VH', 'DEM', 'Slope', 'HAND', 'TWI']


def get_device() -> torch.device:
    """
    Select the best available compute device.

    Order of preference: CUDA (NVIDIA GPUs), MPS (Apple Silicon), then CPU.
    Override with the environment variable PUNE_SAR_DEVICE=cuda|mps|cpu.
    """
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
    """Settings of the released model run and shared defaults for the scripts."""

    # Device (auto-detected: CUDA > MPS > CPU)
    DEVICE = get_device()
    SEED = 42

    # Model architecture
    INPUT_CHANNELS = 6
    BASE_FILTERS = 64
    NUM_CLASSES = 1

    # Optimisation (as used for the released checkpoint)
    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    LEARNING_RATE = 5e-5
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 0.0
    GRADIENT_CLIP_MAX_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 25

    # Loss: Focal BCE (alpha, gamma) weighted with Dice
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    BCE_WEIGHT = 0.6
    DICE_WEIGHT = 0.4

    # Data
    NORM_STATS = NORM_STATS
    BAND_ORDER = BAND_ORDER

    def log_device_info(self) -> str:
        """Return a one-line summary of the active device for logging."""
        if self.DEVICE.type == 'cuda':
            return f"Device: CUDA ({torch.cuda.get_device_name(0)}) | Batch: {self.BATCH_SIZE}"
        if self.DEVICE.type == 'mps':
            return f"Device: MPS (Apple Silicon) | Batch: {self.BATCH_SIZE}"
        return f"Device: CPU | Batch: {self.BATCH_SIZE}"
