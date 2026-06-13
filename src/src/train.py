#!/usr/bin/env python3
"""
Pune SAR Water Detection — Training Script v2 (Merged + Enhanced)
=================================================================
Merged from:
  - train_pune_unet6ch_cpu_v4_robust.py (CPU training, GroupNorm, Attention Gates)
  - train_pune_unet6ch_v4_guerrilla.py  (radiometric jitter, weight decay)

BUG FIXES APPLIED:
  - Bug A: Normalization — z-score only, single source of truth (config.py)
  - Bug B: Architecture — UNet6ChRobust with GroupNorm + Attention Gates
  - Bug C: Band order — explicit [VV, VH, DEM, Slope, HAND, TWI, VV/VH_ratio]
  - Bug D: Weight decay = 1e-4, radiometric jitter ±2 dB
  - Bug E: Gamma-distributed speckle noise augmentation
  - Bug F: Incidence angle normalization for VV/VH
  - Bug G: VV/VH ratio channel added (7 channels total)
  - Bug H: 5-fold stratified cross-validation
  - Bug K: weights_only=True for checkpoint loading
  - Bug B-01: Resume from checkpoint with --fresh flag
  - Bug B-02: Fold-specific LATEST/BEST checkpoint paths
  - Bug B-03: bfloat16 AMP autocast enabled
  - Bug B-05: NaN guard catches -inf in addition to NaN
  - Bug B-06: scaler.update() always called
  - Bug B-07: AMP enable logic consistent between train/validate
  - Bug B-08: no_improve_count persisted in checkpoint
  - Bug B-09: OUTPUT_DIR captured after device override
  - Bug B-10: Loss accumulated as tensor (no GPU-CPU sync per batch)
  - Bug B-11: Fold range validated
  - Bug B-13: Shutdown at epoch 0 saves correctly
  - Bug B-14: loss_dict logged

Usage:
    python train.py [--fresh] [--fold 0] [--n_folds 5] [--device cuda]
"""

import os
import sys
import json
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_device
from model import UNet6ChRobust
from dataset import PuneChipsDataset
from losses import CombinedLoss


# ─── Graceful Shutdown ────────────────────────────────────────────────────────
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    logging.info("Shutdown signal received. Will save checkpoint and exit.")


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(config: Config, fold: int = None):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    fold_suffix = f"_fold{fold}" if fold is not None else ""
    log_file = config.OUTPUT_DIR / f"training_v2{fold_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('pune_sar_train_v2')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── LR Scheduler with Warmup ────────────────────────────────────────────────
class WarmupExponentialDecayLR:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (0.1 ** progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_iou(preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    preds_binary = (preds > threshold).astype(np.float32)
    targets_binary = targets.astype(np.float32)
    tp = np.sum((preds_binary == 1) & (targets_binary == 1))
    fp = np.sum((preds_binary == 1) & (targets_binary == 0))
    fn = np.sum((preds_binary == 0) & (targets_binary == 1))
    iou = tp / (tp + fp + fn + 1e-8)
    return float(np.clip(iou, 0.0, 1.0))


# ─── Cross-Validation Split (Bug H + B-04 fix) ───────────────────────────────
def create_cv_splits(all_indices: list, water_coverage: list, n_folds: int = 5, seed: int = 42):
    """
    Bug H + B-04 fix: Stratified k-fold cross-validation.
    Uses quantile binning to ensure each fold has proportional representation
    of easy (high water) and hard (low water) chips.
    """
    rng = np.random.RandomState(seed)
    coverage_arr = np.array(water_coverage)

    # Create bins by quantile
    n_bins = min(n_folds, 10)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(coverage_arr, quantiles)
    bin_ids = np.digitize(coverage_arr, bin_edges[1:-1])  # 0 to n_bins-1

    # For each bin, split indices into n_folds roughly equal parts
    folds = [[] for _ in range(n_folds)]
    for bin_id in range(n_bins):
        bin_idx = np.where(bin_ids == bin_id)[0]
        rng.shuffle(bin_idx)
        splits = np.array_split(bin_idx, n_folds)
        for i, split in enumerate(splits):
            folds[i].extend(split.tolist())

    cv_splits = []
    for i in range(n_folds):
        val_fold = folds[i]
        train_folds = [folds[j] for j in range(n_folds) if j != i]
        train_fold = []
        for f in train_folds:
            train_fold.extend(f)
        cv_splits.append({
            'train': train_fold,
            'val': val_fold,
        })
    return cv_splits


# ─── Checkpoint Management (Bug B-01, B-02, B-08 fixes) ─────────────────────
def save_checkpoint(model, optimizer, epoch: int, best_iou: float, no_improve_count: int,
                    history: dict, config: Config, is_best: bool = False, fold: int = None,
                    scaler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'no_improve_count': no_improve_count,
        'history': history,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'weight_decay': config.WEIGHT_DECAY,
            'input_channels': config.INPUT_CHANNELS,
        },
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Bug B-02 fix: Fold-specific checkpoint paths
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    fold_latest = config.MODEL_CHECKPOINT_DIR / f"latest{fold_suffix}.pth"
    fold_best = config.MODEL_CHECKPOINT_DIR / f"best{fold_suffix}.pth"

    torch.save(checkpoint, fold_latest)

    if is_best:
        torch.save(checkpoint, fold_best)

    epoch_path = config.MODEL_CHECKPOINT_DIR / f"epoch_{epoch:03d}_iou_{best_iou:.4f}{fold_suffix}.pth"
    torch.save(checkpoint, epoch_path)

    # Bug PERF-2 fix: keep only last N epoch checkpoints to avoid filling disk
    # Keep latest, best, and last 3 epoch checkpoints
    max_epoch_ckpts = 3
    epoch_ckpts = sorted(config.MODEL_CHECKPOINT_DIR.glob(f"epoch_*_iou_*{fold_suffix}.pth"))
    if len(epoch_ckpts) > max_epoch_ckpts:
        for old_ckpt in epoch_ckpts[:-max_epoch_ckpts]:
            try:
                old_ckpt.unlink()
            except OSError:
                pass

    state = {
        'epoch': epoch,
        'best_iou': best_iou,
        'no_improve_count': no_improve_count,
        'last_save': datetime.now().isoformat(),
        'history': history,
        'fold': fold,
    }
    with open(config.TRAINING_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def load_checkpoint(model, optimizer, config: Config, fold: int = None, scaler=None):
    """Bug B-01 fix: Actually call this to resume training."""
    fold_suffix = f"_fold{fold}" if fold is not None else ""
    fold_latest = config.MODEL_CHECKPOINT_DIR / f"latest{fold_suffix}.pth"

    if not fold_latest.exists():
        return 0, 0.0, 0, {'train_loss': [], 'val_iou': [], 'val_loss': [], 'learning_rate': []}

    try:
        # Bug K fix: weights_only=True for security
        checkpoint = torch.load(fold_latest, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        no_improve_count = checkpoint.get('no_improve_count', 0)
        history = checkpoint.get('history', {'train_loss': [], 'val_iou': [], 'val_loss': [], 'learning_rate': []})

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logging.info(f"Resumed from checkpoint: epoch {start_epoch}, best_iou={best_iou:.4f}")
        return start_epoch, best_iou, no_improve_count, history
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return 0, 0.0, 0, {'train_loss': [], 'val_iou': [], 'val_loss': [], 'learning_rate': []}


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, train_loader, criterion, optimizer, config: Config, epoch: int, scaler=None, logger=None):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    nan_batches = 0
    # Bug T-05 fix: Separate autocast from scaler. bfloat16 uses autocast but no scaler.
    use_autocast = config.USE_AMP and config.DEVICE.type == 'cuda'
    use_scaler = use_autocast and scaler is not None
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == 'bfloat16' else torch.float16

    # Bug B-10 fix: accumulate loss as tensor to avoid per-batch GPU-CPU sync
    loss_accumulator = torch.tensor(0.0, device=config.DEVICE)

    for batch_idx, batch_data in enumerate(train_loader):
        if SHUTDOWN_REQUESTED:
            break

        # Bug R4 fix: dataset now returns 3-tuple (data, label, sdt)
        if len(batch_data) == 3:
            images, labels, sdts = batch_data
        else:
            images, labels = batch_data
            sdts = None

        images = images.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        labels = labels.to(config.DEVICE, non_blocking=config.PIN_MEMORY).unsqueeze(1)
        if sdts is not None:
            sdts = sdts.to(config.DEVICE, non_blocking=config.PIN_MEMORY).unsqueeze(1)

        # Bug B-05 fix: catch both NaN and -inf in images/labels
        if torch.isnan(images).any() or torch.isinf(images).any() or torch.isnan(labels).any():
            nan_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        # Bug T-05 fix: autocast runs for both float16 and bfloat16
        if use_autocast:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, labels, epoch=epoch, sdt=sdts)
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, labels, epoch=epoch, sdt=sdts)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            nan_batches += 1
            continue

        # Bug B-05 fix: catch NaN and -inf losses
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > config.NAN_LOSS_THRESHOLD:
            nan_batches += 1
            continue

        # Bug B-06 fix: scaler always gets unscale/update even on NaN
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_MAX_NORM)

        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        if has_nan_grad:
            optimizer.zero_grad(set_to_none=True)
            # Bug BUG-4 fix: call scaler.update() even on NaN-grad path to keep scaler state consistent
            # Per PyTorch docs, unscale_ should be paired with step_/update_
            if use_scaler:
                scaler.update()
            nan_batches += 1
            continue

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Bug B-10 fix: accumulate loss on GPU, sync only once at end
        loss_accumulator += loss.detach()
        valid_batches += 1

        if batch_idx % 10 == 0:
            logging.info(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] loss={loss.item():.4f}")

        # Bug B-14 fix: log loss_dict periodically
        if batch_idx % 50 == 0 and logger is not None:
            logger.info(f"    Loss breakdown: {loss_dict}")

    if nan_batches > config.MAX_NAN_BATCHES_PER_EPOCH:
        logging.error(f"Epoch {epoch}: Too many NaN batches ({nan_batches})")

    # Bug B-10 fix: single GPU-CPU sync
    avg_loss = (loss_accumulator / max(valid_batches, 1)).item()
    return avg_loss, nan_batches


@torch.no_grad()
def validate(model, val_loader, criterion, config: Config, epoch: int = 0, scaler=None):
    model.eval()
    all_preds = []
    all_targets = []
    # Bug T-05 fix: Separate autocast from scaler for bfloat16
    use_autocast = config.USE_AMP and config.DEVICE.type == 'cuda'
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == 'bfloat16' else torch.float16

    # Bug ADD-4 fix: accumulate loss as tensor to avoid per-batch GPU-CPU sync
    loss_accumulator = torch.tensor(0.0, device=config.DEVICE)

    for batch_data in val_loader:
        if len(batch_data) == 3:
            images, labels, sdts = batch_data
        else:
            images, labels = batch_data
            sdts = None
        images = images.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
        labels = labels.to(config.DEVICE, non_blocking=config.PIN_MEMORY).unsqueeze(1)
        if sdts is not None:
            sdts = sdts.to(config.DEVICE, non_blocking=config.PIN_MEMORY).unsqueeze(1)

        # Bug T-05 fix: use autocast for both float16 and bfloat16
        if use_autocast:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, labels, epoch=epoch, sdt=sdts)
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, labels, epoch=epoch, sdt=sdts)

        if not torch.isnan(loss):
            loss_accumulator += loss.detach()
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    iou = compute_iou(all_preds.flatten(), all_targets.flatten())
    # Bug ADD-4 fix: single GPU-CPU sync
    total_loss = loss_accumulator.item()
    return total_loss / max(len(val_loader), 1), iou


# ─── Train Single Fold ───────────────────────────────────────────────────────
def train_fold(fold: int, train_indices: list, val_indices: list, config: Config, logger):
    """Train a single cross-validation fold."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING FOLD {fold + 1}/{config.N_FOLDS}")
    logger.info(f"Train: {len(train_indices)} chips | Val: {len(val_indices)} chips")
    logger.info(f"{'=' * 60}")

    # Create datasets
    train_dataset = PuneChipsDataset(train_indices, config, is_training=True)
    val_dataset = PuneChipsDataset(val_indices, config, is_training=False)

    train_weights = [max(cov, 0.05) for cov in train_dataset.water_coverage]
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
    )

    # Bug T-03 fix: Create model FIRST, then move to device, THEN create optimizer
    model = UNet6ChRobust(
        in_channels=config.INPUT_CHANNELS,
        base_filters=config.BASE_FILTERS,
        num_classes=config.NUM_CLASSES,
        use_dsconv=config.USE_DSCONV,
        dsconv_stages=config.DSCONV_STAGES,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        cldice_lambda=config.CLDICE_LAMBDA,
        boundary_weight=config.BOUNDARY_WEIGHT,
        cldice_num_iter=config.CLDICE_NUM_ITER,
        alpha_min=config.BOUNDARY_ALPHA_MIN,
        alpha_decay_epochs=config.BOUNDARY_DECAY_EPOCHS,
    )

    # Bug T-04 fix: Create optimizer from model.parameters() before state_dict load
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Bug T-03 fix: Move model to device AFTER creating optimizer
    model = model.to(config.DEVICE)
    logger.info(f"Model moved to {config.DEVICE}")

    lr_scheduler = WarmupExponentialDecayLR(optimizer, config.WARMUP_EPOCHS, config.NUM_EPOCHS, config.LEARNING_RATE)

    # GPU AMP gradient scaler (None for CPU)
    scaler = None
    if config.USE_AMP and config.DEVICE.type == 'cuda' and config.AMP_DTYPE == 'float16':
        scaler = torch.amp.GradScaler('cuda')
        logger.info("AMP enabled (float16 GradScaler)")
    elif config.USE_AMP and config.DEVICE.type == 'cuda' and config.AMP_DTYPE == 'bfloat16':
        logger.info("AMP enabled (bfloat16, no GradScaler needed)")

    # Bug B-01 fix: Resume from checkpoint (unless --fresh)
    start_epoch = 0
    best_val_iou = 0.0
    no_improve_count = 0
    history = {'train_loss': [], 'val_iou': [], 'val_loss': [], 'learning_rate': []}

    if not getattr(config, '_fresh', False):
        start_epoch, best_val_iou, no_improve_count, history = load_checkpoint(
            model, optimizer, config, fold=fold, scaler=scaler
        )

    logger.info(f"Starting from epoch {start_epoch}, best_iou={best_val_iou:.4f}, no_improve={no_improve_count}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        if SHUTDOWN_REQUESTED:
            # Bug B-13 fix: save correctly even at epoch 0
            save_checkpoint(model, optimizer, max(epoch - 1, 0), best_val_iou,
                          no_improve_count, history, config, fold=fold, scaler=scaler)
            break

        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"\n{'=' * 60}")
        logger.info(f"FOLD {fold} | EPOCH {epoch + 1}/{config.NUM_EPOCHS} | LR={current_lr:.2e}")
        logger.info(f"{'=' * 60}")

        train_loss, nan_count = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler=scaler, logger=logger)
        logger.info(f"Train loss: {train_loss:.4f} (NaN batches: {nan_count})")

        val_loss, val_iou = validate(model, val_loader, criterion, config, epoch=epoch, scaler=scaler)
        alpha = criterion.get_alpha(epoch)
        logger.info(f"Val loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | α={alpha:.3f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['learning_rate'].append(current_lr)

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            no_improve_count = 0
            logger.info(f"  ⭐ New best model saved! IoU={best_val_iou:.4f}")
        else:
            no_improve_count += 1

        # Bug B-08 fix: persist no_improve_count and scaler state
        save_checkpoint(model, optimizer, epoch, best_val_iou, no_improve_count,
                       history, config, is_best=is_best, fold=fold, scaler=scaler)

        if no_improve_count >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping after {config.EARLY_STOPPING_PATIENCE} epochs without improvement")
            break

        logger.info(f"Best IoU: {best_val_iou:.4f} | No improvement: {no_improve_count}/{config.EARLY_STOPPING_PATIENCE}")

    logger.info(f"FOLD {fold} COMPLETE | Best IoU: {best_val_iou:.4f}")
    return best_val_iou, history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--fold', type=int, default=None, help='Train specific fold (0-indexed), or None for all')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--device', type=str, default=None, help='Force device: cuda|mps|cpu (overrides auto-detect)')
    args = parser.parse_args()

    # Bug B-09 fix: Device override BEFORE creating config
    if args.device:
        os.environ['PUNE_SAR_DEVICE'] = args.device.lower()

    # Now create config (will use overridden device)
    config = Config()

    # Bug B-09 fix: Recompute device-dependent settings AFTER device is finalized
    if args.device:
        config.NUM_WORKERS = 0 if config.DEVICE.type == 'cpu' else min(4, os.cpu_count() or 0)
        config.PIN_MEMORY = config.DEVICE.type == 'cuda'
        config.PERSISTENT_WORKERS = config.NUM_WORKERS > 0
        config.USE_AMP = config.DEVICE.type == 'cuda'
        config.BATCH_SIZE = 8 if config.DEVICE.type == 'cpu' else 16

    # Bug B-09 fix: OUTPUT_DIR now uses correct device
    config.OUTPUT_DIR = Path.home() / 'sar_water_detection' / f'results_{config.DEVICE.type}_v4_robust_v2'
    config.MODEL_CHECKPOINT_DIR = config.OUTPUT_DIR / 'checkpoints'
    # Bug BUG-5 fix: Recompute all path-dependent constants after device override
    config.BEST_MODEL_PATH = config.MODEL_CHECKPOINT_DIR / 'best_v2.pth'
    config.LATEST_CHECKPOINT_PATH = config.MODEL_CHECKPOINT_DIR / 'latest_checkpoint_v2.pth'
    config.TRAINING_STATE_PATH = config.OUTPUT_DIR / 'training_state_v2.json'

    # Bug B-11 fix: Validate fold range
    config.N_FOLDS = args.n_folds
    if args.fold is not None and (args.fold < 0 or args.fold >= config.N_FOLDS):
        logger = setup_logging(config)
        logger.error(f"Invalid fold {args.fold}. Must be in [0, {config.N_FOLDS - 1}]")
        sys.exit(1)

    # Bug B-01 fix: Pass --fresh flag to config
    config._fresh = args.fresh

    logger = setup_logging(config)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    logger.info("=" * 80)
    logger.info("PUNE SAR WATER DETECTION — TRAINING v2 (ENHANCED)")
    logger.info("=" * 80)
    logger.info(config.log_device_info())
    logger.info("Bug fixes applied:")
    logger.info("  ✅ Bug A: Z-score normalization (single source of truth)")
    logger.info("  ✅ Bug B: UNet6ChRobust (GroupNorm + Attention Gates)")
    logger.info("  ✅ Bug C: Band order [VV,VH,DEM,Slope,HAND,TWI,VV/VH_ratio,Frangi]")
    logger.info("  ✅ Bug D: Weight decay=1e-4, radiometric jitter ±2 dB")
    logger.info("  ✅ Bug E: Gamma-distributed speckle noise augmentation")
    logger.info("  ✅ Bug F: Incidence angle normalization for VV/VH")
    logger.info("  ✅ Bug G: VV/VH ratio channel (8 channels total)")
    logger.info("  ✅ Bug H: 5-fold stratified cross-validation")
    logger.info("  ✅ Bug K: weights_only=True for checkpoint loading")
    logger.info("  ✅ Bug M: clDice loss + DSConv + Frangi vesselness (narrow rivers)")
    logger.info("  ✅ Bug N: BoundaryLoss with α-scheduling (sharp edges)")
    logger.info("  ✅ Bug S1: SDT augmented together with label")
    logger.info("  ✅ Bug S2: Loss formula corrected")
    logger.info("  ✅ Bug B-01: Resume from checkpoint (use --fresh to ignore)")
    logger.info("  ✅ Bug B-02: Fold-specific checkpoint paths")
    logger.info("  ✅ Bug B-03: bfloat16 AMP autocast enabled")
    logger.info(f"  📁 Output: {config.OUTPUT_DIR}")
    logger.info("=" * 80)

    # Load all chip files
    import rasterio
    all_files = sorted(list(config.CHIPS_DIR.glob('pune_r*.tif')))
    if len(all_files) == 0:
        logger.error(f"No training chips found in {config.CHIPS_DIR}")
        sys.exit(1)

    # Compute water coverage for stratification
    water_coverage = []
    for f in all_files:
        try:
            with rasterio.open(f) as src:
                label = src.read(7)
                valid = ~np.isnan(label)
                cov = (label[valid] == 1).sum() / valid.sum() if valid.sum() > 0 else 0.0
                water_coverage.append(cov)
        except Exception:
            water_coverage.append(0.0)

    all_indices = list(range(len(all_files)))

    # Bug H + B-04 fix: Create stratified CV splits
    cv_splits = create_cv_splits(all_indices, water_coverage, n_folds=config.N_FOLDS, seed=config.CV_SEED)

    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(config.N_FOLDS))

    fold_results = []

    for fold_idx in folds_to_train:
        split = cv_splits[fold_idx]
        train_indices = split['train']
        val_indices = split['val']

        best_iou, history = train_fold(fold_idx, train_indices, val_indices, config, logger)
        fold_results.append({
            'fold': fold_idx,
            'best_val_iou': best_iou,
            'n_train': len(train_indices),
            'n_val': len(val_indices),
        })

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: IoU={r['best_val_iou']:.4f} (Train={r['n_train']}, Val={r['n_val']})")

    if len(fold_results) > 1:
        mean_iou = np.mean([r['best_val_iou'] for r in fold_results])
        std_iou = np.std([r['best_val_iou'] for r in fold_results])
        logger.info(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")

    logger.info("=" * 80)


if __name__ == '__main__':
    main()
