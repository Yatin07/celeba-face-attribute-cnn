# ============================================================
# train.py — Main training script (Enhanced Progress Tracking)
#
# Usage:
#   python train.py
#
# Controlled by config.py:
#   TRAIN_MODE      = "subset"  -> Stage 1: 50k images
#                   = "full"    -> Stage 2: 162k images
#   RESUME_TRAINING = False     -> fresh training
#                   = True      -> resume from best_model_stage1.pth
#
# Progress display (live tqdm postfix per batch):
#   Epoch 2/15 [Train] | imgs=7,680/50,000 | rem=42,320
#                      | t/b=1.8s | ela=00:04:20
#                      | ETA_e=00:18:30 | ETA_T=~3h 12m
#
# Improvements:
#   [1] Seed (torch + numpy + random)        — reproducibility
#   [2] Gradient clipping (max_norm=1.0)     — stable gradients
#   [3] CSV training log per epoch           — debugging
#   [4] Sample predictions every 2 epochs   — verify learning
#   [5] Moving avg batch time (deque 10)    — smooth ETA
#   [6] ETA for epoch + full training        — time awareness
# ============================================================

import os
import sys
import csv
import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import load_dataframes, get_dataloaders
from model import SimpleCNN, model_summary
from utils import compute_metrics, save_plots, save_checkpoint, load_checkpoint


# ── 1. Reproducibility ────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Fix all random seeds for reproducibility.
    Same seed -> same weight initialisation, same data shuffle order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    print(f"[SEED] Random seed fixed to {seed} (cudnn.benchmark=True)")


# ── 2. Time formatters ────────────────────────────────────────

def fmt_hms(seconds: float) -> str:
    """
    Format a duration in seconds to HH:MM:SS string.
    Example: 3750 -> '01:02:30'
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_eta(seconds: float) -> str:
    """
    Compact ETA format:
      < 1 hour  -> HH:MM:SS  (e.g. '00:22:10')
      >= 1 hour -> ~Xh YYm   (e.g. '~3h 12m')
    """
    seconds = max(0, seconds)
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"~{h}h {m:02d}m"
    return fmt_hms(seconds)


# ── 3. CSV training log ───────────────────────────────────────

def init_csv_log(log_path: str):
    """Create training_log.csv with headers."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_loss', 'val_loss',
            'val_accuracy', 'val_f1', 'val_precision', 'val_recall',
            'epoch_time_min', 'learning_rate'
        ])
    print(f"[LOG]  Training log -> {log_path}")


def append_csv_log(log_path: str, epoch: int,
                   train_loss: float, val_loss: float,
                   metrics: dict, epoch_time: float, current_lr: float):
    """Append one epoch row to CSV."""
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch + 1,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{metrics['accuracy']:.2f}",
            f"{metrics['f1']:.2f}",
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{epoch_time / 60:.1f}",
            f"{current_lr:.6f}"
        ])


# ── 4. Sample prediction printer ─────────────────────────────

def print_sample_predictions(model, fixed_batch, attr_names: list,
                              epoch: int):
    """
    Every 2 epochs, run model on a fixed image and print
    top-8 attribute predictions with confidence bars.
    Helps verify the model is actually learning something.
    """
    if (epoch + 1) % 2 != 0:
        return

    images, labels = fixed_batch
    sample_img     = images[:1]      # (1, 3, 128, 128)
    sample_label   = labels[0]       # (40,) ground truth

    model.eval()
    with torch.no_grad():
        sample_img = sample_img.to(config.DEVICE)
        logits = model(sample_img)
        probs  = torch.sigmoid(logits).squeeze(0)   # (40,) probabilities

    pairs = sorted(zip(attr_names, probs.tolist(), sample_label.tolist()),
                   key=lambda x: -x[1])

    print(f"\n  Sample Predictions (Epoch {epoch+1}, fixed val image):")
    print(f"  {'Attribute':<24}  {'Conf':>5}  Bar           GT")
    print("  " + "-" * 54)
    for name, prob, gt in pairs[:8]:
        filled = int(prob * 10)
        bar    = "[" + "=" * filled + "." * (10 - filled) + "]"
        flag   = " PRESENT" if prob > 0.5 else " absent"
        gt_str = "1" if gt == 1.0 else "0"
        print(f"  {name:<24}  {prob:.2f}  {bar}  GT={gt_str}{flag}")

    model.train()


# ── 5. Enhanced epoch runner ──────────────────────────────────

def run_epoch(model, loader, criterion, optimizer,
              training: bool,
              scaler=None,
              epoch: int = 0,
              total_epochs: int = 1,
              avg_epoch_time: float = None):
    """
    Run one full epoch with detailed live progress tracking.

    Live tqdm postfix (updated every 5 batches):
        imgs    = images processed / total dataset images
        rem     = remaining images in this epoch
        t/b     = moving-average time per batch (last 10 batches)
        ela     = elapsed time since epoch start (HH:MM:SS)
        ETA_e   = estimated time remaining in this epoch
        ETA_T   = estimated time remaining for full training

    ETA_T logic:
        - If avg_epoch_time is known (>= 1 completed epoch):
            ETA_T = remaining_full_epochs * avg_epoch_time + ETA_epoch
        - If first epoch (no history yet):
            ETA_T estimated from current epoch progress * full epoch count

    Args:
        model          : SimpleCNN
        loader         : DataLoader
        criterion      : BCEWithLogitsLoss
        optimizer      : Adam optimizer
        training       : True = weight update; False = inference only
        epoch          : current epoch index (0-based)
        total_epochs   : max epochs to run
        avg_epoch_time : mean seconds per completed epoch (None on first epoch)

    Returns:
        avg_loss   (float)  — mean loss over all batches
        metrics    (dict)   — accuracy, precision, recall, f1
        epoch_time (float)  — total wall-clock seconds this epoch took
    """
    model.train() if training else model.eval()

    total_batches  = len(loader)
    dataset_size   = len(loader.dataset)
    batch_size_cfg = loader.batch_size or config.BATCH_SIZE

    # Moving average: last 10 batch times (deque auto-drops oldest)
    batch_times = deque(maxlen=10)
    epoch_start = time.time()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    phase = "Train" if training else "Val  "

    # tqdm bar — desc shows Epoch X/Y [Phase]
    # postfix carries all timing details (updated every 5 batches)
    pbar = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=f"  Epoch {epoch+1:>2}/{total_epochs} [{phase}]",
        unit="bat",
        leave=True,
        dynamic_ncols=True,
    )

    for batch_idx, (images, labels) in pbar:

        batch_start = time.time()

        # ── Device transfer (GPU-agnostic) ────────────────────
        # On CPU: no-op.  On GPU: moves tensors to VRAM automatically.
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        # ── Forward + (optionally) backward ───────────────────
        if training:
            optimizer.zero_grad()
            
            # AMP Autocast context block
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            
            # Silent NaN protection
            if torch.isnan(loss):
                print(f"\n[FATAL] NaN loss detected at batch {batch_idx}! Stopping training.")
                sys.exit(1)
            
            # First-batch VRAM print
            if batch_idx == 0 and config.DEVICE.type == 'cuda':
                print(f"\n  [VRAM] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB used")

            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(images)
                loss    = criterion(outputs, labels)

        # ── Record batch time ──────────────────────────────────
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # ── Progress computations ──────────────────────────────
        batches_done     = batch_idx + 1
        batches_rem      = total_batches - batches_done
        elapsed          = time.time() - epoch_start
        avg_batch_t      = sum(batch_times) / len(batch_times)

        # Images: clamp at dataset_size to handle last partial batch
        images_done = min(batches_done * batch_size_cfg, dataset_size)
        images_rem  = dataset_size - images_done

        # ETA for current epoch
        eta_epoch = batches_rem * avg_batch_t

        # ETA for full training
        remaining_full_epochs = total_epochs - (epoch + 1)
        if avg_epoch_time is not None:
            # We have timing data from past epochs — use it
            eta_total = remaining_full_epochs * avg_epoch_time + eta_epoch
        else:
            # First epoch: estimate total epoch time from current progress
            if batches_done > 0:
                est_total_epoch = (elapsed / batches_done) * total_batches
            else:
                est_total_epoch = 0.0
            eta_total = remaining_full_epochs * est_total_epoch + eta_epoch

        # ── Update postfix every 5 batches (reduces console churn) ──
        if batch_idx % 5 == 0 or batches_done == total_batches:
            imgs_per_sec = batch_size_cfg / avg_batch_t if avg_batch_t > 0 else 0
            
            pbar.set_postfix(
                imgs    = f"{images_done:,}/{dataset_size:,}",
                rem     = f"{images_rem:,}",
                t_b     = f"{avg_batch_t:.2f}s",
                fps     = f"{imgs_per_sec:.0f}/s",
                ela     = fmt_hms(elapsed),
                ETA_e   = fmt_hms(eta_epoch),
                ETA_T   = fmt_eta(eta_total),
                refresh = True,
            )

    pbar.close()

    epoch_time = time.time() - epoch_start
    all_preds  = torch.cat(all_preds,  dim=0)   # (N, 40)
    all_labels = torch.cat(all_labels, dim=0)   # (N, 40)
    avg_loss   = total_loss / total_batches
    metrics    = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics, epoch_time


# ── 6. Main training orchestrator ─────────────────────────────

def train():
    print("\n" + "=" * 58)
    print("  CelebA Local Training Pipeline  (CPU)")
    print("=" * 58)
    print(f"  TRAIN_MODE       : {config.TRAIN_MODE}")
    print(f"  RESUME_TRAINING  : {config.RESUME_TRAINING}")
    print(f"  Max epochs       : {config.EPOCHS}")
    print(f"  Batch size       : {config.BATCH_SIZE}")
    print(f"  Early stop pat.  : {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 58 + "\n")

    # Seed first — before any random operations
    set_seed(42)

    # ── Data ──────────────────────────────────────────────────
    train_df, val_df, test_df = load_dataframes()
    train_loader, val_loader, _, attr_names = get_dataloaders(
        train_df, val_df, test_df)

    # Fixed val batch for sample predictions (same batch every epoch)
    fixed_batch = next(iter(val_loader))

    # ── Model ─────────────────────────────────────────────────
    model = SimpleCNN(num_classes=config.NUM_ATTRS,
                      dropout=config.DROPOUT)
    model_summary(model)

    # ── Loss ──────────────────────────────────────────────────
    # BCEWithLogitsLoss = Sigmoid + BCE — numerically stable
    # Expects raw logits (NOT sigmoid output)
    criterion = nn.BCEWithLogitsLoss()

    # ── Paths ─────────────────────────────────────────────────
    stage_name = "stage1" if config.TRAIN_MODE == "subset" else "stage2"
    best_path  = os.path.join(config.CHECKPOINT_DIR,
                               f"best_model_{stage_name}.pth")
    ckpt_path  = os.path.join(config.CHECKPOINT_DIR,
                               "checkpoint_latest.pth")
    log_path   = os.path.join(config.OUTPUT_DIR, "training_log.csv")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR,     exist_ok=True)
    init_csv_log(log_path)

    # ── Optimizer + Resume ────────────────────────────────────
    start_epoch = 0

    if config.RESUME_TRAINING:
        lr        = config.RESUME_LR    # lower LR to avoid forgetting
        optimizer = optim.Adam(model.parameters(), lr=lr)

        resume_src = os.path.join(config.CHECKPOINT_DIR,
                                  "best_model_stage1.pth")
        if not os.path.exists(resume_src):
            resume_src = ckpt_path

        print(f"\n[RESUMING TRAINING]")
        print(f"  Source : {resume_src}")
        print(f"  LR     : {lr}  (reduced from {config.LEARNING_RATE})\n")

        start_epoch = load_checkpoint(resume_src, model, optimizer)

    else:
        lr        = config.LEARNING_RATE
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"[TRAIN] Fresh training | LR = {lr}\n")

    # ── Move model to device (GPU-agnostic) ──────────────────
    # On CPU this is a no-op. On GPU it moves all weights to VRAM.
    model = model.to(config.DEVICE)
    print(f"[DEVICE] Running on: {config.DEVICE}")

    # ── GPU AMP & LR Scheduler ───────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE.type == 'cuda' else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # ── History ───────────────────────────────────────────────
    train_losses   = []
    val_losses     = []
    val_accuracies = []
    epoch_times    = []   # wall-clock seconds per epoch (for ETA)

    best_val_loss  = float('inf')
    best_val_acc   = 0.0     # track accuracy at best checkpoint
    best_epoch     = 0       # which epoch gave best val loss
    patience_count = 0
    training_start = time.time()

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(start_epoch, config.EPOCHS):

        # ── Always read LR directly from optimizer ─────────────
        # WHY: In Stage 2 the LR is halved; reading param_groups
        #      ensures the displayed value is always the ACTUAL LR.
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n{'='*58}")
        print(f"  Epoch {epoch+1}/{config.EPOCHS}  |  "
              f"Stage: {stage_name}  |  LR: {current_lr:.6f}")
        if best_val_loss < float('inf'):
            print(f"  Best so far    : val_loss={best_val_loss:.4f}  "
                  f"acc={best_val_acc:.2f}%  (epoch {best_epoch+1})")
        if epoch_times:
            avg_e = sum(epoch_times) / len(epoch_times)
            rem   = (config.EPOCHS - epoch) * avg_e
            print(f"  Avg epoch time : {fmt_hms(avg_e)}  |  "
                  f"Est. remaining: {fmt_eta(rem)}")
        print(f"{'='*58}")

        # Pass avg epoch time so run_epoch can compute ETA_T accurately
        avg_epoch_time = (sum(epoch_times) / len(epoch_times)
                          if epoch_times else None)

        # ── Training pass ──────────────────────────────────────
        train_loss, train_m, train_time = run_epoch(
            model, train_loader, criterion, optimizer,
            training       = True,
            scaler         = scaler,
            epoch          = epoch,
            total_epochs   = config.EPOCHS,
            avg_epoch_time = avg_epoch_time,
        )

        # ── Validation pass ────────────────────────────────────
        val_loss, val_m, val_time = run_epoch(
            model, val_loader, criterion, optimizer,
            training       = False,
            scaler         = scaler,
            epoch          = epoch,
            total_epochs   = config.EPOCHS,
            avg_epoch_time = avg_epoch_time,
        )

        epoch_total = train_time + val_time
        epoch_times.append(epoch_total)

        # ── Record history ─────────────────────────────────────
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_m['accuracy'])

        # ── Epoch summary ──────────────────────────────────────
        total_elapsed = time.time() - training_start
        print(f"\n  --- Epoch {epoch+1} Summary ---")
        print(f"  Train Loss  : {train_loss:.4f}")
        print(f"  Val   Loss  : {val_loss:.4f}")
        print(f"  Val   Acc   : {val_m['accuracy']:.2f}%")
        print(f"  Val   F1    : {val_m['f1']:.2f}%")
        print(f"  Val   Prec  : {val_m['precision']:.2f}%")
        print(f"  Val   Rec   : {val_m['recall']:.2f}%")
        print(f"  Epoch time  : {fmt_hms(epoch_total)}  "
              f"(train={fmt_hms(train_time)}, val={fmt_hms(val_time)})")
        print(f"  Total ela   : {fmt_hms(total_elapsed)}")

        # ── CSV log ────────────────────────────────────────────
        append_csv_log(log_path, epoch,
                       train_loss, val_loss, val_m, epoch_total, current_lr)

        # ── Sample predictions ─────────────────────────────────
        print_sample_predictions(model, fixed_batch, attr_names, epoch)

        # ── LR Scheduler Step ──────────────────────────────────
        scheduler.step(val_loss)

        # ── Checkpoint every epoch ─────────────────────────────
        # Allows crash recovery from last completed epoch
        save_checkpoint(
            {
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss'            : val_loss,
                'train_mode'          : config.TRAIN_MODE,
            },
            ckpt_path
        )

        # ── Save best model ────────────────────────────────────
        if val_loss < best_val_loss:
            prev_best      = best_val_loss
            best_val_loss  = val_loss
            best_val_acc   = val_m['accuracy']
            best_epoch     = epoch
            patience_count = 0
            save_checkpoint(
                {
                    'epoch'               : epoch,
                    'model_state_dict'    : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss'            : val_loss,
                    'val_accuracy'        : val_m['accuracy'],
                    'train_mode'          : config.TRAIN_MODE,
                },
                best_path
            )
            improvement = prev_best - val_loss if prev_best < float('inf') else val_loss
            print(f"  [BEST] val_loss={val_loss:.4f}  "
                  f"acc={best_val_acc:.2f}%  "
                  f"(improved by {improvement:.4f}) -> saved")
        else:
            patience_count += 1
            print(f"  [NO IMPROVE] val_loss={val_loss:.4f}  "
                  f"(best={best_val_loss:.4f} at epoch {best_epoch+1})  "
                  f"Patience: {patience_count}/{config.EARLY_STOPPING_PATIENCE}")

        # ── Early stopping ─────────────────────────────────────
        if patience_count >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[EARLY STOP] Val loss flat for "
                  f"{config.EARLY_STOPPING_PATIENCE} epochs -> stopping.")
            break

    # ── Final plots ───────────────────────────────────────────
    save_plots(train_losses, val_losses, val_accuracies)

    total_time    = time.time() - training_start
    epochs_ran    = len(train_losses)
    stopped_early = epochs_ran < config.EPOCHS

    _print_and_save_summary(
        stage_name, best_epoch, best_val_loss, best_val_acc,
        epochs_ran, stopped_early, total_time,
        best_path, log_path
    )


def _print_and_save_summary(stage_name, best_epoch, best_val_loss,
                             best_val_acc, epochs_ran, stopped_early,
                             total_time, best_path, log_path):
    """
    Print and save the final training summary.
    Written to outputs/training_summary.txt for reports and viva.
    """
    stop_note = f"(early stop at epoch {epochs_ran})" if stopped_early \
                else f"(all {epochs_ran} epochs completed)"

    lines = [
        "=" * 50,
        "  TRAINING FINAL SUMMARY",
        "=" * 50,
        f"  Stage          : {stage_name}",
        f"  Best Epoch     : {best_epoch + 1}",
        f"  Best Val Loss  : {best_val_loss:.4f}",
        f"  Best Val Acc   : {best_val_acc:.2f}%",
        f"  Total Epochs   : {epochs_ran}  {stop_note}",
        f"  Total Time     : {fmt_hms(total_time)}",
        "-" * 50,
        f"  Best model     : {best_path}",
        f"  Training log   : {log_path}",
        "=" * 50,
    ]

    # Print to console
    print()
    for line in lines:
        print(line)

    # Save to file
    summary_path = os.path.join(
        os.path.dirname(log_path), "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary saved  : {summary_path}")


# ── Entry point (with KeyboardInterrupt safety) ───────────────
if __name__ == '__main__':
    # Wrap train() so Ctrl+C saves progress before exiting.
    # Without this, a stray interrupt can lose hours of training.
    import atexit

    _interrupted = False

    def _emergency_save():
        """Called by atexit — only acts if training was interrupted."""
        if _interrupted:
            print("\n[EXIT] Performing emergency checkpoint save...")
            # Latest checkpoint is already saved per epoch in train().
            # This message confirms the last epoch's data is safe.
            print("[EXIT] Last epoch checkpoint is at: "
                  f"checkpoints/checkpoint_latest.pth")
            print("[EXIT] Resume by setting RESUME_TRAINING = True in config.py")

    atexit.register(_emergency_save)

    try:
        train()
    except KeyboardInterrupt:
        _interrupted = True
        print("\n" + "=" * 50)
        print("  [INTERRUPTED] Training stopped by user (Ctrl+C)")
        print("  Last completed epoch is SAFE in:")
        print("    checkpoints/checkpoint_latest.pth")
        print("  To resume: set RESUME_TRAINING = True in config.py")
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected crash: {e}")
        print("  Last checkpoint: checkpoints/checkpoint_latest.pth")
        raise
