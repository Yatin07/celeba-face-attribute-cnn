# ============================================================
# utils.py — Shared helper functions
#
# Contains:
#   1. compute_metrics()  — accuracy, precision, recall, F1
#   2. save_plots()       — loss + accuracy curves
#   3. save_checkpoint()  — save model state dict + metadata
#   4. load_checkpoint()  — restore model (+ optimizer) + epoch
# ============================================================

import os
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display required
import matplotlib.pyplot as plt

import config


# ── 1. Metrics ────────────────────────────────────────────────

def compute_metrics(all_preds, all_labels):
    """
    Compute multi-label classification metrics.

    Args:
        all_preds  : (N, 40) float tensor — binary predictions (0 or 1)
        all_labels : (N, 40) float tensor — ground truth (0 or 1)

    Returns:
        dict with keys: accuracy, precision, recall, f1  (all in %)

    How each metric is computed (per-label then averaged):
        Accuracy  = correct predictions / total predictions
        Precision = TP / (TP + FP)   — how often positive pred is right
        Recall    = TP / (TP + FN)   — how many positives are caught
        F1        = 2 * P * R / (P + R)  — harmonic mean of P and R
    """
    # Element-wise counts per label (shape: 40,)
    tp = (all_preds *       all_labels ).sum(dim=0)   # True  Positive
    fp = (all_preds * (1 - all_labels) ).sum(dim=0)   # False Positive
    fn = ((1 - all_preds) * all_labels ).sum(dim=0)   # False Negative

    # Per-label accuracy then mean over all 40 attributes
    per_label_acc = (all_preds == all_labels).float().sum(dim=0) \
                    / all_labels.shape[0]
    accuracy  = per_label_acc.mean().item()

    # Macro-averaged precision, recall, F1 (1e-8 prevents div-by-zero)
    per_precision = tp / (tp + fp + 1e-8)
    per_recall    = tp / (tp + fn + 1e-8)
    precision = per_precision.mean().item()
    recall    = per_recall.mean().item()
    f1        = (2 * precision * recall / (precision + recall + 1e-8))

    return {
        'accuracy' : accuracy  * 100,
        'precision': precision * 100,
        'recall'   : recall    * 100,
        'f1'       : f1        * 100,
    }


# ── 2. Plots ──────────────────────────────────────────────────

def save_plots(train_losses, val_losses, val_accuracies):
    """
    Save two plots to OUTPUT_DIR:
        loss_curve.png      — train vs val loss per epoch
        accuracy_curve.png  — val accuracy per epoch

    Args:
        train_losses    : list of float (one per epoch)
        val_losses      : list of float (one per epoch)
        val_accuracies  : list of float in % (one per epoch)
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # ── Loss curve ────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'o-', label='Train Loss',
             color='#E63946', linewidth=2, markersize=5)
    plt.plot(epochs, val_losses,   's-', label='Validation Loss',
             color='#457B9D', linewidth=2, markersize=5)
    plt.xlabel('Epoch',     fontsize=12)
    plt.ylabel('Loss',      fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(config.OUTPUT_DIR, 'loss_curve.png')
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {loss_path}")

    # ── Accuracy curve ────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracies, 'D-', label='Val Accuracy (%)',
             color='#2A9D8F', linewidth=2, markersize=5)
    plt.xlabel('Epoch',        fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(config.OUTPUT_DIR, 'accuracy_curve.png')
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {acc_path}")


# ── 3. Checkpoint ─────────────────────────────────────────────

def save_checkpoint(state: dict, filepath: str):
    """
    Save a checkpoint dictionary to disk.

    Args:
        state    : dict with model_state_dict, optimizer_state_dict, epoch, etc.
        filepath : target .pth file path
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        filepath  : path to .pth file
        model     : SimpleCNN instance
        optimizer : torch.optim.Optimizer (optional)

    Returns:
        start_epoch (int) — epoch to resume FROM (last_saved + 1)
    """
    ckpt = torch.load(filepath, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("[RESUME] Optimizer state restored.")
    else:
        print("[RESUME] Optimizer state NOT restored (fresh optimizer).")

    last_epoch  = ckpt.get('epoch', 0)
    start_epoch = last_epoch + 1
    print(f"[RESUME] Loaded checkpoint: '{filepath}'")
    print(f"[RESUME] Last completed epoch : {last_epoch}")
    print(f"[RESUME] Resuming from epoch  : {start_epoch}")
    return start_epoch
