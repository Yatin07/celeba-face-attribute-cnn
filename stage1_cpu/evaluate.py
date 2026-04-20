# ============================================================
# evaluate.py — Test set evaluation using best saved model
#
# Usage:
#   python evaluate.py
#
# Loads:
#   best_model_stage1.pth  (if TRAIN_MODE="subset")
#   best_model_stage2.pth  (if TRAIN_MODE="full")
#
# Output:
#   Overall: loss, accuracy, precision, recall, F1
#   Per-label accuracy table (sorted, with bar chart)
# ============================================================

import os
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import load_dataframes, get_dataloaders
from model import SimpleCNN
from utils import compute_metrics


def evaluate():
    print("\n" + "=" * 55)
    print("  CelebA — Test Set Evaluation")
    print("=" * 55)

    # ── 1. Load data ──────────────────────────────────────────
    # We need all three DataFrames to call get_dataloaders()
    # but only test_loader is used here
    train_df, val_df, test_df = load_dataframes()
    _, _, test_loader, attr_names = get_dataloaders(train_df, val_df, test_df)

    # ── 2. Determine which checkpoint to load ─────────────────
    stage_name = "stage1" if config.TRAIN_MODE == "subset" else "stage2"
    best_path  = os.path.join(config.CHECKPOINT_DIR,
                               f"best_model_{stage_name}.pth")

    if not os.path.exists(best_path):
        print(f"\n[ERROR] No trained model found at: {best_path}")
        print("        Please run  python train.py  first.")
        return

    # ── 3. Load model ─────────────────────────────────────────
    model = SimpleCNN(num_classes=config.NUM_ATTRS,
                      dropout=config.DROPOUT)
    ckpt  = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"\n[INFO] Loaded : {best_path}")
    print(f"[INFO] Epoch  : {ckpt.get('epoch', '?')}")
    print(f"[INFO] ValLoss: {ckpt.get('val_loss', '?'):.4f}")

    # ── 4. Evaluate on test set ───────────────────────────────
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader,
                                   desc="  Testing", unit="batch"):
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds,  dim=0)   # (N, 40)
    all_labels = torch.cat(all_labels, dim=0)   # (N, 40)

    avg_loss = total_loss / len(test_loader)
    metrics  = compute_metrics(all_preds, all_labels)

    # ── 5. Print overall results ──────────────────────────────
    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    print(f"  Test Loss      : {avg_loss:.4f}")
    print(f"  Test Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Test Precision : {metrics['precision']:.2f}%")
    print(f"  Test Recall    : {metrics['recall']:.2f}%")
    print(f"  Test F1-Score  : {metrics['f1']:.2f}%")
    print("=" * 55)

    # ── 6. Per-label breakdown ────────────────────────────────
    # Per-label accuracy (shape: 40,)
    per_label_acc = (all_preds == all_labels).float().mean(dim=0) * 100

    # Sort attributes by accuracy (descending) for easy reading
    attr_acc = sorted(zip(attr_names, per_label_acc.tolist()),
                      key=lambda x: -x[1])

    print("\n  Per-Attribute Accuracy (sorted, high → low):")
    print(f"  {'Attribute':<24} {'Acc':>6}  {'Bar'}")
    print("  " + "-" * 48)
    for name, acc in attr_acc:
        filled = int(acc / 10)            # each block = 10%
        bar    = "█" * filled + "░" * (10 - filled)
        print(f"  {name:<24} {acc:>5.1f}%  {bar}")
    print("=" * 55)


if __name__ == '__main__':
    evaluate()
