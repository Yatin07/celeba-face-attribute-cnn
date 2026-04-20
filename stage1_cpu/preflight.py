# -*- coding: utf-8 -*-
# ============================================================
# preflight.py — Pre-flight sanity check before full training
#
# Usage:
#   python preflight.py
#
# Runs 5 checks in sequence:
#   [1] Directory structure
#   [2] Dataset files exist
#   [3] Dataset shape test (load 1 batch checks labels + image size)
#   [4] Model forward pass test
#   [5] One mini training step (loss + backward + optimizer)
#
# If ALL 5 pass → safe to run: python train.py
# If ANY fails  → fix the reported issue first
# ============================================================

import os
import sys
import traceback

# ── Color helpers (Windows CMD + PowerShell compatible) ───────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

ok   = lambda msg: print(f"  {GREEN}[OK]  {msg}{RESET}")
fail = lambda msg: print(f"  {RED}[FAIL] {msg}{RESET}")
info = lambda msg: print(f"  {YELLOW}[INFO] {msg}{RESET}")


# ── Check tracker ─────────────────────────────────────────────
passed = []
failed = []


def run_check(name, fn):
    """Run one check, catch any exception, record pass/fail."""
    print(f"\n{CYAN}[CHECK] {name}{RESET}")
    print("  " + "-" * 45)
    try:
        fn()
        ok(f"PASSED: {name}")
        passed.append(name)
    except Exception as e:
        fail(f"FAILED: {name}")
        print(f"  {RED}  Reason: {e}{RESET}")
        print(f"  {RED}  Detail: {traceback.format_exc().splitlines()[-1]}{RESET}")
        failed.append(name)


# ══════════════════════════════════════════════════════════════
# CHECK 1 — Directory structure
# ══════════════════════════════════════════════════════════════

def check_directories():
    import config

    required_files = [
        'config.py', 'dataset.py', 'model.py',
        'train.py', 'evaluate.py', 'predict.py', 'utils.py',
    ]

    # Python files
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")
        info(f"Found: {f}")

    # Output directories (auto-create if missing)
    for d in [config.CHECKPOINT_DIR, config.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
        info(f"Directory OK: {d}")


# ══════════════════════════════════════════════════════════════
# CHECK 2 — Dataset files exist and are readable
# ══════════════════════════════════════════════════════════════

def check_dataset_files():
    import config

    # Check attribute file
    if not os.path.exists(config.ATTR_PATH):
        raise FileNotFoundError(
            f"Attr file missing: {config.ATTR_PATH}")
    size_mb = os.path.getsize(config.ATTR_PATH) / 1024 / 1024
    info(f"Attr file  : {config.ATTR_PATH}  ({size_mb:.1f} MB)")

    # Check partition file
    if not os.path.exists(config.PARTITION_PATH):
        raise FileNotFoundError(
            f"Partition file missing: {config.PARTITION_PATH}")
    size_mb = os.path.getsize(config.PARTITION_PATH) / 1024 / 1024
    info(f"Split file : {config.PARTITION_PATH}  ({size_mb:.1f} MB)")

    # Check image directory
    if not os.path.isdir(config.IMAGE_DIR):
        raise NotADirectoryError(
            f"Image folder missing: {config.IMAGE_DIR}")

    # Count a few images
    sample_files = [f for f in os.listdir(config.IMAGE_DIR)
                    if f.endswith('.jpg')][:5]
    if not sample_files:
        raise RuntimeError("No .jpg files found in image directory!")

    info(f"Image dir  : {config.IMAGE_DIR}")
    info(f"Sample images: {sample_files}")

    # Verify first image actually opens
    from PIL import Image
    first_img_path = os.path.join(config.IMAGE_DIR, sample_files[0])
    img = Image.open(first_img_path).convert('RGB')
    info(f"First image size (raw): {img.size}  -> will be resized to "
         f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}")


# ══════════════════════════════════════════════════════════════
# CHECK 3 — Dataset + DataLoader shapes
# ══════════════════════════════════════════════════════════════

def check_dataset_shapes():
    import torch
    import config
    from dataset import load_dataframes, get_dataloaders

    # Use only a tiny slice to keep this check fast
    config.TRAIN_MODE = "subset"
    config.SUBSET_SIZE = 200    # tiny — just enough to form 3 batches

    train_df, val_df, test_df = load_dataframes()

    # Temporarily shrink val/test for speed
    val_df_small  = val_df.iloc[:200].reset_index(drop=True)
    test_df_small = test_df.iloc[:200].reset_index(drop=True)

    from dataset import get_dataloaders
    train_loader, val_loader, test_loader, attr_names = get_dataloaders(
        train_df, val_df_small, test_df_small)

    info(f"Attr names (first 5): {attr_names[:5]}")
    info(f"Train batches: {len(train_loader)}")

    images, labels = next(iter(train_loader))

    # Shape checks
    exp_img   = (config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    exp_label = (config.BATCH_SIZE, config.NUM_ATTRS)

    info(f"images.shape = {tuple(images.shape)}  (expected {exp_img})")
    info(f"labels.shape = {tuple(labels.shape)}  (expected {exp_label})")

    assert tuple(images.shape) == exp_img,  \
        f"Image shape mismatch: got {tuple(images.shape)}"
    assert tuple(labels.shape) == exp_label, \
        f"Label shape mismatch: got {tuple(labels.shape)}"

    # Label value check — must be 0 or 1 (never -1)
    unique_vals = labels.unique().tolist()
    info(f"Unique label values: {unique_vals}  (must be [0.0, 1.0])")
    assert set(unique_vals).issubset({0.0, 1.0}), \
        f"Labels contain unexpected values: {unique_vals}"

    # Image range check — after normalisation should be roughly [-1, 1]
    img_min = images.min().item()
    img_max = images.max().item()
    info(f"Image value range: [{img_min:.2f}, {img_max:.2f}]  "
         f"(expect approx [-1, 1] after normalisation)")
    assert img_min >= -2.0 and img_max <= 2.0, \
        f"Image values out of expected range: [{img_min}, {img_max}]"


# ══════════════════════════════════════════════════════════════
# CHECK 4 — Model forward pass
# ══════════════════════════════════════════════════════════════

def check_model_forward():
    import torch
    import config
    from model import SimpleCNN, model_summary

    model = SimpleCNN(num_classes=config.NUM_ATTRS,
                      dropout=config.DROPOUT)
    model_summary(model)

    # Single image forward pass
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = model(x)

    info(f"Input  shape : {tuple(x.shape)}")
    info(f"Output shape : {tuple(y.shape)}  (must be (1, 40))")

    assert tuple(y.shape) == (1, config.NUM_ATTRS), \
        f"Output shape wrong: {tuple(y.shape)}"

    # Verify output is raw logits (NOT clamped to [0,1])
    # If sigmoid was applied inside model, all values would be in [0,1]
    # With raw logits, we should see some values outside that range
    info(f"Output range : [{y.min().item():.3f}, {y.max().item():.3f}]  "
         f"(raw logits - sigmoid applied ONLY in predict.py/loss)")

    # Batch forward pass
    x_batch = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y_batch = model(x_batch)
    assert tuple(y_batch.shape) == (4, config.NUM_ATTRS), \
        f"Batch output shape wrong: {tuple(y_batch.shape)}"
    info(f"Batch test   : (4,3,128,128) -> {tuple(y_batch.shape)}  OK")


# ══════════════════════════════════════════════════════════════
# CHECK 5 — One mini training step (end-to-end)
# ══════════════════════════════════════════════════════════════

def check_one_training_step():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import config
    from model import SimpleCNN
    from dataset import load_dataframes, get_dataloaders

    # Use tiny subset — just 1 batch needed
    config.TRAIN_MODE  = "subset"
    config.SUBSET_SIZE = 100

    train_df, val_df, test_df = load_dataframes()
    val_small  = val_df.iloc[:100].reset_index(drop=True)
    test_small = test_df.iloc[:100].reset_index(drop=True)

    train_loader, _, _, _ = get_dataloaders(train_df, val_small, test_small)

    model     = SimpleCNN(num_classes=config.NUM_ATTRS)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    images, labels = next(iter(train_loader))

    info(f"Mini batch - images: {tuple(images.shape)}, "
         f"labels: {tuple(labels.shape)}")

    # Forward
    optimizer.zero_grad()
    outputs = model(images)
    info(f"Forward pass output shape: {tuple(outputs.shape)}")

    # Loss
    loss = criterion(outputs, labels)
    info(f"Loss value: {loss.item():.4f}  "
         f"(expect ~0.6-0.7 at random initialisation)")

    assert loss.item() > 0, "Loss is zero or negative - something is wrong"
    assert not torch.isnan(loss),  "Loss is NaN - check label values"
    assert not torch.isinf(loss),  "Loss is Inf - check model outputs"

    # Backward
    loss.backward()
    info("Backward pass: OK")

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    info("Gradient clipping: OK")

    # Optimizer step
    optimizer.step()
    info("Optimizer step: OK")

    # Verify predictions
    preds = (torch.sigmoid(outputs) > 0.5).float()
    info(f"Predictions shape: {tuple(preds.shape)}  "
         f"- unique values: {preds.unique().tolist()}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  CelebA Pre-Flight Sanity Check")
    print("=" * 55)

    run_check("1. Directory Structure",         check_directories)
    run_check("2. Dataset Files Exist",         check_dataset_files)
    run_check("3. Dataset Shape (1 batch)",     check_dataset_shapes)
    run_check("4. Model Forward Pass",          check_model_forward)
    run_check("5. One Mini Training Step",      check_one_training_step)

    # ── Final report ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PRE-FLIGHT REPORT")
    print("=" * 55)

    total = len(passed) + len(failed)
    for c in passed:
        print(f"  {GREEN}[PASS] {c}{RESET}")
    for c in failed:
        print(f"  {RED}[FAIL] {c}{RESET}")

    print(f"\n  {total - len(failed)}/{total} checks passed")

    if not failed:
        print(f"\n  {GREEN}ALL CLEAR -- run:  python train.py{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n  {RED}ISSUES FOUND -- fix the failing checks before training.{RESET}\n")
        sys.exit(1)
