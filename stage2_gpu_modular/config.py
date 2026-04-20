# ============================================================
# config.py — All settings in ONE place
# Change values here only — rest of pipeline adapts
#
# PIPELINE: LOCAL CPU TRAINING
#   Train  → first 50,000 images  (TRAIN_MODE="subset")
#           → all 162,770 images  (TRAIN_MODE="full")
#   Val    → official CelebA val  (~19,867 images)
#   Test   → official CelebA test (~19,962 images)
# ============================================================

import os
from pathlib import Path
import torch

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ONLY CHANGE THIS PATH IF YOU MOVE YOUR IMAGES FOLDER
IMAGE_DIR      = r"C:\MLA\img_align_celeba"

# These auto-resolve relative to the project root
ATTR_PATH      = str(BASE_DIR / "data" / "annotations" / "list_attr_celeba.txt")
PARTITION_PATH = str(BASE_DIR / "data" / "annotations" / "list_eval_partition.txt")

CHECKPOINT_DIR = "checkpoints"   # relative to project root
OUTPUT_DIR     = "outputs"       # plots saved here

# ── Data ──────────────────────────────────────────────────────
IMAGE_SIZE   = 160        # resize all images to 160x160
BATCH_SIZE   = 160    # Optimized for 160x160 on 12GB VRAM
NUM_WORKERS  = 4          # Set to a stable value for Windows
PIN_MEMORY   = True       # False on CPU-only systems
SUBSET_SIZE  = 50_000     # number of train images used in Stage 1

# ── Model ─────────────────────────────────────────────────────
NUM_ATTRS    = 40         # CelebA has 40 binary attributes
DROPOUT      = 0.4        # dropout before final FC layer

# ── Training ─────────────────────────────────────────────────
EPOCHS                   = 20    # maximum training epochs
LEARNING_RATE            = 1e-3  # Adam lr for fresh training
EARLY_STOPPING_PATIENCE  = 3     # stop if val_loss doesn't improve

# ── Mode flags ────────────────────────────────────────────────
# TRAIN_MODE:
#   "subset" → Stage 1: train on first 50k images
#   "full"   → Stage 2: train on all 162k images
TRAIN_MODE       = "full"

# RESUME_TRAINING:
#   False → start fresh (Stage 1)
#   True  → load best_model_stage1.pth and continue (Stage 2)
RESUME_TRAINING  = False

# When resuming (Stage 2), use a lower LR to avoid forgetting
RESUME_LR        = 1e-4
