# ============================================================
# dataset.py — Data loading, preprocessing, and DataLoaders
#
# DEPENDENCY NOTE:
#   torchvision is NOT used here.
#   All image transforms are implemented using PIL + numpy + torch
#   (PIL and numpy are already installed with torch).
#
# Responsibilities:
#   1. Read list_attr_celeba.txt + list_eval_partition.txt
#   2. Merge and split into train / val / test DataFrames
#   3. Apply TRAIN_MODE (subset=50k, full=162k)
#   4. Define CelebADataset (PyTorch Dataset class)
#   5. Return DataLoaders via get_dataloaders()
#
# Memory efficiency:
#   Images are loaded ONE AT A TIME in __getitem__
#   The entire dataset is NEVER held in RAM
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps

import config


# ── Image transforms (no torchvision needed) ─────────────────

def to_tensor_normalised(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL RGB image to a normalised float tensor.

    Steps:
      1. np.array(img)       → (H, W, 3)  uint8  [0, 255]
      2. / 255.0             → (H, W, 3)  float  [0,   1]
      3. .permute(2, 0, 1)  → (3, H, W)  float  [0,   1]
      4. Normalise:  (x - 0.5) / 0.5  →  [-1, +1]
         (same as torchvision Normalize(mean=0.5, std=0.5))
    """
    arr    = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)           # (3, H, W)
    tensor = (tensor - 0.5) / 0.5                             # [-1, +1]
    return tensor


def train_transform(img: Image.Image) -> torch.Tensor:
    """
    Training transform: resize + random horizontal flip + normalise.
    RandomHorizontalFlip(p=0.5): face images can safely be mirrored.
    """
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                     Image.BILINEAR)
    if random.random() < 0.5:
        img = ImageOps.mirror(img)     # horizontal flip
    return to_tensor_normalised(img)


def eval_transform(img: Image.Image) -> torch.Tensor:
    """
    Eval/test transform: resize + normalise only (no augmentation).
    Deterministic — same result every call for same image.
    """
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                     Image.BILINEAR)
    return to_tensor_normalised(img)


# ── Step 1: Load and split DataFrames ────────────────────────

def load_dataframes():
    """
    Load CelebA attribute labels and official train/val/test split.

    Returns:
        train_df : DataFrame — image_id + 40 attribute columns
        val_df   : DataFrame — same structure
        test_df  : DataFrame — same structure

    Notes:
        - list_attr_celeba.txt format: first line = count, second = headers
        - Labels converted from {-1, +1} to {0, 1} for BCEWithLogitsLoss
        - If TRAIN_MODE="subset", train_df is trimmed to first SUBSET_SIZE rows
    """
    print("[DATA] Loading attribute file...")
    # header=1  -> skip the count row, use second row as column names
    # index_col=0 -> first column (filenames) becomes the index
    attr = pd.read_csv(config.ATTR_PATH,
                       sep=r'\s+', header=1, index_col=0)
    attr.index.name = 'image_id'

    # Convert labels: -1 -> 0,  +1 -> 1
    # Formula: (x + 1) // 2  maps -1->0 and +1->1
    attr = ((attr + 1) // 2).reset_index()   # image_id becomes a column
    print(f"[DATA] Attributes loaded: {attr.shape}  "
          f"(rows=images, cols=id+40attrs)")

    print("[DATA] Loading partition file...")
    splits = pd.read_csv(config.PARTITION_PATH, sep=' ', header=None,
                         names=['image_id', 'split'])

    # Merge attributes with partition codes on image filename
    df = splits.merge(attr, on='image_id')
    assert len(df) == 202_599, \
        f"[ERROR] Expected 202,599 rows after merge, got {len(df)}"

    # Official CelebA split codes: 0=train  1=val  2=test
    train_df = df[df['split'] == 0].drop('split', axis=1)\
                                   .reset_index(drop=True)
    val_df   = df[df['split'] == 1].drop('split', axis=1)\
                                   .reset_index(drop=True)
    test_df  = df[df['split'] == 2].drop('split', axis=1)\
                                   .reset_index(drop=True)

    # Apply training mode
    if config.TRAIN_MODE == "subset":
        train_df = train_df.iloc[:config.SUBSET_SIZE]\
                           .reset_index(drop=True)
        print(f"[DATA] TRAIN_MODE = subset  -> "
              f"{len(train_df):,} training images")
    else:
        print(f"[DATA] TRAIN_MODE = full    -> "
              f"{len(train_df):,} training images")

    print(f"[DATA] Val  : {len(val_df):,} images")
    print(f"[DATA] Test : {len(test_df):,} images")
    return train_df, val_df, test_df


# ── Step 2: PyTorch Dataset class ─────────────────────────────

class CelebADataset(Dataset):
    """
    PyTorch Dataset for CelebA multi-label classification.

    Memory efficient: images are opened from disk one at a time
    in __getitem__ — nothing is pre-loaded into RAM.

    Args:
        df        : DataFrame with columns [image_id, attr1, ..., attr40]
        img_dir   : path to img_align_celeba/ folder
        is_train  : if True, apply training augmentations (random flip)
                    if False, apply deterministic eval transform
    """

    def __init__(self, df, img_dir, is_train: bool = False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.is_train  = is_train
        # All columns except image_id are attribute labels
        self.attr_cols = [c for c in df.columns if c != 'image_id']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from disk (not pre-loaded into RAM)
        img_path = os.path.join(self.img_dir, row['image_id'])
        img      = Image.open(img_path).convert('RGB')

        # Apply appropriate transform
        img_tensor = train_transform(img) if self.is_train \
                     else eval_transform(img)

        # Build label tensor: shape (40,), float32 for BCEWithLogitsLoss
        label = torch.tensor(
            row[self.attr_cols].values.astype(float),
            dtype=torch.float32
        )
        return img_tensor, label

    def get_attr_names(self):
        """Return list of 40 attribute names in order."""
        return self.attr_cols


# ── Step 3: Build DataLoaders ─────────────────────────────────

def get_dataloaders(train_df, val_df, test_df):
    """
    Create and return DataLoaders for all three splits.

    Returns:
        train_loader, val_loader, test_loader, attr_names
    """
    train_ds = CelebADataset(train_df, config.IMAGE_DIR, is_train=True)
    val_ds   = CelebADataset(val_df,   config.IMAGE_DIR, is_train=False)
    test_ds  = CelebADataset(test_df,  config.IMAGE_DIR, is_train=False)

    loader_kwargs = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory =config.PIN_MEMORY,
    )
    
    if config.NUM_WORKERS > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_ds,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              **loader_kwargs)
    val_loader   = DataLoader(val_ds,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              **loader_kwargs)
    test_loader  = DataLoader(test_ds,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              **loader_kwargs)

    print(f"[DATA] Train batches : {len(train_loader)}")
    print(f"[DATA] Val   batches : {len(val_loader)}")
    print(f"[DATA] Test  batches : {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_ds.get_attr_names()
