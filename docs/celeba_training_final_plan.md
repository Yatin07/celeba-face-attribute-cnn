# 🎯 CelebA Chunked Training — FINAL SETTLED PLAN
> Consolidated from multi-round review. This is the definitive reference before coding.

---

## 📊 Dataset Facts (Know Before You Code)

| Item | Value |
|------|-------|
| Total images | 202,599 |
| Train split | 162,770 |
| Validation split | 19,867 (official CelebA split) |
| Test split | 19,962 (official CelebA split) |
| Attributes | 40 binary labels per image |
| Label values | +1 / -1 in raw file → convert to 1 / 0 |
| Split file | `list_eval_partition.txt` (0=train, 1=val, 2=test) |

> ⚠️ Use the OFFICIAL CelebA splits — do NOT manually split.
> The splits are predefined in `list_eval_partition.txt`.

---

## 🔥 STEP 1: Data Preparation

### 1a. Load & Convert Labels
```python
import pandas as pd

# Load attributes
attr = pd.read_csv('list_attr_celeba.txt', sep=r'\s+', header=1)
attr = (attr + 1) // 2  # Convert -1/+1 → 0/1  ← MOST COMMON SILENT BUG

# Load official splits
splits = pd.read_csv('list_eval_partition.txt', sep=' ', header=None,
                     names=['image_id', 'split'])

# Merge
df = splits.merge(
    attr.reset_index().rename(columns={'index': 'image_id'}),
    on='image_id'
)
```

### 1b. Separate Train / Val / Test using Official Splits
```python
train_df = df[df['split'] == 0].reset_index(drop=True)  # 162,770
val_df   = df[df['split'] == 1].reset_index(drop=True)  # 19,867
test_df  = df[df['split'] == 2].reset_index(drop=True)  # 19,962
```

### 1c. Shuffle Training Set ONCE
```python
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# ✅ Done ONCE — never shuffle again between epochs
```

### 1d. Split Into Chunks (No Overlap, Full Coverage)
```python
CHUNK_SIZE = 50_000
chunks = [train_df.iloc[i:i+CHUNK_SIZE] for i in range(0, len(train_df), CHUNK_SIZE)]
# → 4 chunks: 50,000 | 50,000 | 50,000 | 12,770

# ✅ Verify no missing / no overlap
assert sum(len(c) for c in chunks) == len(train_df), "Missing images!"
all_ids = [id_ for c in chunks for id_ in c['image_id']]
assert len(all_ids) == len(set(all_ids)), "Overlap detected!"
```

---

## 🔥 STEP 2: Distribution Check (Run ONCE Before Training)

```python
attr_cols = [c for c in train_df.columns if c not in ['image_id', 'split']]

global_means = train_df[attr_cols].mean()
print("Global means (reference):")
print(global_means[['Smiling', 'Male', 'Bald', 'Wearing_Hat']].round(3))

for i, chunk in enumerate(chunks):
    chunk_means = chunk[attr_cols].mean()
    diff = (chunk_means - global_means).abs()
    worst = diff.nlargest(3)
    print(f"\nChunk {i+1} ({len(chunk)} images) — worst deviations:")
    print(worst.round(4))
```

> ✅ Values should be within ±2% of global mean.
> 🚨 If any attribute differs by >5% → shuffle failed, re-check.

---

## 🔥 STEP 3: I/O Setup (Before Training Starts)

```python
import os, shutil

SRC = '/content/drive/MyDrive/celeba/img_align_celeba/'
DST = '/content/celeba/img_align_celeba/'

if not os.path.exists(DST):
    print("Copying dataset to local SSD...")
    shutil.copytree(SRC, DST)
    print("✅ Done — dataset on local storage")
else:
    print("✅ Dataset already on local storage")
```

> **Why:** Google Drive = 200k file open/close per epoch = major bottleneck.
> Colab local SSD is ~10x faster. LMDB/WebDataset NOT recommended — too complex.

---

## 🔥 STEP 4: Dataset Class

```python
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T

class CelebAChunk(Dataset):
    def __init__(self, chunk_df, img_dir, transform=None):
        self.df = chunk_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.attr_cols = [c for c in chunk_df.columns
                          if c not in ['image_id', 'split']]
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['image_id']}").convert('RGB')
        label = torch.tensor(
            row[self.attr_cols].values.astype(float),
            dtype=torch.float32
        )
        return self.transform(img), label
```

---

## 🔥 STEP 5: Weighted Loss (Critical for CelebA Imbalance)

```python
import torch.nn as nn

pos_counts = train_df[attr_cols].sum()
neg_counts = len(train_df) - pos_counts
pos_weight = neg_counts / pos_counts

# ✅ Clip at 10x max — prevents overcompensation for very rare attributes
# Without clipping: Bald → ~42.5x → model over-predicts bald on every image
pos_weight = pos_weight.clip(upper=10.0)

pos_weight_tensor = torch.tensor(
    pos_weight.values, dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

**Attribute imbalance reference:**

| Attribute    | Prevalence | Raw pos_weight | After clip |
|--------------|-----------|----------------|------------|
| Smiling      | 48%       | 1.08x          | 1.08x      |
| Male         | 42%       | 1.38x          | 1.38x      |
| Bald         | 2.3%      | ~42.5x         | **10x**    |
| Wearing_Hat  | 4.7%      | ~20.3x         | **10x**    |
| Rosy_Cheeks  | 6.5%      | ~14.4x         | **10x**    |

> Monitor per-label precision/recall after training — NOT just overall accuracy.

---

## 🔥 STEP 6: Training Loop — FINAL SETTLED VERSION

```python
from torch.utils.data import DataLoader
import gc

# ── Config ───────────────────────────────────────────────────────
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR         = 1e-3
IMG_DIR    = '/content/celeba/img_align_celeba/'
CKPT_PATH  = '/content/drive/MyDrive/celeba/checkpoint.pth'   # on Drive = survives reset
NUM_ATTRS  = 40
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Model + Optimizer + Scheduler ────────────────────────────────
model     = YourCNN(num_classes=NUM_ATTRS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS)
# ⚠️ scheduler.step() called ONCE PER EPOCH — never inside chunk loop

# ── Resume from checkpoint if exists ─────────────────────────────
start_epoch = 0
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"✅ Resumed from epoch {start_epoch}")

# ── Validation loader (19,867 images — load FULL, no chunking) ───
val_dataset = CelebAChunk(val_df, IMG_DIR)
val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=True)

# ── Training ─────────────────────────────────────────────────────
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss    = 0.0
    total_batches = 0

    for chunk_idx, chunk_df in enumerate(chunks):      # ← FIXED order every epoch
        dataset = CelebAChunk(chunk_df, IMG_DIR)
        loader  = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,           # ← re-shuffle within chunk each epoch
            drop_last=True,         # ← stable BatchNorm, tiny data loss (<1%)
            num_workers=2,
            pin_memory=True
        )

        chunk_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            chunk_loss += loss.item()

        epoch_loss    += chunk_loss
        total_batches += len(loader)
        print(f"  Epoch {epoch+1} | Chunk {chunk_idx+1}/{len(chunks)} | "
              f"Loss: {chunk_loss/len(loader):.4f}")

        # ── Memory cleanup between chunks (CRITICAL for Colab) ────
        del dataset, loader
        gc.collect()
        torch.cuda.empty_cache()

    # ── Validate AFTER all chunks (not per chunk) ─────────────────
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            val_loss += criterion(outputs, labels).item()
            preds    = (torch.sigmoid(outputs) > 0.5).float()
            correct  += (preds == labels).sum().item()
            total    += labels.numel()

    val_acc = correct / total * 100
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Complete")
    print(f"  Train Loss : {epoch_loss/total_batches:.4f}")
    print(f"  Val Loss   : {val_loss/len(val_loader):.4f}")
    print(f"  Val Acc    : {val_acc:.2f}%")
    print(f"{'='*60}\n")

    # ── Step scheduler ONCE per epoch ─────────────────────────────
    scheduler.step()

    # ── Save checkpoint to Drive (survives Colab session reset) ───
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc':              val_acc,
    }, CKPT_PATH)
    print(f"✅ Checkpoint saved → epoch {epoch+1}\n")
```

---

## 🔥 STEP 7: Final Testing (After All Epochs)

```python
test_dataset = CelebAChunk(test_df, IMG_DIR)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False,
                          num_workers=2, pin_memory=True)

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds    = (torch.sigmoid(model(images)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.numel()

print(f"Final Test Accuracy: {correct/total*100:.2f}%")
```

---

## ✅ ALL SETTLED DECISIONS (FINAL TABLE)

| Decision | Choice | Reason |
|----------|--------|--------|
| Dataset splits | Official (`list_eval_partition.txt`) | Reproducible, standard |
| Label conversion | -1/+1 → 0/1 | BCELoss requires 0/1 |
| Shuffle | Once, `random_state=42` | Fixed distribution, reproducible |
| Chunk size | 50k / 50k / 50k / 12,770 | Colab RAM safe |
| Chunk order | Fixed every epoch | Stable, reproducible |
| DataLoader shuffle | `True` | Per-epoch intra-chunk randomness |
| drop_last | `True` | Stable BatchNorm, <1% data loss |
| num_workers | 2 | Parallel prefetch |
| pin_memory | `True` | Faster GPU transfer |
| I/O | Copy to `/content/` | ~10x faster than Drive |
| Loss function | BCEWithLogitsLoss + pos_weight | 40-label imbalance |
| pos_weight cap | 10x max | Prevents overcompensation |
| Validation set | Full load (no chunking) | 19,867 fits RAM easily |
| Validate when | After ALL chunks per epoch | Stable, accurate metrics |
| Checkpoint | Per epoch, saved to Drive | Survives session reset |
| Scheduler | CosineAnnealingLR, step per epoch | Smooth LR decay |
| Memory cleanup | del + gc.collect() + cuda.empty_cache() | Prevent chunk RAM buildup |

---

## 🚨 PRE-TRAINING CHECKLIST (Verify ALL before running)

- [ ] `sum(len(c) for c in chunks) == 162770` → no missing images
- [ ] No image_id in more than one chunk → no overlap
- [ ] Labels converted from -1/+1 → **0/1** ← most common silent bug
- [ ] Distribution check done → no attribute deviates >5% across chunks
- [ ] Dataset copied to `/content/` → NOT reading from Drive
- [ ] `torch.cuda.is_available() == True` → GPU confirmed
- [ ] Checkpoint path is on **Drive** → survives session reset
- [ ] `pos_weight_tensor` moved to correct device with `.to(device)`
- [ ] `scheduler.step()` is OUTSIDE the chunk loop, INSIDE epoch loop

---

## 📈 Expected Training Behavior

| Epoch | Expected Val Acc | Notes |
|-------|-----------------|-------|
| 1     | 75–80%          | Learning basic features |
| 3–5   | 85–88%          | Main convergence zone |
| 7–10  | 88–91%          | Diminishing returns |

> ⚠️ **Overall accuracy is misleading for CelebA.**
> A model predicting always-0 for Bald gets 97.7% accuracy on that label alone.
> Always check **per-label precision and recall**, especially:
> `Bald`, `Wearing_Hat`, `Rosy_Cheeks`, `Sideburns`, `Narrow_Eyes`

---

*Review this plan → then next step: CNN architecture design.*
