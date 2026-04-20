# CelebA Local Training Pipeline — README

## Project Overview

A complete, CPU-friendly deep learning pipeline to train a CNN from scratch
on the CelebA dataset for 40-attribute multi-label classification.

---

## Project Structure

```
C:\MLA\CelebA_Local\
│
├── config.py        ← All settings (paths, modes, hyperparams)
├── dataset.py       ← Data loading + PyTorch Dataset/DataLoaders
├── model.py         ← SimpleCNN architecture (4 conv blocks, ~400K params)
├── train.py         ← Training loop (early stopping, checkpointing)
├── evaluate.py      ← Test evaluation (loss, accuracy, F1, per-label)
├── predict.py       ← Single-image inference with confidence scores
├── utils.py         ← Metrics, plots, checkpoint helpers
├── requirements.txt ← Python dependencies
│
├── checkpoints/
│   ├── checkpoint_latest.pth      ← Saved every epoch (for crash recovery)
│   ├── best_model_stage1.pth      ← Best model from 50k training
│   └── best_model_stage2.pth      ← Best model from full 162k training
│
└── outputs/
    ├── loss_curve.png             ← Train vs Val loss plot
    └── accuracy_curve.png         ← Validation accuracy plot
```

---

## Dataset Required

```
C:\MLA\celeba\
├── img_align_celeba\        ← 202,599 face images (.jpg)
├── list_attr_celeba.txt     ← 40 attribute labels per image
└── list_eval_partition.txt  ← Official train/val/test split
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Stage 1 — Train on 50k images

In `config.py`:
```python
TRAIN_MODE      = "subset"
RESUME_TRAINING = False
```

```bash
python train.py
```

### Evaluate on test set

```bash
python evaluate.py
```

### Predict attributes for a single image

```bash
python predict.py --image path\to\face.jpg
python predict.py --image path\to\face.jpg --topk 10
```

---

## Stage 2: Continue Training on Full Dataset

After Stage 1 is complete, update `config.py`:
```python
TRAIN_MODE      = "full"
RESUME_TRAINING = True
```

Run:
```bash
python train.py
```

This loads `best_model_stage1.pth`, reduces LR to 0.0001, and saves the best result as `best_model_stage2.pth`.

---

## Architecture: SimpleCNN (~400K parameters)

```
Input  : (B, 3, 128, 128)

Block 1: Conv(3→32)   + BN + ReLU + MaxPool → (B,  32, 64, 64)
Block 2: Conv(32→64)  + BN + ReLU + MaxPool → (B,  64, 32, 32)
Block 3: Conv(64→128) + BN + ReLU + MaxPool → (B, 128, 16, 16)
Block 4: Conv(128→256)+ BN + ReLU + MaxPool → (B, 256,  8,  8)

GlobalAvgPool                               → (B, 256)
Dropout(0.4)
FC(256 → 40)                                → (B, 40) raw logits
```

---

## Key Config Parameters

| Parameter | Value | Reason |
|---|---|---|
| IMAGE_SIZE | 128 | Standard for CelebA |
| BATCH_SIZE | 64 | Good for CPU RAM |
| NUM_WORKERS | 0 | Stable on Windows CPU |
| PIN_MEMORY | False | No GPU |
| EPOCHS | 15 | Max (early stopping applies) |
| PATIENCE | 3 | Stop after 3 no-improvement epochs |
| LEARNING_RATE | 0.001 | Adam default |
| RESUME_LR | 0.0001 | Reduced for Stage 2 |
| DROPOUT | 0.4 | Prevents overfitting |

---

## Time Estimate (Intel Iris CPU)

| Phase | Per Epoch | Total (15 epochs) |
|---|---|---|
| Training (50k images) | ~15–25 min | ~5–8 hours |
| Validation (~20k images) | ~5–8 min | per epoch |
| Test evaluation | — | ~10 min (once) |

> Training runs overnight. Early stopping usually triggers before epoch 15.
