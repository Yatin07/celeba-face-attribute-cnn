# Stage 3: Single-File GPU Optimization (The Final Pipeline) 🏆

This directory contains the ultimate iteration of the project. Having diagnosed the "conservative prediction" flaw in Stage 2, Stage 3 focuses on algorithmic optimizations, data augmentation, and threshold mathematics to extract the absolute maximum nuance from the CNN.

## 🎯 Objective
To fix the recall issues, to make the model deeply perceptive of rare attributes, and to refactor the entire, sprawling modular codebase into a single, clean, easily reproducible `Python` script and `Jupyter Notebook`.

## 🧠 Algorithmic Optimizations Introduced

### 1. The AdamW Optimizer
We ripped out standard `Adam` and replaced it with `AdamW` (`torch.optim.AdamW`). Multi-label classification on imbalanced datasets easily leads to overfitting on common classes. AdamW enforces decoupled weight decay, acting as a strict regularizer to force the network to learn generalized features rather than memorizing noise.

### 2. Learning Rate Warmup
Training deep networks on large batch sizes can cause massive gradient spikes in the first epoch that ruin the randomly initialized weights. We wrote a custom scheduler block that starts the `INITIAL_LR` at `0.3 × base_lr` and linearly scales it up over the first 3 epochs before handling control back to the `ReduceLROnPlateau` scheduler.

### 3. PIL Data Augmentation Pipeline
In Stage 1 & 2, we only used a 50% chance of a horizontal flip. This was not enough to prevent memorization. We built a native `augment_image()` function using Pillow (`ImageEnhance`) that applies:
* Random Flips (p=0.5)
* Dynamic Brightness (`0.8` to `1.2` multiplier)
* Dynamic Contrast (`0.8` to `1.2` multiplier)
* Small angle Rotations (`-10` to `+10` degrees)
This guarantees the model almost never sees the exact same image pixels twice across its 20 epochs.

### 4. Dynamic Threshold Tuning
By default, PyTorch forces binary classification by assuming a sigmoid probability `> 0.50` is a `Yes` and `< 0.50` is a `No`. We algorithmically evaluated the raw tensor probabilities on the validation set against thresholds from `0.30` to `0.55` and discovered empirically that a threshold of **`0.40`** maximized the F1 Harmonic Mean Score without breaking precision.

## 🗂️ The Single-File consolidation
Instead of juggling 5 different Python scripts, everything was merged into `celeba_full_gpu_training.py` (and perfectly documented cell-by-cell in `CelebA_Training_Notebook.ipynb`). 
* The `checkpoints/best_model_gpu.pth` only saves the `model.state_dict()` taking up a lean `3.72 MB` instead of `11.15 MB`.
* Prediction tensors (`.npy`) are automatically dumped to `outputs/` for future offline graphing.

## 📊 Final Results
The results speak for themselves. In the global `compare_all_models.py` benchmark, Stage 3 absolutely demolished the previous models. It accurately tags up to 16 out of 16 expected attributes on totally unseen images, correctly registering *5 o' Clock Shadows*, *Heavy Makeup*, *Sideburns*, and *Receding Hairlines* that the Stage 2 GPU model completely ignored.

## 🚀 How to Run
Ensure `IMAGE_DIR` on line 28 of `celeba_full_gpu_training.py` (or inside the notebook) points to your active image directory.
```bash
python celeba_full_gpu_training.py
```
