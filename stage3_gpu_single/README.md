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

## 🗂️ Project Directory & File Structure
Instead of juggling 5 different Python modular scripts, everything was deliberately merged for execution simplicity. Here is what is inside this stage:

### 📄 Core Code Files
* **`celeba_full_gpu_training.py`**: The master 680-line Python script. It handles everything: configurations, `DataFrame` parsing, the PyTorch `Dataset` class object, the `SimpleCNN` network declaration, validation threshold tuning, and the overarching training loop.
* **`CelebA_Training_Notebook.ipynb`**: A beautifully documented equivalent of the master script, broken down cell-by-cell with markdown for interactive learning. 
  * *Note on Notebook Issues:* During early development, the notebook's training loop crashed with an `ImportError: IProgress not found`. This occurred because the interactive `tqdm` progress bar required the `ipywidgets` library, which is typically missing from base headless PyTorch environments. We resolved this by migrating back to a standard text-based progress output to ensure the notebook runs cleanly out-of-the-box anywhere and it was having num_workers issues which causes low data loading during training so not efficient in .ipynb.

### 📁 Generated Output Directories
* **`checkpoints/`**: The active weight checkpoint directory.
  * `best_model_gpu.pth` (3.72 MB): The absolute best performing version of the network. Because we cleaned up the save mechanics in the master script, we optimized the checkpoint to **only** store the raw `model.state_dict()`. Stripping out the heavy AdamW optimizer momentum buffers dropped the file size from 11.15 MB back down to a lean 3.72 MB!
  * `last_model_gpu.pth` (3.72 MB): The final epoch's raw weights, saved for post-training evaluation.
* **`outputs/`**: The offline metric directory.
  * `training_curves_gpu.png`: A comprehensive 4-panel visual chart plotting Loss, Accuracy, F1 Score, and the dynamically adjusting Learning Rate schedule.
  * `test_predictions_gpu.npy` & `test_labels_gpu.npy`: The raw `float32` tensor predictions automatically exported to numpy files after evaluation. This allows analysts to mathematically evaluate the model's exact confidence scores across all 40 attributes without having to load up PyTorch or a GPU again!

## 📊 Final Results
The results speak for themselves. In the global `compare_all_models.py` benchmark, Stage 3 absolutely demolished the previous models. It accurately tags up to 16 out of 16 expected attributes on totally unseen images, correctly registering *5 o' Clock Shadows*, *Heavy Makeup*, *Sideburns*, and *Receding Hairlines* that the Stage 2 GPU model completely ignored.

## 🚀 How to Run
Ensure `IMAGE_DIR` on line 28 of `celeba_full_gpu_training.py` (or inside the notebook) points to your active image directory.
```bash
python celeba_full_gpu_training.py
```
