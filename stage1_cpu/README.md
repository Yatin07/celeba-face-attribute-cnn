# Stage 1: CPU Training (The Proof of Concept)

This directory represents the foundational phase of the project: proving that a multi-label classification pipeline could be built from scratch and run successfully on a local CPU machine.

## 🎯 Objective
To build a functional PyTorch environment, establish the data loading logic to parse the text-based attribute annotations (`list_attr_celeba.txt`), and validate a custom Convolutional Neural Network (CNN) architecture without the memory and processing power of a GPU.

## 🛠️ Technical Constraints & Solutions

Training 202,599 high-resolution images across 40 distinct labels locally on a CPU is computationally infeasible. To solve this, Stage 1 implements intentional bottlenecks:

1. **Subset Training (`TRAIN_MODE = "subset"`):** 
   Configured in `config.py`, the training engine only loads the first **50,000 images** of the official train split. 
2. **Resolution Downscaling:** 
   Images are squashed from standard `160x160` to **`128x128` pixels** to reduce memory overhead and speed up tensor manipulation.
3. **Architecture Miniaturization:**
   We designed `SimpleCNN_CPU` in `model.py` to be intentionally shallow.
   * **Channels:** `3 → 32 → 64 → 128 → 256`
   * **Pooling:** Adaptive Average Pooling `(1,1)` making the model dimension-agnostic.
   * **Parameters:** `399,176` total parameters.
   * **Model File Size:** The saved `best_model_stage1.pth` is roughly `4.59 MB` (which includes the state_dict of the Adam optimizer).

## 🗂️ The Modular Engineering
The codebase is strictly modular to enforce clean engineering practices:
* **`config.py`**: The single source of truth for paths, learning rates, epochs, and toggles.
* **`dataset.py`**: A memory-efficient custom PyTorch Dataset that loads images *one by one* from the disk (via PIL) only when `__getitem__` is called, rather than holding 50k tensors in RAM. Implements simple horizontal flipping.
* **`train.py`**: The central loop featuring `BCEWithLogitsLoss` for multi-label logic, and a `ReduceLROnPlateau` scheduler.
* **`evaluate.py` & `predict.py`**: Standalone scoring and inference scripts.

## 📊 Results and Limitations
Training took **5 hours, 6 minutes, and 32 seconds** to run for 15 epochs. 
* **Best Validation Loss:** `0.2169` (Epoch 14)
* **Validation Accuracy:** `90.55%`

**Qualitative Analysis:** 
Upon hitting `compare_all_models.py` with unseen test images, this CPU model performs surprisingly well on "broad" categories (e.g., *Male*, *Smiling*, *Young*, *Black Hair*). However, because it only saw 50k images and lacked visual depth, it significantly underperforms on nuanced, rare attributes like *Goatee*, *Sideburns*, or *Bags Under Eyes*.

## 🚀 How to Run
Ensure the `IMAGE_DIR` in `config.py` correctly points to your extracted `img_align_celeba` directory.
```bash
python train.py
```
