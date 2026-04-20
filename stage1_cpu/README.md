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

## 🗂️ Project Directory & File Structure
The codebase is strictly modular to enforce clean engineering practices and separate concerns. Here is exactly what is inside this stage:

### 📄 Core Code Files
* **`config.py`**: The single source of truth. Contains dynamic `pathlib` paths, `TRAIN_MODE="subset"`, learning rates, epochs (15), and the `128x128` resolution toggle.
* **`dataset.py`**: A memory-efficient custom PyTorch `Dataset`. It maps the text annotations to tensors and loads images **one by one** from the disk (via PIL) only when `__getitem__` is called, rather than holding 50k tensors in RAM. Implements simple horizontal flipping.
* **`model.py`**: Defines the `SimpleCNN_CPU` architecture, the pooling layers, and the Kaiming Normal weight initialization logic.
* **`train.py`**: The central training loop. It features PyTorch's `BCEWithLogitsLoss` for multi-label logic, tracking loss/accuracy, and applies the `ReduceLROnPlateau` scheduler.
* **`evaluate.py`**: A standalone script used to test a saved model strictly on the official CelebA validation split to get isolated precision/recall metrics.
* **`predict.py`**: A deployment script to test the model on unseen raw `.jpg` files locally.
* **`utils.py`**: Contains helper functions for cleanly plotting and saving matplotlib graphs.
* **`preflight.py`**: A sanity-check script to verify your CUDA/CPU environment and dataset paths are valid before starting a 5-hour training run.
* **`requirements.txt`**: The `pip` dependencies specifically required for this stage.

### 📁 Generated Output Directories
* **`checkpoints/`**: The directory dedicated to storing model weight snapshots during training.
  * `best_model_stage1.pth` (4.59 MB): This is the absolute best performing model throughout the entire 15 epoch run based on the lowest Validation Loss. What makes this file larger than raw weights is that it stores the entire dictionary state (`state_dict`) of both the CNN parameters *and* the `Adam` optimizer's momentum buffers.
  * `checkpoint_latest.pth` (4.59 MB): A crash-recovery save file. It simply stores the exact state of the network at the very end of the most recently completed epoch.
* **`outputs/`**: A programmatic logging directory generated automatically during training by `utils.py`.
  * `training_log.csv`: A permanent record of epoch progress. It contains tabulated columns for `Epoch, Train Loss, Val Loss, Accuracy, F1 Score, Precision, Recall, Time`.
  * `training_summary.txt`: A clean text readout of the run (verifying the 5 hour 6 minute completion time, and pointing to epoch 14 as the best iteration).
  * `loss_curve.png` & `accuracy_curve.png`: Matplotlib charts graphing the training vs. validation metrics over time, proving that the model was smoothly converging without severe overfitting.

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
