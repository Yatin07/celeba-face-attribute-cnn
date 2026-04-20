# Stage 2: Modular GPU Training (Scaling Pipeline)

This directory represents the migration phase: moving the local codebase to a college laboratory workstation equipped with an **NVIDIA RTX 4070 (12GB VRAM)**. 

## 🎯 Objective
To unleash the pipeline on the entirety of the CelebA dataset, significantly increase the depth of the neural network to capture fine facial features, and leverage modern GPU hardware acceleration techniques to minimize training loops from hours to minutes.

## 🛠️ The Scaling Process

Because the codebase from Stage 1 was cleanly separated via `config.py`, scaling required no major rewrites—only configuration adjustments:

1. **Full Dataset Extraction (`TRAIN_MODE = "full"`):** 
   The pipeline now processes the entire training split of **162,770 images**.
2. **Resolution Restoration:** 
   The target `IMAGE_SIZE` was raised back to the standard **`160x160` pixels**, allowing the convolutional filters to detect smaller attributes like *Arched Eyebrows* or *Mustache*.
3. **Architecture Scaling (`SimpleCNN_GPU`):**
   The neural network in `model.py` was deepened to process the larger resolution.
   * **Channels:** `3 → 64 → 128 → 256 → 256`
   * **Parameters:** `971,880` (A 2.5x increase in trainable parameters).
   * **Model File Size:** `best_model_stage2.pth` is `11.15 MB` (Adam stores 2 moment buffers per parameter, causing the bloated disk size).

## ⚡ Hardware Acceleration Techniques
To fit `160x160` batches of size `160` into 12GB of VRAM and keep the GPU engaged, we introduced:
* **Automatic Mixed Precision (AMP):** Wrapped the forward passes in `torch.amp.autocast`. This computes layers using `float16` instead of `float32`, cutting memory load in half and accelerating tensor math, while `torch.amp.GradScaler` safely scales gradients to prevent underflow.
* **Persistent Workers:** Utilizing `num_workers=4` and `prefetch_factor=2` inside the `DataLoader` to allow the CPU to constantly pipeline images directly into the GPU safely.

## 🗂️ Project Directory & File Structure
Despite migrating to a GPU, the core architecture remains modular. Here is exactly what is inside this stage:

### 📄 Core Code Files
* **`config.py`**: The single source of truth. Contains dynamic `pathlib` paths, `TRAIN_MODE="full"`, epochs (20), GPU settings, and the scaled `160x160` resolution toggle.
* **`dataset.py`**: Similar to Stage 1, but processes all 162k images.
* **`model.py`**: Defines the deeper `SimpleCNN_GPU` architecture.
* **`train.py`**: The central training loop, now heavily modified to implement PyTorch's Automatic Mixed Precision (`GradScaler`).
* **`evaluate.py` & `predict.py`**: Standard validation and inference scripts similar to Stage 1.
* **`evaluate_model.py`**: A newly introduced, robust evaluation framework designed specifically to compare checkpoint anomalies (Best vs. Last Epochs) to diagnose the conservative prediction issue.
* **`predict_single_image.py`**: A specialized deployment script built to execute fast, visually annotated inferences on test images.
* **`utils.py`**, **`preflight.py`**, **`requirements.txt`**: Helper files and GPU-dependency configs.

### 📁 Generated Output Directories
* **`checkpoints/`**: The directory dedicated to storing model weight snapshots during training.
  * `best_model_stage2.pth` (11.15 MB): The absolute best performing model across the 18 epochs based on Validation Loss. The file ballooned to 11.15 MB because the `Adam` optimizer calculates and stores two moment vectors per parameter across nearly a million parameters.
  * `checkpoint_latest.pth` (11.15 MB): The auto-saved exact state of the network at the very end of the run (used for resuming training).
* **`outputs/`**: A programmatic logging directory generated automatically during training by `utils.py`.
  * `training_log.csv`: A permanent record of epoch progress with an additional column tracking the dynamic `Learning Rate`.
  * `training_summary.txt`: A clean text readout of the run (verifying the rapid 21 minute and 24 second completion time).
  * `loss_curve.png` & `accuracy_curve.png`: Matplotlib charts showcasing the massively accelerated learning curve due to the AMP implementation.

## 📊 Results and The Inference Problem
The performance gain was staggering. The model swept through 18 epochs in just **21 minutes and 24 seconds**.
* **Best Validation Loss:** `0.1950` (Epoch 15)
* **Validation Accuracy:** `91.47%`

**The Inference Bottleneck:**
Despite better on-paper validation metrics, real-world testing via `compare_all_models.py` revealed a major problem. The model became highly **conservative**. It prioritized driving the Loss function down by guessing "No" (0) for rare attributes. It missed predicting a completely *Bald* man as being bald because it wasn't statically confident enough to push the Sigmoid activation past `0.5`, leading to high precision but terribly low recall.

## 🚀 How to Run
Ensure the `IMAGE_DIR` in `config.py` correctly points to your extracted `img_align_celeba` directory.
```bash
python train.py
```
