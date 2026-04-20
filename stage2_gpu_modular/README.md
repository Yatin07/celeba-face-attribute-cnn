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
