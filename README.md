# CelebA Multi-Label Attribute Classification: A Deep Learning Journey

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![GPU Accelerated](https://img.shields.io/badge/GPU-RTX_4070-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

This repository documents the end-to-end engineering journey of building, training, and heavily optimizing a **Multi-Label Convolutional Neural Network (CNN)** entirely from scratch (without pre-trained weights) to predict 40 distinct facial attributes using the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

Instead of hiding the struggle and only presenting the final code, this repository is deliberately structured into **three chronological stages**. It serves as a comprehensive portfolio piece demonstrating the process of scaling Deep Learning infrastructure—moving from heavy constraints on a local CPU to an optimized, high-performance GPU pipeline.

---

## 📂 Repository Architecture

The repository is modular and self-contained. The official annotation text files (`list_attr_celeba.txt`, `list_eval_partition.txt`) are already included inside `data/annotations/`. 

```text
celeba-journey-cpu-to-gpu/
├── stage1_cpu/              # Phase 1: Local CPU (50k image subset, lightweight architecture)
├── stage2_gpu_modular/      # Phase 2: College RTX 4070 (162k full dataset, Auto Mixed Precision)
├── stage3_gpu_single/       # Phase 3: Final Optimized Pipeline (AdamW, Augmentation, Warmup) 🏆
├── data/                    # Included official CelebA attribute configurations
├── test_images/             # Completely unseen demo images for real-world inference
├── compare_all_models.py    # The 4-way dynamic model comparison engine
└── README.md
```

*(Note: Data paths are dynamically resolved via `pathlib`. You **only** need to download the `img_align_celeba` image directory and set the `IMAGE_DIR` variable in the code to get started).*

---

## 🚀 The 3-Stage Engineering Journey

### Phase 1: `stage1_cpu/` (The Proof of Concept)
* **The Goal**: Validate the `Dataset` processing logic, attribute mapping, and Custom CNN forward-passes locally without the luxury of GPU hardware.
* **The Problem**: Passing a 3GB dataset of 202,599 high-res images through a neural network on a local CPU takes days.
* **The Solution**: We implemented a heavy bottleneck. The `DataLoader` was clamped to a **50,000 image subset**, images were drastically downscaled to **128x128px**, and we designed `SimpleCNN_CPU`, a lightweight architecture with only ~400k parameters.
* **The Result**: Achieved **90.55% validation accuracy** after an arduous **5-hour** training run. The model could detect broad traits (*Male*, *Smiling*), but failed completely on nuanced features (*Goatee*, *Arched Eyebrows*) due to the limited visual data.

### Phase 2: `stage2_gpu_modular/` (Scaling the Pipeline)
* **The Goal**: Migrate the code to a college laboratory RTX 4070 (12GB VRAM), train on the **full dataset** (162,770 training images), and increase the architecture's depth.
* **The Implementation**: We scaled the resolution back up to **160x160px** and expanded the network to `SimpleCNN_GPU` (~1 Million parameters). To fit `batch_sizes` of 160 into VRAM, we wrapped the training loop in `torch.amp.GradScaler` (Automatic Mixed Precision).
* **The Result**: Training speed plummeted from 5 hours to **21 minutes**. Validation accuracy hit **91.47%**. However, analyzing the inference revealed a "Conservative Prediction" flaw: the model maximized its loss function by guessing "No" (0) on rare attributes, heavily skewing towards high precision but terrible recall.

### Phase 3: `stage3_gpu_single/` (The Final Mastery) 🏆
* **The Goal**: Eradicate the conservative prediction flaw, maximize nuanced attribute detection, and merge the sprawling multi-file codebase into a clean, highly documented master-script.
* **The Implementation**: We executed a suite of advanced Deep Learning mechanics:
  * **Optimizer**: Replaced standard `Adam` with `AdamW` (Weight Decay) to aggressively regularize the network and prevent majority-class overfitting.
  * **LR Warmup**: Wrote a custom linear 3-epoch warmup scheduling block to prevent gradient explosions.
  * **Data Augmentation**: Built a dynamic `PIL` engine applying randomized Flips, Brightness, Contrast, and Rotations per epoch.
  * **Threshold Math**: Algorithmically scanned the validation probabilities and lowered the Sigmoid activation threshold from `0.50` to an empirically proven F1-optimal `0.40`.
* **The Result**: A flawless inference engine. The Stage 3 model correctly identifies incredibly subtle attributes (*Receding Hairlines*, *Bags Under Eyes*, *Heavy Makeup*) without hallucinating false positives.

---

## 🧠 Custom CNN Architectures

Both networks were written strictly using PyTorch `nn.Module` primitives, intentionally avoiding pre-trained models like ResNet to completely understand the backpropagation behavior.

| Dimension | Stage 1 (CPU-Light) | Stage 2 & 3 (GPU-Standard) |
|---|---|---|
| **Input Res** | `128x128` | `160x160` |
| **Channels** | `3 → 32 → 64 → 128 → 256` | `3 → 64 → 128 → 256 → 256` |
| **Max Pooling** | `2x2`, Stride 2 | `2x2`, Stride 2 |
| **Linear Head** | AdaptiveAvgPool2d + Dropout(0.4) + FC(40) | AdaptiveAvgPool2d + Dropout(0.4) + FC(40) |
| **Total Params** | `399,176` | `971,880` |

---

## 📊 Global Inference Benchmark

You can dynamically evaluate all checkpoints compiled across this project's lifespan side-by-side using the `compare_all_models.py` engine. 

*(Here are the inference test results evaluating 4 distinct models against unseen internet images:)*

| Unseen Image Subject | CPU-Stage 1 | GPU-Stage 2 | GPU-Best (Stage 3) 🏆 |
|---|---|---|---|
| **Dwayne Johnson** (Bald) | 5/5 expected | 3/5 expected | **5/5 expected** *(18 attrs total)* |
| **Shraddha Kapoor** (Makeup) | 6/6 expected | 5/6 expected | **6/6 expected** *(19 attrs total)* |
| **Older Bald Male** | 4/5 expected | 1/5 expected ❌ | **5/5 expected** *(11 attrs total)* |
| **Young Indian Male** | 5/7 expected | 7/7 expected | **7/7 expected** *(16 attrs total)* |

> **Key Takeaway:** You can clearly see Stage 2's conservative flaw (missing that the older man was bald entirely!). Stage 3 completely remedies this through `AdamW` and heavy data augmentation, catching every single expected attribute perfectly.

---

## 💻 How to Run Locally

### 1. Download the Dataset
The raw images are NOT included in this repository (it's 3GB). 
You must download the `img_align_celeba.zip` package from the [Official CelebA Repository](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and extract it to a folder on your machine (e.g. `C:\img_align_celeba`).

### 2. Run the Comparison Benchmark
Ensure `torch`, `torchvision`, `pandas`, `numpy`, and `Pillow` are installed.
```bash
# This script will auto-detect paths and run immediately
python compare_all_models.py
```

### 3. Run the Best Training Script
If you want to train the final mastery model from scratch on your own GPU:
1. Open `stage3_gpu_single/celeba_full_gpu_training.py`
2. Change line 28 to point to your extracted dataset: `IMAGE_DIR = r'C:\Your\Path\To\img_align_celeba'`
3. Execute:
```bash
python stage3_gpu_single/celeba_full_gpu_training.py
```
*(Optionally: You can also open the `CelebA_Training_Notebook.ipynb` in that same folder to read a cell-by-cell breakdown of the whole architecture).*
