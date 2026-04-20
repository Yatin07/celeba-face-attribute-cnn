# CelebA Multi-Label Classification - Project Context

## Project Overview
This is a PyTorch-based deep learning project for multi-label classification on the CelebA dataset (CelebFaces Attributes Dataset). The goal is to predict 40 binary facial attributes from celebrity face images.

## Dataset Information
- **Name**: CelebA (CelebFaces Attributes Dataset)
- **Total Images**: 202,599 celebrity face images
- **Image Size**: 178x218 pixels (original), resized to 160x160 for training
- **Attributes**: 40 binary labels per image
- **Splits**:
  - Training: 162,770 images
  - Validation: 19,867 images
  - Test: 19,961 images
- **Labels**: {-1, +1} converted to {0, 1} for binary classification

## 40 Facial Attributes (in order)
1. 5_o_Clock_Shadow
2. Arched_Eyebrows
3. Attractive
4. Bags_Under_Eyes
5. Bald
6. Bangs
7. Big_Lips
8. Big_Nose
9. Black_Hair
10. Blond_Hair
11. Blurry
12. Brown_Hair
13. Bushy_Eyebrows
14. Chubby
15. Double_Chin
16. Eyeglasses
17. Goatee
18. Gray_Hair
19. Heavy_Makeup
20. High_Cheekbones
21. Male
22. Mouth_Slightly_Open
23. Mustache
24. Narrow_Eyes
25. No_Beard
26. Oval_Face
27. Pale_Skin
28. Pointy_Nose
29. Receding_Hairline
30. Rosy_Cheeks
31. Sideburns
32. Smiling
33. Straight_Hair
34. Wavy_Hair
35. Wearing_Earrings
36. Wearing_Hat
37. Wearing_Lipstick
38. Wearing_Necklace
39. Wearing_Necktie
40. Young

## Model Architecture: SimpleCNN
```
Input: 160x160x3 (RGB image)
↓
Block 1: Conv(3→64) + BN + ReLU + MaxPool → 80x80x64
↓
Block 2: Conv(64→128) + BN + ReLU + MaxPool → 40x40x128
↓
Block 3: Conv(128→256) + BN + ReLU + MaxPool → 20x20x256
↓
Block 4: Conv(256→256) + BN + ReLU + MaxPool → 10x10x256
↓
Global Average Pooling → 1x1x256
↓
Dropout (p=0.4)
↓
Linear(256→40) → 40 logits (one per attribute)
↓
Sigmoid (applied during inference) for probabilities
```

**Parameters**: ~400K (small, efficient model)

## Training Configuration
- **Framework**: PyTorch 2.x
- **Device**: CUDA (GPU) with Automatic Mixed Precision (AMP)
- **Batch Size**: 160
- **Epochs**: 20 (max), with early stopping
- **Optimizer**: AdamW (learning rate: 1e-3)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Loss Function**: BCEWithLogitsLoss with pos_weight (handles class imbalance)
- **Warmup**: 3 epochs with linear warmup from 0.3x to 1.0x LR
- **Gradient Clipping**: max_norm=1.0
- **Data Augmentation**:
  - Horizontal flip (50%)
  - Brightness adjustment (0.8x-1.2x)
  - Contrast adjustment (0.8x-1.2x)
  - Rotation (-10° to +10°)
- **Early Stopping**: Patience=3 epochs based on validation loss
- **TF32 Enabled**: True (for Ampere GPUs: RTX 30xx/40xx)

## Key Features & Optimizations
1. **Modern AMP API**: Uses `torch.amp.autocast` and `torch.amp.GradScaler` (PyTorch 2.x)
2. **Class Imbalance Handling**: pos_weight computed from training data, clamped at max=10.0
3. **LR Warmup**: Stable training start
4. **Non-blocking Transfers**: `images.to(device, non_blocking=True)`
5. **Threshold Tuning**: Post-training search over [0.3, 0.35, 0.4, 0.45, 0.5, 0.55] for best F1
6. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, per-class metrics

## File Structure
```
CelebA_Local_GPU/
├── celeba_full_gpu_training.py    # Main training script (GPU optimized)
├── CelebA_Training_Notebook.ipynb # Jupyter notebook version
├── evaluate_model.py              # Evaluate on test set
├── predict_single_image.py        # Predict on single image
├── config.py                      # Configuration (if exists)
├── checkpoints/
│   ├── best_model_gpu.pth         # Best model (lowest val loss)
│   └── last_model_gpu.pth         # Last epoch model
└── outputs/
    ├── training_curves_gpu.png
    ├── test_predictions_gpu.npy
    ├── test_labels_gpu.npy
    └── prediction_[image_name].png
```

## Data File Paths (Windows)
- **Images**: `C:\MLA\img_align_celeba\`
- **Attributes**: `C:\MLA\drive-download-20260318T181243Z-1-001\list_attr_celeba.txt`
- **Partitions**: `C:\MLA\drive-download-20260318T181243Z-1-001\list_eval_partition.txt`

## Metrics Definition
- **Accuracy**: Percentage of correct predictions across all attributes
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many were found
- **F1 Score**: Harmonic mean of precision and recall
- **Per-class Accuracy**: Individual attribute accuracies

## Prediction Threshold
- Default: 0.4 (determined via validation set tuning)
- Range tested: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
- Selection criteria: Maximum F1 score

## Training Time Estimate
- Full dataset (162k images): ~3-4 hours on RTX 4070 Ti
- Per epoch: ~8-12 minutes
- With early stopping: ~2.5-3.5 hours

## GPU Requirements
- **Minimum**: 8GB VRAM (batch size 64)
- **Recommended**: 12GB VRAM (batch size 160)
- **AMP Enabled**: Reduces memory usage by ~30%

## Common Commands

### Training
```bash
python celeba_full_gpu_training.py
```

### Evaluation (compare best vs last)
```bash
python evaluate_model.py --model both
```

### Predict on single image
```bash
python predict_single_image.py "path/to/your/photo.jpg"
```

## Key Design Decisions
1. **Multi-label not Multi-class**: Each image can have multiple attributes (not mutually exclusive)
2. **BCEWithLogitsLoss**: Combines sigmoid + BCE for numerical stability
3. **Global Average Pooling**: Reduces parameters compared to flatten
4. **Dropout (0.4)**: Prevents overfitting on full dataset
5. **BatchNorm**: Stabilizes training and allows higher learning rates
6. **No Sigmoid in Forward**: Applied during inference only (BCEWithLogitsLoss handles it)

## Known Issues & Solutions
- **Class Imbalance**: Solved with pos_weight (some attributes like "Bald" are rare)
- **Overfitting**: Early stopping + dropout + data augmentation
- **Memory**: AMP + gradient scaling enables larger batch sizes
- **Jupyter Multiprocessing**: NUM_WORKERS=0 in notebook version

## Evaluation Results (Expected)
- **Accuracy**: ~90-92%
- **F1 Score**: ~85-88%
- **Best Attributes**: Male, Smiling, Young (~95%+ accuracy)
- **Hardest Attributes**: Bald, Wearing_Hat, Goatee (~70-80% accuracy)

## Model Output
- **Logits**: 40 raw values (pre-sigmoid)
- **Probabilities**: 40 values in [0, 1] after sigmoid
- **Predictions**: Binary 0/1 after thresholding

## Training Outputs
1. **best_model_gpu.pth**: Model with lowest validation loss (recommended)
2. **last_model_gpu.pth**: Model from final epoch
3. **training_curves_gpu.png**: Loss, accuracy, F1, LR plots
4. **test_predictions_gpu.npy**: Raw test predictions (numpy array)
5. **test_labels_gpu.npy**: Ground truth test labels

## Code Style
- Type hints where appropriate
- Docstrings for all functions
- TQDM progress bars for training
- Comprehensive logging with [TAGS]
- Error handling for image loading
- Non-blocking GPU transfers for speed
