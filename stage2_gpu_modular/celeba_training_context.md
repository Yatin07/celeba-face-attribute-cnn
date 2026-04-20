# celeba_full_gpu_training.py - Context & Structure

## File Purpose
Main GPU training script for CelebA multi-label classification (40 facial attributes).
Trains on FULL 162,770 images with modern PyTorch 2.x optimizations.

## File Location
`c:/MLA/CelebA_Local_GPU/celeba_full_gpu_training.py`

## Related Files
```
CelebA_Local_GPU/
├── celeba_full_gpu_training.py      ← THIS FILE (training script)
├── checkpoints/best_model_gpu.pth   ← Output: Best model (lowest val loss)
├── checkpoints/last_model_gpu.pth   ← Output: Final epoch model
├── outputs/training_curves_gpu.png  ← Output: Loss/accuracy/F1/LR plots
├── outputs/test_predictions_gpu.npy ← Output: Test set predictions
├── outputs/test_labels_gpu.npy      ← Output: Test ground truth
├── evaluate_model.py                ← Evaluate saved models
├── predict_single_image.py          ← Predict on your photos
└── config.py                        ← Config (if separate)
```

## Required Input Data
- **Images**: `C:\MLA\img_align_celeba\*.jpg` (162,770 training images)
- **Attributes**: `C:\MLA\drive-download-20260318T181243Z-1-001\list_attr_celeba.txt`
- **Splits**: `C:\MLA\drive-download-20260318T181243Z-1-001\list_eval_partition.txt`

## Key Configuration (in-file)
```python
IMAGE_SIZE = 160        # 160x160 resolution
BATCH_SIZE = 160        # Optimized for 12GB VRAM
NUM_WORKERS = 4         # Data loading workers
NUM_ATTRS = 40          # 40 facial attributes
DROPOUT = 0.4
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3
WARMUP_EPOCHS = 3
PREDICTION_THRESHOLD = 0.4
```

## Script Structure
```
celeba_full_gpu_training.py
│
├── IMPORTS
│   ├── torch, torch.nn, torch.optim
│   ├── numpy, pandas, PIL, matplotlib, tqdm
│   └── warnings (ignored)
│
├── CONFIGURATION SECTION
│   ├── Data paths (IMAGE_DIR, ATTR_PATH, PARTITION_PATH)
│   ├── Model params (IMAGE_SIZE, BATCH_SIZE, NUM_ATTRS, DROPOUT)
│   └── Training params (EPOCHS, LR, WARMUP_EPOCHS, etc.)
│
├── setup()
│   ├── Create output/checkpoint directories
│   ├── Set random seeds (SEED=42)
│   ├── Enable TF32 for Ampere GPUs
│   └── Return device (cuda/cpu)
│
├── load_dataframes()
│   ├── Load list_attr_celeba.txt
│   ├── Load list_eval_partition.txt
│   ├── Convert labels {-1,+1} → {0,1}
│   └── Return train_df, val_df, test_df
│
├── compute_class_weights()
│   ├── Calculate pos_weight = neg/pos
│   ├── Clamp max=10.0
│   └── Return pos_weight_tensor, attr_cols
│
├── to_tensor_normalized()
├── augment_image()        # Flip, brightness, contrast, rotation
├── train_transform()
├── eval_transform()
│
├── class CelebADataset(Dataset)
│   ├── __init__()
│   ├── __len__()
│   └── __getitem__()      # Load image, apply transform, return (tensor, label)
│
├── get_dataloaders()
│   └── Return train_loader, val_loader, test_loader
│
├── conv_block()
│
├── class SimpleCNN(nn.Module)
│   ├── __init__()         # 4 conv blocks: 3→64→128→256→256
│   ├── forward()          # Returns 40 logits
│   └── _init_weights()    # Kaiming for Conv, Xavier for Linear
│
├── compute_metrics()
│   ├── Inputs: predictions, labels, threshold
│   └── Returns: accuracy, precision, recall, f1, tp, fp, fn
│
├── train_model()          ← MAIN TRAINING LOOP
│   ├── For each epoch:
│   │   ├── LR warmup (first 3 epochs)
│   │   ├── Training phase (with AMP)
│   │   │   ├── torch.amp.autocast(device_type='cuda', ...)
│   │   │   ├── scaler.scale(loss).backward()
│   │   │   └── gradient clipping (max_norm=1.0)
│   │   ├── Validation phase
│   │   ├── Compute metrics
│   │   ├── Early stopping check
│   │   └── Save best model
│   └── Return: train_losses, val_losses, metrics_history, learning_rates, best_epoch
│
├── plot_training_curves()
│   └── 4-panel plot: Loss, Accuracy, F1, LR
│
├── tune_threshold()
│   ├── Load best model
│   ├── Collect validation predictions
│   ├── Test thresholds [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
│   └── Return best threshold (max F1)
│
├── evaluate_test()
│   ├── Load best model
│   ├── Run on test_loader
│   ├── Compute metrics with best_threshold
│   └── Save predictions to .npy files
│
└── main()
    ├── device = setup()
    ├── train_df, val_df, test_df = load_dataframes()
    ├── pos_weight_tensor, attr_cols = compute_class_weights()
    ├── train_loader, val_loader, test_loader = get_dataloaders()
    ├── model = SimpleCNN().to(device)
    ├── criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    ├── optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    ├── scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    ├── scaler = torch.amp.GradScaler('cuda')  # Modern AMP API
    ├── train_model(...)
    ├── plot_training_curves(...)
    ├── best_threshold = tune_threshold(...)
    └── evaluate_test(...)
```

## Key Functions Reference

### setup()
Initialize training environment.
**Returns:** `device` (torch.device)

### load_dataframes()
Load CelebA dataset splits.
**Returns:** `(train_df, val_df, test_df)` - pandas DataFrames

### compute_class_weights(train_df, device)
Calculate class imbalance weights.
**Returns:** `(pos_weight_tensor, attr_cols)`

### get_dataloaders(train_df, val_df, test_df)
Create PyTorch DataLoaders.
**Returns:** `(train_loader, val_loader, test_loader)`

### train_model(...)
Main training loop with AMP, warmup, early stopping.
**Returns:** `(train_losses, val_losses, val_metrics_history, learning_rates, best_epoch)`

### tune_threshold(model, val_loader, scaler, device)
Find optimal prediction threshold on validation set.
**Returns:** `best_threshold` (float)

### evaluate_test(model, test_loader, criterion, scaler, device, best_threshold)
Final evaluation on test set.
**Saves:** `outputs/test_predictions_gpu.npy`, `outputs/test_labels_gpu.npy`

## Modern PyTorch 2.x Features Used
1. **torch.amp.autocast** (not torch.cuda.amp)
2. **torch.amp.GradScaler('cuda')** (device-specific)
3. **AdamW** optimizer (instead of Adam)
4. **TF32** enabled for matrix multiplication
5. **non_blocking=True** for GPU transfers

## Training Flow
```
1. Load Data → 2. Compute Weights → 3. Create Loaders
     ↓
4. Build Model → 5. Setup Loss/Optimizer/Scheduler/Scaler
     ↓
6. Train Loop (20 epochs max)
   - Warmup (3 epochs)
   - AMP forward/backward
   - Validation
   - Early stopping check
     ↓
7. Plot Curves → 8. Tune Threshold → 9. Test Evaluation
```

## Outputs Generated
| File | Description |
|------|-------------|
| `checkpoints/best_model_gpu.pth` | Best validation loss model |
| `checkpoints/last_model_gpu.pth` | Final epoch model |
| `outputs/training_curves_gpu.png` | 4-panel training visualization |
| `outputs/test_predictions_gpu.npy` | Test set predictions array |
| `outputs/test_labels_gpu.npy` | Test set ground truth array |

## How to Run
```bash
cd c:\MLA\CelebA_Local_GPU
python celeba_full_gpu_training.py
```

## Expected Runtime
- ~3-4 hours on RTX 4070 Ti (12GB)
- ~8-12 minutes per epoch
- Early stopping may finish in ~2.5-3 hours

## Model Architecture Details
```
SimpleCNN(
  (block1): Sequential Conv(3→64) + BN + ReLU + MaxPool
  (block2): Sequential Conv(64→128) + BN + ReLU + MaxPool
  (block3): Sequential Conv(128→256) + BN + ReLU + MaxPool
  (block4): Sequential Conv(256→256) + BN + ReLU + MaxPool
  (pool): AdaptiveAvgPool2d((1,1))
  (dropout): Dropout(p=0.4)
  (fc): Linear(256→40)
)
Total params: ~400,000
```

## Next Steps After Training
1. **Evaluate models:** `python evaluate_model.py --model both`
2. **Predict on photos:** `python predict_single_image.py "photo.jpg"`
