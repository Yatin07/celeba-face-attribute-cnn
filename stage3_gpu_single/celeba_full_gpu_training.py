# =============================================================================
# CelebA Multi-label Classification - Full GPU Training Script
# Trains on FULL 162k dataset with all optimizations
# =============================================================================

import os
import time
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# =============================================================================

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

# DATA PATHS - UPDATE THESE TO YOUR ACTUAL PATHS
# ONLY CHANGE THIS PATH IF YOU MOVE YOUR IMAGES FOLDER
IMAGE_DIR = r'C:\MLA\img_align_celeba'

# These auto-resolve relative to the project root
ATTR_PATH = str(BASE_DIR / 'data' / 'annotations' / 'list_attr_celeba.txt')
PARTITION_PATH = str(BASE_DIR / 'data' / 'annotations' / 'list_eval_partition.txt')

# MODEL PARAMETERS - FULL DATASET OPTIMIZED
IMAGE_SIZE = 160        # 160x160 resolution
BATCH_SIZE = 160        # Optimized for 12GB VRAM
NUM_WORKERS = 4         # Data loading workers
PIN_MEMORY = True

NUM_ATTRS = 40
DROPOUT = 0.4

# TRAINING PARAMETERS
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3
PREDICTION_THRESHOLD = 0.4

# WARMUP CONFIGURATION
WARMUP_EPOCHS = 3
INITIAL_LR = LEARNING_RATE * 0.3

# RANDOM SEED
SEED = 42

# =============================================================================
# SETUP
# =============================================================================

def setup():
    """Initialize directories and random seeds."""
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster training on Ampere+ GPUs (RTX 30xx/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')
    if device.type == 'cuda':
        print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
        print(f'[INFO] CUDA Version: {torch.version.cuda}')
        print(f'[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    print(f'[INFO] Training Mode: FULL DATASET (162k images)')
    print(f'[INFO] Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}')
    print(f'[INFO] Batch size: {BATCH_SIZE}')
    print(f'[INFO] AMP Enabled: {device.type == "cuda"}')
    
    return device

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataframes():
    """Load FULL CelebA dataset (162k images)."""
    print('\n[DATA] Loading attribute file...')
    
    # Load attributes (skip count row, second row is headers)
    attr = pd.read_csv(ATTR_PATH, sep=r'\s+', header=1, index_col=0)
    attr.index.name = 'image_id'
    
    # Convert labels: -1 -> 0, +1 -> 1
    attr = ((attr + 1) // 2).reset_index()
    print(f'[DATA] Attributes shape: {attr.shape}')
    
    # Load partition file
    print('[DATA] Loading partition file...')
    splits = pd.read_csv(PARTITION_PATH, sep=' ', header=None,
                         names=['image_id', 'split'])
    
    # Merge
    df = splits.merge(attr, on='image_id')
    print(f'[DATA] Total samples: {len(df):,}')
    
    # Split (0=train, 1=val, 2=test) - FULL DATASET
    train_df = df[df['split'] == 0].drop('split', axis=1).reset_index(drop=True)
    val_df = df[df['split'] == 1].drop('split', axis=1).reset_index(drop=True)
    test_df = df[df['split'] == 2].drop('split', axis=1).reset_index(drop=True)
    
    print(f'[DATA] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
    print(f'[DATA] Using FULL dataset - no subset mode')
    return train_df, val_df, test_df


def compute_class_weights(train_df, device):
    """Compute class weights for rare attributes."""
    print('\n[WEIGHTS] Computing pos_weight for class imbalance...')
    attr_cols = [c for c in train_df.columns if c != 'image_id']
    y_train = train_df[attr_cols].values
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    
    # Calculate pos_weight = neg/pos, clamp to prevent extreme values
    pos_weight_tensor = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, max=10.0)
    pos_weight_tensor = pos_weight_tensor.to(device)
    
    print(f'[WEIGHTS] Computed for {len(pos_weight_tensor)} attributes')
    print(f'[WEIGHTS] Max: {pos_weight_tensor.max():.2f}, Min: {pos_weight_tensor.min():.2f}')
    print(f'[WEIGHTS] Mean: {pos_weight_tensor.mean():.2f}')
    
    return pos_weight_tensor, attr_cols

# =============================================================================
# DATASET AND AUGMENTATION
# =============================================================================

def to_tensor_normalized(img):
    """Convert PIL image to normalized tensor [-1, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def augment_image(img):
    """Apply data augmentation for better generalization."""
    # Horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    
    # Brightness adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Contrast adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Small rotation
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-10, 10))
    
    return img


def train_transform(img):
    """Training transform with augmentation."""
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = augment_image(img)
    return to_tensor_normalized(img)


def eval_transform(img):
    """Evaluation transform (deterministic)."""
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return to_tensor_normalized(img)


class CelebADataset(Dataset):
    """CelebA Dataset with error handling."""
    
    def __init__(self, df, img_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.attr_cols = [c for c in df.columns if c != 'image_id']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_id = row['image_id']
            img_path = os.path.join(self.img_dir, img_id)
            img = Image.open(img_path).convert('RGB')
            
            img_tensor = train_transform(img) if self.is_train else eval_transform(img)
            label = torch.tensor(row[self.attr_cols].values.astype(float), dtype=torch.float32)
            
            return img_tensor, label
        except Exception as e:
            print(f'Warning: Error loading image at idx {idx}: {e}')
            # Return placeholder to keep training going
            return (torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), 
                    torch.zeros(len(self.attr_cols)))
    
    def get_attr_names(self):
        return self.attr_cols


def get_dataloaders(train_df, val_df, test_df):
    """Create optimized DataLoaders."""
    train_ds = CelebADataset(train_df, IMAGE_DIR, is_train=True)
    val_ds = CelebADataset(val_df, IMAGE_DIR, is_train=False)
    test_ds = CelebADataset(test_df, IMAGE_DIR, is_train=False)
    
    # GPU-optimized loader configuration
    loader_kwargs = {
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
    }
    
    if NUM_WORKERS > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================

def conv_block(in_channels, out_channels):
    """Convolutional block with BatchNorm."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SimpleCNN(nn.Module):
    """SimpleCNN for multi-label classification."""
    
    def __init__(self, num_classes=40, dropout=0.4):
        super().__init__()
        # 160x160 -> 80x80 -> 40x40 -> 20x20 -> 10x10
        self.block1 = conv_block(3, 64)      # 160 -> 80
        self.block2 = conv_block(64, 128)    # 80 -> 40
        self.block3 = conv_block(128, 256)   # 40 -> 20
        self.block4 = conv_block(256, 256)   # 20 -> 10
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)
        
        self._init_weights()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(predictions, labels, threshold=0.4):
    """Compute comprehensive metrics."""
    preds_binary = (torch.sigmoid(predictions) > threshold).float()
    
    # Per-class metrics
    tp = (preds_binary * labels).sum(dim=0)
    fp = (preds_binary * (1 - labels)).sum(dim=0)
    fn = ((1 - preds_binary) * labels).sum(dim=0)
    
    # Overall metrics
    accuracy = (preds_binary == labels).float().mean().item() * 100
    precision = (tp / (tp + fp + 1e-8)).mean().item() * 100
    recall = (tp / (tp + fn + 1e-8)).mean().item() * 100
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                scaler, device, epochs, warmup_epochs, initial_lr, learning_rate,
                early_stopping_patience, prediction_threshold):
    """Train the model with comprehensive monitoring."""
    
    train_losses = []
    val_losses = []
    val_metrics_history = []
    learning_rates = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print('\n[TRAIN] Starting training...')
    print('='*70)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = initial_lr + (learning_rate - initial_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Training phase with AMP
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Non-blocking transfers for GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Compute metrics
        all_preds_tensor = torch.cat(all_preds)
        all_labels_tensor = torch.cat(all_labels)
        metrics = compute_metrics(all_preds_tensor, all_labels_tensor, threshold=prediction_threshold)
        val_metrics_history.append(metrics)
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Print comprehensive summary
        print(f'\nEpoch {epoch+1:02d}/{epochs} | Time: {epoch_time:.1f}s')
        print(f'  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Acc: {metrics["accuracy"]:.2f}% | F1: {metrics["f1"]:.2f}% | Recall: {metrics["recall"]:.2f}%')
        print(f'  LR: {current_lr:.6f}')
        
        # Overfitting warning
        if avg_val_loss > avg_train_loss * 1.2:
            print('  ⚠ Warning: Possible overfitting detected')
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model_gpu.pth')
            print(f'  ✓ Best model saved (epoch {best_epoch})')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'  ⏹ Early stopping triggered!')
                break
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/last_model_gpu.pth')
    print(f'\n[SAVED] Last model saved')
    print(f'[BEST] Best epoch: {best_epoch} with val_loss: {best_val_loss:.4f}')
    print('='*70)
    
    return train_losses, val_losses, val_metrics_history, learning_rates, best_epoch

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(train_losses, val_losses, val_metrics_history, learning_rates):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, 'o-', label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, 's-', label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    acc_values = [m['accuracy'] for m in val_metrics_history]
    axes[0, 1].plot(acc_values, 'o-', label='Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    f1_values = [m['f1'] for m in val_metrics_history]
    axes[1, 0].plot(f1_values, 'o-', label='F1 Score', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score (%)')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(learning_rates, 'o-', label='Learning Rate', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('outputs/training_curves_gpu.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print('[SAVED] Training curves saved to outputs/training_curves_gpu.png')

# =============================================================================
# THRESHOLD TUNING
# =============================================================================

def tune_threshold(model, val_loader, scaler, device):
    """Tune prediction threshold on validation set."""
    print('\n[THRESHOLD] Tuning thresholds on validation set...')
    print('='*70)
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints/best_model_gpu.pth'))
    model.eval()
    
    # Collect validation predictions
    all_val_probs = []
    all_val_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Collecting val preds'):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            all_val_probs.append(probs)
            all_val_labels.append(labels)
    
    all_val_probs = torch.cat(all_val_probs)
    all_val_labels = torch.cat(all_val_labels)
    
    # Test different thresholds
    thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_results = {}
    
    for t in thresholds_to_test:
        metrics = compute_metrics(all_val_probs, all_val_labels, threshold=t)
        threshold_results[t] = metrics
        print(f'Threshold {t:.2f}: F1={metrics["f1"]:.2f}% | Recall={metrics["recall"]:.2f}% | Prec={metrics["precision"]:.2f}%')
    
    # Find best threshold
    best_threshold = max(threshold_results, key=lambda x: threshold_results[x]['f1'])
    best_metrics = threshold_results[best_threshold]
    
    print(f'\n✅ OPTIMAL THRESHOLD: {best_threshold:.2f}')
    print(f'   F1: {best_metrics["f1"]:.2f}% | Recall: {best_metrics["recall"]:.2f}% | Precision: {best_metrics["precision"]:.2f}%')
    print('='*70)
    
    return best_threshold

# =============================================================================
# FINAL EVALUATION
# =============================================================================

def evaluate_test(model, test_loader, criterion, scaler, device, best_threshold):
    """Evaluate on test set."""
    print('\n[TEST] Final evaluation on test set...')
    print('='*70)
    
    # Use best model with optimal threshold
    model.load_state_dict(torch.load('checkpoints/best_model_gpu.pth'))
    model.eval()
    
    test_loss = 0.0
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            all_test_preds.append(outputs.cpu())
            all_test_labels.append(labels.cpu())
    
    avg_test_loss = test_loss / len(test_loader)
    all_test_preds_tensor = torch.cat(all_test_preds)
    all_test_labels_tensor = torch.cat(all_test_labels)
    
    # Compute metrics with optimal threshold
    test_metrics = compute_metrics(all_test_preds_tensor, all_test_labels_tensor, threshold=best_threshold)
    
    print(f'\nTEST SET RESULTS (threshold={best_threshold:.2f}):')
    print(f'  Test Loss:   {avg_test_loss:.4f}')
    print(f'  Accuracy:    {test_metrics["accuracy"]:.2f}%')
    print(f'  F1 Score:    {test_metrics["f1"]:.2f}%')
    print(f'  Recall:      {test_metrics["recall"]:.2f}%')
    print(f'  Precision:   {test_metrics["precision"]:.2f}%')
    
    # Save predictions
    np.save('outputs/test_predictions_gpu.npy', torch.sigmoid(all_test_preds_tensor).numpy())
    np.save('outputs/test_labels_gpu.npy', all_test_labels_tensor.numpy())
    print(f'\n[SAVED] Predictions saved to outputs/')
    
    print('\n' + '='*70)
    print('TRAINING COMPLETE!')
    print('='*70)
    print(f'Best model: checkpoints/best_model_gpu.pth')
    print(f'Last model: checkpoints/last_model_gpu.pth')
    print(f'Training curves: outputs/training_curves_gpu.png')
    print(f'Test predictions: outputs/test_predictions_gpu.npy')
    print(f'Optimal threshold: {best_threshold:.2f}')

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    # Setup
    device = setup()
    
    # Load data
    train_df, val_df, test_df = load_dataframes()
    pos_weight_tensor, attr_cols = compute_class_weights(train_df, device)
    
    # Create dataloaders
    print('\n[DATA] Creating DataLoaders...')
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)
    print(f'[DATA] Train batches: {len(train_loader)} ({len(train_loader)*BATCH_SIZE:,} samples)')
    print(f'[DATA] Val batches: {len(val_loader)}')
    print(f'[DATA] Test batches: {len(test_loader)}')
    
    # Create model
    print('\n[MODEL] Creating model...')
    model = SimpleCNN(num_classes=NUM_ATTRS, dropout=DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[MODEL] Total parameters: {total_params:,}')
    print(f'[MODEL] Model size: {total_params * 4 / 1024**2:.2f} MB')
    print(f'[MODEL] Channels: 3→64→128→256→256')
    print(f'[MODEL] Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}')
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print('[SUCCESS] Training setup complete!')
    print(f'[TRAIN] Epochs: {EPOCHS}, LR: {LEARNING_RATE}')
    print(f'[TRAIN] AMP Enabled: {scaler is not None}')
    
    # Train
    train_losses, val_losses, val_metrics_history, learning_rates, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, EPOCHS, WARMUP_EPOCHS, INITIAL_LR, LEARNING_RATE,
        EARLY_STOPPING_PATIENCE, PREDICTION_THRESHOLD
    )
    
    # Plot curves
    plot_training_curves(train_losses, val_losses, val_metrics_history, learning_rates)
    
    # Tune threshold
    best_threshold = tune_threshold(model, val_loader, scaler, device)
    
    # Final evaluation
    evaluate_test(model, test_loader, criterion, scaler, device, best_threshold)


if __name__ == '__main__':
    main()
