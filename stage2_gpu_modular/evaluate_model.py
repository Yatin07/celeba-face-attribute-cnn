# =============================================================================
# CelebA Model Evaluation Script
# Evaluate trained model on test set with detailed metrics
# =============================================================================

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIGURATION - UPDATE THESE TO MATCH YOUR TRAINING CONFIG
# =============================================================================

# DATA PATHS - MUST MATCH TRAINING PATHS
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = r'C:\MLA\img_align_celeba'
ATTR_PATH = str(BASE_DIR / 'data' / 'annotations' / 'list_attr_celeba.txt')
PARTITION_PATH = str(BASE_DIR / 'data' / 'annotations' / 'list_eval_partition.txt')

# MODEL PARAMETERS - MUST MATCH TRAINING CONFIG
IMAGE_SIZE = 160
BATCH_SIZE = 160
NUM_WORKERS = 4  # Can use workers for evaluation (no training)
PIN_MEMORY = True
NUM_ATTRS = 40
DROPOUT = 0.4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataframes():
    """Load CelebA dataset."""
    print('\n[DATA] Loading dataset...')
    
    attr = pd.read_csv(ATTR_PATH, sep=r'\s+', header=1, index_col=0)
    attr.index.name = 'image_id'
    attr = ((attr + 1) // 2).reset_index()
    
    splits = pd.read_csv(PARTITION_PATH, sep=' ', header=None,
                         names=['image_id', 'split'])
    
    df = splits.merge(attr, on='image_id')
    
    # Get test split only (split=2)
    test_df = df[df['split'] == 2].drop('split', axis=1).reset_index(drop=True)
    
    # Get attribute names
    attr_cols = [c for c in test_df.columns if c != 'image_id']
    
    print(f'[DATA] Test samples: {len(test_df):,}')
    print(f'[DATA] Attributes: {len(attr_cols)}')
    
    return test_df, attr_cols


# =============================================================================
# DATASET
# =============================================================================

def to_tensor_normalized(img):
    """Convert PIL image to normalized tensor [-1, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def eval_transform(img):
    """Evaluation transform (no augmentation)."""
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return to_tensor_normalized(img)


class CelebADataset(Dataset):
    """CelebA Dataset for evaluation."""
    
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.attr_cols = [c for c in df.columns if c != 'image_id']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_id = row['image_id']
            img_path = os.path.join(self.img_dir, img_id)
            img = Image.open(img_path).convert('RGB')
            
            img_tensor = eval_transform(img)
            label = torch.tensor(row[self.attr_cols].values.astype(float), dtype=torch.float32)
            
            return img_tensor, label, img_id
        except Exception as e:
            print(f'Warning: Error loading {img_id}: {e}')
            return (torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), 
                    torch.zeros(len(self.attr_cols)), 
                    'error.jpg')
    
    def get_attr_names(self):
        return self.attr_cols


def get_test_loader(test_df):
    """Create test DataLoader."""
    test_ds = CelebADataset(test_df, IMAGE_DIR)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    return test_loader, test_ds.get_attr_names()


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
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = conv_block(128, 256)
        self.block4 = conv_block(256, 256)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)
    
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
    tn = ((1 - preds_binary) * (1 - labels)).sum(dim=0)
    
    # Per-class accuracy
    per_class_acc = ((tp + tn) / (tp + fp + fn + tn + 1e-8)) * 100
    
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
        'fn': fn,
        'tn': tn,
        'per_class_acc': per_class_acc
    }


def find_best_threshold(predictions, labels):
    """Find optimal threshold for F1 score."""
    print('\n[THRESHOLD] Finding optimal threshold...')
    
    best_f1 = 0
    best_threshold = 0.4
    
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        metrics = compute_metrics(predictions, labels, threshold=t)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = t
        print(f'  Threshold {t:.2f}: F1={metrics["f1"]:.2f}%')
    
    print(f'\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1:.2f}%)')
    return best_threshold


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(checkpoint_path, test_df, attr_names, find_threshold=True):
    """Evaluate a trained model."""
    print('\n' + '='*70)
    print(f'EVALUATING: {os.path.basename(checkpoint_path)}')
    print('='*70)
    
    # Load model
    model = SimpleCNN(num_classes=NUM_ATTRS, dropout=DROPOUT).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f'[ERROR] Checkpoint not found: {checkpoint_path}')
        return None
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f'[INFO] Loaded checkpoint: {checkpoint_path}')
    
    # Create DataLoader
    test_loader, _ = get_test_loader(test_df)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_image_ids = []
    
    print('\n[EVAL] Running inference on test set...')
    
    with torch.no_grad():
        for images, labels, img_ids in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels)
            all_image_ids.extend(img_ids)
    
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    
    # Find best threshold if requested
    if find_threshold:
        best_threshold = find_best_threshold(all_preds_tensor, all_labels_tensor)
    else:
        best_threshold = 0.4
    
    # Compute final metrics
    metrics = compute_metrics(all_preds_tensor, all_labels_tensor, threshold=best_threshold)
    
    # Print results
    print('\n' + '='*70)
    print(f'TEST SET RESULTS (threshold={best_threshold:.2f}):')
    print('='*70)
    print(f'  Accuracy:    {metrics["accuracy"]:.2f}%')
    print(f'  F1 Score:    {metrics["f1"]:.2f}%')
    print(f'  Precision:   {metrics["precision"]:.2f}%')
    print(f'  Recall:      {metrics["recall"]:.2f}%')
    
    # Per-class metrics
    print('\n[PER-CLASS] Top 10 Most Accurate Attributes:')
    per_class_acc = metrics['per_class_acc']
    sorted_indices = torch.argsort(per_class_acc, descending=True)
    for i in range(min(10, len(attr_names))):
        idx = sorted_indices[i].item()
        print(f'  {i+1:2d}. {attr_names[idx]:25s}: {per_class_acc[idx]:5.2f}%')
    
    print('\n[PER-CLASS] Bottom 5 Least Accurate Attributes:')
    for i in range(min(5, len(attr_names))):
        idx = sorted_indices[-(i+1)].item()
        print(f'  {i+1:2d}. {attr_names[idx]:25s}: {per_class_acc[idx]:5.2f}%')
    
    # Save predictions
    output_name = os.path.basename(checkpoint_path).replace('.pth', '_predictions.npy')
    np.save(f'outputs/{output_name}', torch.sigmoid(all_preds_tensor).numpy())
    print(f'\n[SAVED] Predictions saved to outputs/{output_name}')
    
    return metrics


def compare_models(test_df, attr_names):
    """Compare best and last models."""
    print('\n' + '='*70)
    print('COMPARING BEST vs LAST MODEL')
    print('='*70)
    
    best_path = 'checkpoints/best_model_stage2.pth'
    last_path = 'checkpoints/last_model_gpu.pth'
    
    results = {}
    
    # Evaluate best model
    if os.path.exists(best_path):
        print(f'\n📊 Evaluating BEST model...')
        results['best'] = evaluate_model(best_path, test_df, attr_names, find_threshold=True)
    else:
        print(f'[WARNING] Best model not found: {best_path}')
    
    # Evaluate last model
    if os.path.exists(last_path):
        print(f'\n📊 Evaluating LAST model...')
        results['last'] = evaluate_model(last_path, test_df, attr_names, find_threshold=True)
    else:
        print(f'[WARNING] Last model not found: {last_path}')
    
    # Comparison
    if 'best' in results and 'last' in results:
        print('\n' + '='*70)
        print('COMPARISON SUMMARY')
        print('='*70)
        print(f'{"Metric":<15} {"Best Model":>12} {"Last Model":>12} {"Winner":>10}')
        print('-'*70)
        
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            best_val = results['best'][metric]
            last_val = results['last'][metric]
            winner = 'BEST' if best_val > last_val else 'LAST'
            print(f'{metric.capitalize():<15} {best_val:>11.2f}% {last_val:>11.2f}% {winner:>10}')
        
        # Recommendation
        best_f1 = results['best']['f1']
        last_f1 = results['last']['f1']
        print('\n' + '='*70)
        if best_f1 >= last_f1:
            print('✅ RECOMMENDATION: Use BEST model (best_model_gpu.pth)')
            print(f'   Reason: Higher F1 score ({best_f1:.2f}% vs {last_f1:.2f}%)')
        else:
            print('✅ RECOMMENDATION: Use LAST model (last_model_gpu.pth)')
            print(f'   Reason: Higher F1 score ({last_f1:.2f}% vs {best_f1:.2f}%)')
        print('='*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate CelebA trained model')
    parser.add_argument('--model', type=str, choices=['best', 'last', 'both'], 
                        default='both',
                        help='Which model to evaluate: best, last, or both (default: both)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to custom checkpoint file (overrides --model)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Use fixed threshold (default: auto-tune)')
    
    args = parser.parse_args()
    
    # Load data
    test_df, attr_names = load_dataframes()
    
    # Evaluation
    if args.checkpoint:
        # Custom checkpoint
        evaluate_model(args.checkpoint, test_df, attr_names, find_threshold=(args.threshold is None))
    elif args.model == 'both':
        # Compare both models
        compare_models(test_df, attr_names)
    elif args.model == 'best':
        # Best only
        evaluate_model('checkpoints/best_model_stage2.pth', test_df, attr_names, 
                      find_threshold=(args.threshold is None))
    elif args.model == 'last':
        # Last only
        evaluate_model('checkpoints/last_model_gpu.pth', test_df, attr_names,
                      find_threshold=(args.threshold is None))


if __name__ == '__main__':
    main()
