# =============================================================================
# CelebA Single Image Prediction Script
# Predict 40 facial attributes on your own photo
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# =============================================================================
# CONFIGURATION - MUST MATCH TRAINING
# =============================================================================

IMAGE_SIZE = 160
NUM_ATTRS = 40
DROPOUT = 0.4

# CelebA 40 attribute names (in order)
ATTRIBUTE_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]

# Attribute descriptions (for better understanding)
ATTRIBUTE_DESCRIPTIONS = {
    '5_o_Clock_Shadow': 'Has stubble/beard shadow',
    'Arched_Eyebrows': 'Eyebrows have an arch',
    'Attractive': 'Facially attractive',
    'Bags_Under_Eyes': 'Dark circles under eyes',
    'Bald': 'Little or no hair',
    'Bangs': 'Hair covers forehead',
    'Big_Lips': 'Full/large lips',
    'Big_Nose': 'Large nose',
    'Black_Hair': 'Black hair color',
    'Blond_Hair': 'Blonde hair color',
    'Blurry': 'Image is blurry',
    'Brown_Hair': 'Brown hair color',
    'Bushy_Eyebrows': 'Thick/full eyebrows',
    'Chubby': 'Round/full face',
    'Double_Chin': 'Visible double chin',
    'Eyeglasses': 'Wearing glasses',
    'Goatee': 'Goatee beard style',
    'Gray_Hair': 'Gray/white hair',
    'Heavy_Makeup': 'Heavy cosmetics',
    'High_Cheekbones': 'Prominent cheekbones',
    'Male': 'Male gender',
    'Mouth_Slightly_Open': 'Mouth not fully closed',
    'Mustache': 'Has mustache',
    'Narrow_Eyes': 'Small/narrow eye opening',
    'No_Beard': 'Clean shaven',
    'Oval_Face': 'Oval face shape',
    'Pale_Skin': 'Light skin tone',
    'Pointy_Nose': 'Pointed nose tip',
    'Receding_Hairline': 'Hairline going back',
    'Rosy_Cheeks': 'Red/pink cheeks',
    'Sideburns': 'Has sideburns',
    'Smiling': 'Smiling expression',
    'Straight_Hair': 'Straight hair texture',
    'Wavy_Hair': 'Wavy/curly hair',
    'Wearing_Earrings': 'Has earrings',
    'Wearing_Hat': 'Wearing headwear',
    'Wearing_Lipstick': 'Has lipstick',
    'Wearing_Necklace': 'Has necklace',
    'Wearing_Necktie': 'Wearing tie',
    'Young': 'Young appearance'
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')


# =============================================================================
# MODEL
# =============================================================================

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SimpleCNN(nn.Module):
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
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path):
    """Load and preprocess image for model."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Store original for display
    original = img.copy()
    
    # Resize
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Convert to numpy and normalize to [-1, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor, original


# =============================================================================
# PREDICTION
# =============================================================================

def predict_image(model, image_path, threshold=0.4):
    """Predict attributes on a single image."""
    print(f'\n{"="*70}')
    print(f'PREDICTING: {image_path}')
    print(f'{"="*70}')
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f'[ERROR] Image not found: {image_path}')
        return None
    
    # Preprocess
    print('[INFO] Loading and preprocessing image...')
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Predict
    print('[INFO] Running inference...')
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(img_tensor)
    
    # Get probabilities
    probs = torch.sigmoid(outputs).float().cpu().numpy()[0]
    
    # Create results
    results = []
    for i, attr in enumerate(ATTRIBUTE_NAMES):
        pred = 1 if probs[i] > threshold else 0
        results.append({
            'attribute': attr,
            'description': ATTRIBUTE_DESCRIPTIONS.get(attr, ''),
            'probability': probs[i],
            'prediction': pred
        })
    
    # Sort by probability
    results_sorted = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    # Print all predictions
    print(f'\n📊 ALL 40 ATTRIBUTES (threshold={threshold}):')
    print(f'{"Attribute":<25} {"Prob":>8} {"Pred":>6} {"Description"}')
    print('-'*70)
    
    for r in results_sorted:
        mark = '✓' if r['prediction'] == 1 else ' '
        print(f"{r['attribute']:<25} {r['probability']*100:>7.1f}%  [{mark}]  {r['description']}")
    
    # Print top 10 predictions
    print(f'\n TOP 10 HIGHEST CONFIDENCE PREDICTIONS:')
    print(f'{"#":<4} {"Attribute":<25} {"Confidence":>12} {"Present?"}')
    print('-'*70)
    
    for i, r in enumerate(results_sorted[:10], 1):
        status = 'YES ✓' if r['prediction'] == 1 else 'NO'
        bar = '█' * int(r['probability'] * 20)
        print(f"{i:<4} {r['attribute']:<25} {r['probability']*100:>11.1f}% [{bar:<20}] {status}")
    
    # Summary
    detected = sum([r['prediction'] for r in results])
    print(f'\n📋 SUMMARY:')
    print(f'   Total attributes detected: {detected}/40')
    print(f'   Prediction threshold: {threshold}')
    print(f'   Model confidence: {np.mean(probs)*100:.1f}% average')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Top attributes bar chart
    top_attrs = [r['attribute'] for r in results_sorted[:15]]
    top_probs = [r['probability']*100 for r in results_sorted[:15]]
    colors = ['green' if r['prediction'] == 1 else 'gray' for r in results_sorted[:15]]
    
    axes[1].barh(range(len(top_attrs)), top_probs, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(top_attrs)))
    axes[1].set_yticklabels(top_attrs, fontsize=9)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title(f'Top 15 Predictions (threshold={threshold})')
    axes[1].set_xlim(0, 100)
    axes[1].invert_yaxis()
    axes[1].axvline(x=threshold*100, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_name = os.path.splitext(os.path.basename(image_path))[0]
    plot_path = f'outputs/prediction_{output_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'\n[SAVED] Prediction visualization: {plot_path}')
    
    plt.show()
    
    return results


def load_model():
    """Load trained model from checkpoint."""
    checkpoint_path = 'checkpoints/best_model_gpu.pth'
    print(f'[INFO] Loading model from: {checkpoint_path}')
    
    if not os.path.exists(checkpoint_path):
        print(f'[ERROR] Checkpoint not found: {checkpoint_path}')
        return None
    
    model = SimpleCNN(num_classes=NUM_ATTRS, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print('[SUCCESS] Model loaded successfully!')
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print('Usage: python predict_single_image.py <image_path>')
        print('Example: python predict_single_image.py download2.jpg')
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load model (always uses best_model_gpu.pth)
    model = load_model()
    if model is None:
        return
    
    # Predict (fixed threshold 0.4)
    results = predict_image(model, image_path, threshold=0.4)
    
    if results:
        print(f'\n{"="*70}')
        print('PREDICTION COMPLETE!')
        print(f'{"="*70}')


if __name__ == '__main__':
    main()
