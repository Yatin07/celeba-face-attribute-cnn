# =============================================================================
# CelebA Web UI - Single File Gradio App
# Upload image and predict 40 facial attributes
# =============================================================================

import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import gradio as gr

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 160
NUM_ATTRS = 40
DROPOUT = 0.4

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# LOAD MODEL
# =============================================================================

def load_model():
    model = SimpleCNN(num_classes=NUM_ATTRS, dropout=DROPOUT).to(device)
    checkpoint_path = str(BASE_DIR / 'checkpoints' / 'best_model_gpu.pth')
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = str(BASE_DIR / 'checkpoints' / 'last_model_gpu.pth')
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, os.path.basename(checkpoint_path)
    else:
        raise FileNotFoundError(f"No model checkpoint found in {BASE_DIR / 'checkpoints'}")


model, model_name = load_model()

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_image(input_image):
    """Predict attributes on uploaded image."""
    if input_image is None:
        return "Please upload an image", None
    
    # Convert numpy array to PIL if needed
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image.astype('uint8'))
    else:
        img = input_image
    
    # Fixed threshold
    threshold = 0.4
    
    # Preprocess
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        if device.type == 'cuda':
             with torch.amp.autocast(device_type='cuda'):
                 outputs = model(tensor)
        else:
             outputs = model(tensor)
    
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
    
    # Format output
    detected_count = sum([r['prediction'] for r in results])
    avg_confidence = np.mean(probs) * 100
    
    # Build text output
    lines = []
    lines.append(f"📊 PREDICTION RESULTS (Model: {model_name})")
    lines.append(f"Threshold: {threshold} | Device: {device.type}")
    lines.append(f"Detected: {detected_count}/40 attributes | Avg Confidence: {avg_confidence:.1f}%")
    lines.append("")
    lines.append(f"{'Attribute':<25} {'Prob':>8} {'Pred':>6} {'Description'}")
    lines.append("-" * 70)
    
    for r in results_sorted:
        mark = '✓' if r['prediction'] == 1 else ' '
        lines.append(f"{r['attribute']:<25} {r['probability']*100:>7.1f}%  [{mark}]  {r['description']}")
    
    # Create visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_attrs = [r['attribute'] for r in results_sorted[:15]]
    top_probs = [r['probability']*100 for r in results_sorted[:15]]
    colors = ['green' if r['prediction'] == 1 else 'gray' for r in results_sorted[:15]]
    
    bars = ax.barh(range(len(top_attrs)), top_probs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_attrs)))
    ax.set_yticklabels(top_attrs, fontsize=10)
    ax.set_xlabel('Confidence (%)', fontsize=11)
    ax.set_title(f'Top 15 Predictions (threshold={threshold})', fontsize=12)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.axvline(x=threshold*100, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to temp file
    output_dir = BASE_DIR / 'outputs'
    output_dir.mkdir(exist_ok=True)
    temp_path = str(output_dir / 'temp_prediction.png')
    plt.savefig(temp_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return "\n".join(lines), temp_path


# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    with gr.Blocks(title="CelebA Face Attribute Predictor", css="""
        .container { max-width: 1200px; margin: auto; }
        .output-text { font-family: monospace; white-space: pre; }
    """) as demo:
        
        gr.Markdown("""
        # 🎭 CelebA Face Attribute Predictor
        
        Upload a face image to predict 40 facial attributes using deep learning.
        
        **Model:** SimpleCNN (~900K parameters) | **Device:** {} | **Checkpoint:** {}
        """.format(device.type.upper(), model_name))
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    label="Input Face Image",
                    type="numpy",
                    height=300
                )
                
                predict_btn = gr.Button("🔮 Predict Attributes", variant="primary", size="lg")
                
                gr.Markdown("""
                **Instructions:**
                1. Upload a face photo (JPG/PNG)
                2. Click "Predict Attributes"
                3. View results below
                
                **Attributes include:** Gender, age, hair color, glasses, smile, and more.
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Prediction Results")
                
                output_plot = gr.Image(
                    label="Top 15 Predictions Chart",
                    type="filepath",
                    height=400
                )
                
                output_text = gr.Textbox(
                    label="All 40 Attributes (sorted by confidence)",
                    lines=25,
                    max_lines=50,
                    elem_classes="output-text"
                )
        
        # Event handlers
        predict_btn.click(
            fn=predict_image,
            inputs=[input_image],
            outputs=[output_text, output_plot]
        )
        
        # Auto-predict when image uploaded
        input_image.change(
            fn=predict_image,
            inputs=[input_image],
            outputs=[output_text, output_plot]
        )
        
        gr.Markdown("""
        ---
        **Note:** This model was trained on CelebA dataset (celebrity faces). 
        Performance may vary on different demographics or image quality.
        """)
    
    return demo


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f'[INFO] Starting CelebA Web UI')
    print(f'[INFO] Device: {device}')
    print(f'[INFO] Model: {model_name}')
    print(f'[INFO] Loading Gradio interface...')
    
    demo = create_ui()
    print('[INFO] Launching at http://127.0.0.1:7860')
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False
    )
