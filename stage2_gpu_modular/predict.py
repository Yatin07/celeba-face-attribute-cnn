# ============================================================
# predict.py — Predict CelebA attributes for a single image
#
# Usage:
#   python predict.py --image path\to\face.jpg
#   python predict.py --image path\to\face.jpg --topk 10
#
# Output example:
#   ============================================
#   Predictions for: face.jpg
#   ============================================
#   Smiling              0.92  █████████░  ✓
#   Young                0.88  ████████░░  ✓
#   Attractive           0.81  ████████░░  ✓
#   Eyeglasses           0.08  █░░░░░░░░░
#   ============================================
#
# Note:
#   sigmoid is applied HERE (not in the model) — raw logits
#   are converted to probabilities only during inference.
# ============================================================

import os
import sys
import argparse
import torch
from PIL import Image
import config
from model import SimpleCNN
from dataset import load_dataframes, eval_transform


# ── Load model from best checkpoint ──────────────────────────

def load_model():
    """
    Load best trained model for inference.

    Returns:
        model : SimpleCNN in eval mode
    """
    stage_name = "stage1" if config.TRAIN_MODE == "subset" else "stage2"
    best_path  = os.path.join(config.CHECKPOINT_DIR,
                               f"best_model_{stage_name}.pth")

    if not os.path.exists(best_path):
        print(f"[ERROR] No model found at: {best_path}")
        print("        Run  python train.py  first.")
        sys.exit(1)

    model = SimpleCNN(num_classes=config.NUM_ATTRS,
                      dropout=config.DROPOUT)
    ckpt  = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"[INFO] Model loaded from : {best_path}")
    print(f"[INFO] Trained at epoch  : {ckpt.get('epoch', '?')}")
    return model


# ── Prediction function ───────────────────────────────────────

def predict_image(image_path: str, model, attr_names: list,
                  top_k: int = None):
    """
    Predict all 40 CelebA attributes for one image.

    Args:
        image_path : path to any .jpg / .png face image
        model      : loaded SimpleCNN in eval mode
        attr_names : list of 40 attribute name strings
        top_k      : if set, show only the top-K highest confidence attrs

    Returns:
        results : list of (attribute_name, probability) sorted by prob desc

    How confidence works:
        1. Model outputs raw logits (B, 40)
        2. torch.sigmoid converts each logit to a probability in [0, 1]
        3. probability > 0.5  → attribute is predicted PRESENT  (✓)
        4. probability ≤ 0.5  → attribute is predicted ABSENT
        Example: Smiling: 0.92 means 92% confidence the person is smiling
    """
    # Apply same eval transform as val/test (Resize + Normalize, no augmentation)
    img   = Image.open(image_path).convert('RGB')
    img_t = eval_transform(img).unsqueeze(0)   # (1, 3, 128, 128)

    # ── Forward pass ──────────────────────────────────────────
    with torch.no_grad():
        logits = model(img_t)                        # (1, 40) raw logits
        probs  = torch.sigmoid(logits).squeeze(0)    # (40,)   probabilities

    # ── Format results ─────────────────────────────────────────
    results = sorted(zip(attr_names, probs.tolist()),
                     key=lambda x: -x[1])    # sort by confidence (desc)

    if top_k:
        results = results[:top_k]

    # ── Print table ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Predictions: {os.path.basename(image_path)}")
    print("=" * 50)
    print(f"  {'Attribute':<24} {'Prob':>5}  {'Confidence Bar':15}  Flag")
    print("  " + "-" * 52)

    for name, prob in results:
        filled = int(prob * 10)
        bar    = "█" * filled + "░" * (10 - filled)
        flag   = "✓ PRESENT" if prob > 0.5 else "  absent"
        print(f"  {name:<24} {prob:.2f}  {bar}  {flag}")

    print("=" * 50)
    print(f"\n  ✓ = predicted present (confidence > 50%)")
    print(f"  Showing {'top ' + str(top_k) if top_k else 'all 40'} attributes\n")

    return results


# ── Entry point ───────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict CelebA attributes for a single image.')
    parser.add_argument('--image', required=True,
                        help='Path to input image (jpg/png)')
    parser.add_argument('--topk', type=int, default=None,
                        help='Show only top-K attributes (default: all 40)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    # Load attribute names from training data
    train_df, _, _ = load_dataframes()
    attr_names = [c for c in train_df.columns if c != 'image_id']

    # Load model and predict
    model   = load_model()
    results = predict_image(args.image, model, attr_names,
                            top_k=args.topk)
