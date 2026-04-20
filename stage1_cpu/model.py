# ============================================================
# model.py — Simple Plain CNN for CelebA multi-label classification
#
# Architecture (4 Conv Blocks):
#   Block 1: Conv(3→32)   + BN + ReLU + MaxPool → (B, 32, 64, 64)
#   Block 2: Conv(32→64)  + BN + ReLU + MaxPool → (B, 64, 32, 32)
#   Block 3: Conv(64→128) + BN + ReLU + MaxPool → (B,128, 16, 16)
#   Block 4: Conv(128→256)+ BN + ReLU + MaxPool → (B,256,  8,  8)
#   GlobalAvgPool → (B, 256)
#   Dropout(0.4) → FC(256→40)
#
# Input : (B, 3, 128, 128)
# Output: (B, 40) raw logits — sigmoid applied ONLY during inference
#
# Params: ~400K  (fast on CPU, easy to explain)
# ============================================================

import torch
import torch.nn as nn


# ── Helper: one convolutional block ──────────────────────────

def _conv_block(in_channels, out_channels):
    """
    A single convolutional block:
        Conv2D → BatchNorm → ReLU → MaxPool(2×2)

    - No bias in Conv2D because BatchNorm handles the bias role
    - padding=1 keeps spatial size before pooling
    - MaxPool halves spatial dimensions: H×W → H/2 × W/2
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


# ── Main Model ────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    4-block Plain CNN for CelebA 40-attribute multi-label prediction.

    Why plain CNN (not ResNet)?
      - Fewer parameters → faster on CPU
      - Easier to understand and explain
      - Sufficient depth for 128×128 face images
      - No skip connections needed for this depth

    Why GlobalAvgPool instead of Flatten?
      - Reduces parameters dramatically (no large FC layer)
      - Acts as regularizer — less overfitting
      - Preserves spatial meaning of feature maps
    """

    def __init__(self, num_classes=40, dropout=0.4):
        super().__init__()

        # ── Four convolutional blocks ──────────────────────────
        self.block1 = _conv_block(3,   32)   # 128×128 → 64×64
        self.block2 = _conv_block(32,  64)   #  64×64  → 32×32
        self.block3 = _conv_block(64,  128)  #  32×32  → 16×16
        self.block4 = _conv_block(128, 256)  #  16×16  →  8×8

        # ── Classification head ────────────────────────────────
        # AdaptiveAvgPool collapses 8×8 → 1×1 regardless of input size
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, num_classes)

        # ── Weight initialisation ──────────────────────────────
        self._init_weights()

    def forward(self, x):
        # x: (B, 3, 128, 128)
        x = self.block1(x)          # → (B,  32, 64, 64)
        x = self.block2(x)          # → (B,  64, 32, 32)
        x = self.block3(x)          # → (B, 128, 16, 16)
        x = self.block4(x)          # → (B, 256,  8,  8)
        x = self.pool(x)            # → (B, 256,  1,  1)
        x = x.view(x.size(0), -1)  # → (B, 256)
        x = self.dropout(x)
        x = self.fc(x)              # → (B, 40) raw logits
        return x
        # NOTE: sigmoid is NOT applied here.
        # BCEWithLogitsLoss does it during training.
        # predict.py does it during inference.

    def _init_weights(self):
        """
        Weight initialisation strategy:
          Conv2D  → Kaiming Normal (designed for ReLU activations)
          BatchNorm → weight=1, bias=0 (standard)
          Linear  → Xavier Normal (balances fan-in and fan-out)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


# ── Quick summary utility ─────────────────────────────────────

def model_summary(model):
    """Print parameter count and trace shapes through each block."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print("=" * 48)
    print("  MODEL: SimpleCNN (Plain CNN, from scratch)")
    print("=" * 48)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Approx size      : {total * 4 / 1024**2:.2f} MB (float32)")
    print()

    # Trace shapes (CPU, no grad)
    x = torch.zeros(1, 3, 128, 128)
    print(f"  Input  : {tuple(x.shape)}")
    x = model.block1(x); print(f"  Block1 : {tuple(x.shape)}  [Conv 3->32]")
    x = model.block2(x); print(f"  Block2 : {tuple(x.shape)}  [Conv 32->64]")
    x = model.block3(x); print(f"  Block3 : {tuple(x.shape)}  [Conv 64->128]")
    x = model.block4(x); print(f"  Block4 : {tuple(x.shape)}  [Conv 128->256]")
    x = model.pool(x);   print(f"  GAP    : {tuple(x.shape)}  [GlobalAvgPool]")
    x = x.view(x.size(0), -1)
    x = model.fc(x);     print(f"  Output : {tuple(x.shape)}  [FC->40 logits]")
    print("=" * 48)
    print("  [OK] Forward pass verified")
    print("=" * 48)


# ── Self-test ─────────────────────────────────────────────────

if __name__ == '__main__':
    model = SimpleCNN(num_classes=40, dropout=0.4)
    model_summary(model)

    dummy = torch.randn(4, 3, 128, 128)
    out   = model(dummy)
    assert out.shape == (4, 40), f"Shape error: {out.shape}"
    print(f"\n✅ Batch test passed: (4,3,128,128) → {tuple(out.shape)}")
