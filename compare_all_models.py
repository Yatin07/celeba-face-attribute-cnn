"""
compare_all_models.py
Loads all 4 model checkpoints with their CORRECT architectures and
runs all 4 test images through each, printing a side-by-side comparison.

Run from: C:\MLA\
    python compare_all_models.py
"""

import os, sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MODEL A: CPU (CelebA_Local) -- Small arch: 3->32->64->128->256
#          ~400K params | 128x128 input | trained 50k images, 15 epochs
# ─────────────────────────────────────────────────────────────────────────────
def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    )

class SimpleCNN_CPU(nn.Module):
    """CPU architecture: 3->32->64->128->256, ~400K params"""
    def __init__(self, num_classes=40, dropout=0.4):
        super().__init__()
        self.block1 = _conv_block(3,   32)
        self.block2 = _conv_block(32,  64)
        self.block3 = _conv_block(64,  128)
        self.block4 = _conv_block(128, 256)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B: GPU (CelebA_Local_GPU) -- Large arch: 3->64->128->256->256
#          ~2M params | 160x160 input | trained 162k images, 18 epochs
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCNN_GPU(nn.Module):
    """GPU architecture: 3->64->128->256->256, ~2M params"""
    def __init__(self, num_classes=40, dropout=0.4):
        super().__init__()
        self.block1 = _conv_block(3,   64)
        self.block2 = _conv_block(64,  128)
        self.block3 = _conv_block(128, 256)
        self.block4 = _conv_block(256, 256)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# CelebA 40 attribute names (official order)
# ─────────────────────────────────────────────────────────────────────────────
ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
    'Young'
]

THRESHOLD = 0.40   # >= this → attribute is "Detected"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

MODELS = [
    {
        "label":    "CPU-Stage1",
        "detail":   "stage1_cpu/train.py | 50k imgs | 128px | 15 ep | 90.55% acc",
        "path":     str(BASE_DIR / "stage1_cpu" / "checkpoints" / "best_model_stage1.pth"),
        "arch":     "cpu",      # SimpleCNN_CPU
        "is_dict":  True,       # saved as {'model_state_dict': ..., ...}
        "img_size": 128,
    },
    {
        "label":    "GPU-Stage2",
        "detail":   "stage2_gpu_modular/train.py | 162k imgs | 160px | 18 ep | 91.47% acc",
        "path":     str(BASE_DIR / "stage2_gpu_modular" / "checkpoints" / "best_model_stage2.pth"),
        "arch":     "gpu",      # SimpleCNN_GPU
        "is_dict":  True,
        "img_size": 160,
    },
    {
        "label":    "GPU-Best",
        "detail":   "stage3_gpu_single/celeba_full_gpu_training.py | 162k imgs | 160px | best epoch",
        "path":     str(BASE_DIR / "stage3_gpu_single" / "checkpoints" / "best_model_gpu.pth"),
        "arch":     "gpu",
        "is_dict":  False,      # raw model.state_dict()
        "img_size": 160,
    },
    {
        "label":    "GPU-Last",
        "detail":   "stage3_gpu_single/celeba_full_gpu_training.py | 162k imgs | 160px | last epoch",
        "path":     str(BASE_DIR / "stage3_gpu_single" / "checkpoints" / "last_model_gpu.pth"),
        "arch":     "gpu",
        "is_dict":  False,
        "img_size": 160,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST IMAGES
# ─────────────────────────────────────────────────────────────────────────────
TEST_IMAGES = [
    {
        "label": "download.jpg",
        "desc":  "Dwayne Johnson | Expected: Male, Bald, Smiling, 5oClock_Shadow",
        "path":  str(BASE_DIR / "test_images" / "download.jpg"),
    },
    {
        "label": "download2.jpg",
        "desc":  "Shraddha Kapoor | Expected: Female, Smiling, Heavy_Makeup, Young, Wearing_Lipstick",
        "path":  str(BASE_DIR / "test_images" / "download2.jpg"),
    },
    {
        "label": "download3.jpg",
        "desc":  "Older Bald Male | Expected: Male, Bald, Receding_Hairline, Big_Nose",
        "path":  str(BASE_DIR / "test_images" / "download3.jpg"),
    },
    {
        "label": "Image2.jpg",
        "desc":  "Young Indian Male | Expected: Male, Young, Black_Hair, Wearing_Necktie, Goatee",
        "path":  str(BASE_DIR / "test_images" / "Image2.jpg"),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg):
    """Load the right architecture with the right checkpoint."""
    if cfg["arch"] == "cpu":
        model = SimpleCNN_CPU(num_classes=40, dropout=0.4)
    else:
        model = SimpleCNN_GPU(num_classes=40, dropout=0.4)

    model.eval()
    ckpt = torch.load(cfg["path"], map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if cfg["is_dict"] else ckpt
    model.load_state_dict(state)
    return model


def preprocess(img_path, img_size):
    """Load image, resize, normalize to [-1, 1], add batch dim."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)   # HWC -> CHW
    tensor = (tensor - 0.5) / 0.5                      # normalize [-1, 1]
    return tensor.unsqueeze(0)                          # add batch dim -> (1,3,H,W)


def predict(model, img_path, img_size):
    """Return numpy array of 40 sigmoid probabilities."""
    img_tensor = preprocess(img_path, img_size)
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.sigmoid(logits)[0]
    return probs.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PRINTERS
# ─────────────────────────────────────────────────────────────────────────────

COL_W = 11   # width of each model column

def print_separator(char="-", width=88):
    print("  " + char * width)

def print_header_block():
    print()
    print("=" * 92)
    print("  CELEBA MULTI-MODEL PREDICTION COMPARISON  |  4 Models x 4 Images")
    print("=" * 92)
    print()
    print("  MODELS:")
    for i, m in enumerate(MODELS):
        print(f"    [{i+1}] {m['label']:<12}  {m['detail']}")
    print()
    print("  THRESHOLD: {:.0f}% confidence -> attribute DETECTED".format(THRESHOLD * 100))
    print("=" * 92)


def print_image_block(image_cfg, all_probs_dict):
    """Print full comparison table for one image."""
    labels  = [m["label"] for m in MODELS]
    probs_list = [all_probs_dict[m["label"]] for m in MODELS]

    print()
    print("-" * 92)
    print(f"  IMAGE : {image_cfg['label']}")
    print(f"  NOTE  : {image_cfg['desc']}")
    print("-" * 92)

    # Column headers
    hdr = f"  {'Attribute':<26}"
    for lbl in labels:
        hdr += f"  {lbl:^{COL_W}}"
    print(hdr)

    sub = f"  {'':26}"
    sub += f"  {'(50k/128px)':^{COL_W}}"
    sub += f"  {'(162k/160px)':^{COL_W}}"
    sub += f"  {'(162k/160px)':^{COL_W}}"
    sub += f"  {'(162k/160px)':^{COL_W}}"
    print(sub)
    print_separator()

    # Sort attributes: any model >= 15% first, rest at bottom
    interesting = []
    low_conf    = []
    for i, name in enumerate(ATTR_NAMES):
        vals = [p[i] for p in probs_list]
        row = (i, name, vals)
        if max(vals) >= 0.15:
            interesting.append(row)
        else:
            low_conf.append(row)

    # Sort by max confidence descending
    interesting.sort(key=lambda x: -max(x[2]))

    def fmt_cell(p):
        marker = "YES" if p >= THRESHOLD else "---"
        return f"{p*100:>5.1f}% {marker}"

    print(f"  [Attributes with >=15% confidence in at least one model]")
    print_separator(".")
    for i, name, vals in interesting:
        row = f"  {name:<26}"
        for p in vals:
            cell = fmt_cell(p)
            row += f"  {cell:^{COL_W}}"
        print(row)

    print_separator(".")
    print(f"  [Not detected anywhere -- <15% in all models]")
    not_det = [name for _, name, _ in low_conf]
    # print in wrapped lines of ~80 chars
    line = "  "
    for name in not_det:
        if len(line) + len(name) + 2 > 90:
            print(line.rstrip(", "))
            line = "  "
        line += name + ", "
    if line.strip():
        print(line.rstrip(", "))

    # Agreement summary
    print()
    print_separator(".")
    print(f"  [Model Agreement: which models agree on each detected attribute]")
    print_separator(".")

    detected_per_model = []
    for probs in probs_list:
        detected_per_model.append({ATTR_NAMES[i] for i, p in enumerate(probs) if p >= THRESHOLD})

    all_detected = set()
    for s in detected_per_model:
        all_detected |= s

    agr_hdr = f"  {'Attribute':<26}"
    for lbl in labels:
        agr_hdr += f"  {lbl:^{COL_W}}"
    print(agr_hdr)
    print_separator(".")

    for attr in sorted(all_detected):
        row = f"  {attr:<26}"
        for det_set in detected_per_model:
            mark = " DETECTED " if attr in det_set else "    ---   "
            row += f"  {mark:^{COL_W}}"
        print(row)

    # Count summary
    print()
    print(f"  Total attrs detected (>={THRESHOLD*100:.0f}%):")
    for m, det in zip(MODELS, detected_per_model):
        print(f"    {m['label']:<12}  {len(det):>2} attrs")

    print()


def print_global_summary(all_results):
    """Print a compact cross-image, cross-model summary."""
    print()
    print("=" * 92)
    print("  GLOBAL SUMMARY: Correct detections vs. Expected")
    print("=" * 92)
    print()

    expected = {
        "download.jpg":  ["Male", "Bald", "Smiling", "5_o_Clock_Shadow", "High_Cheekbones"],
        "download2.jpg": ["No_Beard", "Smiling", "Heavy_Makeup", "Young", "Wearing_Lipstick", "Attractive"],
        "download3.jpg": ["Male", "Bald", "Receding_Hairline", "Big_Nose", "Bags_Under_Eyes"],
        "Image2.jpg":    ["Male", "Young", "Black_Hair", "Wearing_Necktie", "Goatee", "Mustache", "Sideburns"],
    }

    header = f"  {'Image':<15}"
    for m in MODELS:
        header += f"  {m['label']:^12}"
    print(header)
    print_separator()

    for img_cfg in TEST_IMAGES:
        key  = img_cfg["label"]
        exp  = set(expected.get(key, []))
        row  = f"  {key:<15}"
        for m in MODELS:
            probs   = all_results[key][m["label"]]
            det     = {ATTR_NAMES[i] for i, p in enumerate(probs) if p >= THRESHOLD}
            correct = len(det & exp)
            total_e = len(exp)
            total_d = len(det)
            row += f"  {correct}/{total_e} exp, {total_d} tot  "[:14]
        print(row)

    print()
    print("  Legend: X/Y exp = X out of Y expected attrs correctly detected")
    print("          Z tot   = total attrs detected (including non-expected)")
    print()
    print("  ARCHITECTURE COMPARISON:")
    print("  +---------------------+-----------------------------+")
    print("  | Model               | Details                     |")
    print("  +---------------------+-----------------------------+")
    print("  | CPU-Stage1          | 3->32->64->128->256 (~400K) |")
    print("  | GPU-Stage2/Best/Last| 3->64->128->256->256 (~2M)  |")
    print("  +---------------------+-----------------------------+")
    print()
    print("=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_header_block()

    # Load all models
    print("\n[LOADING MODELS]")
    loaded_models = {}
    for cfg in MODELS:
        if not os.path.exists(cfg["path"]):
            print(f"  NOT FOUND : {cfg['label']} -> {cfg['path']}")
            loaded_models[cfg["label"]] = None
            continue
        try:
            m = load_model(cfg)
            loaded_models[cfg["label"]] = (m, cfg)
            params = sum(p.numel() for p in m.parameters())
            print(f"  Loaded OK : {cfg['label']:<12}  arch={cfg['arch'].upper()}  "
                  f"params={params:,}  img={cfg['img_size']}px")
        except Exception as e:
            print(f"  ERROR     : {cfg['label']} -- {e}")
            loaded_models[cfg["label"]] = None

    print()

    # Run inference on all images
    all_results = {}   # {image_label: {model_label: probs_array}}

    for img_cfg in TEST_IMAGES:
        img_label = img_cfg["label"]
        all_results[img_label] = {}

        if not os.path.exists(img_cfg["path"]):
            print(f"  [SKIP] Image not found: {img_cfg['path']}")
            continue

        for cfg in MODELS:
            entry = loaded_models.get(cfg["label"])
            if entry is None:
                all_results[img_label][cfg["label"]] = np.zeros(40)
                continue
            model, mcfg = entry
            probs = predict(model, img_cfg["path"], mcfg["img_size"])
            all_results[img_label][cfg["label"]] = probs

        # Print this image's table
        print_image_block(img_cfg, all_results[img_label])

    # Global summary
    print_global_summary(all_results)
    print("Done. Full output saved to: model_comparison_output.txt\n")


if __name__ == "__main__":
    # Write output to both console and file simultaneously
    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    out_file = open(r"C:\MLA\model_comparison_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, out_file)

    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        out_file.close()
        print("Output also saved to: C:\\MLA\\model_comparison_output.txt")
