import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path

sys.path.append(os.path.abspath("."))

from configs.config import Config
from src.models.unet import UNetModel
from src.models.attention_unet import AttentionUNetModel
from src.inference.predict import Predictor
from src.evaluation.metrics import Evaluator

# ── Output directories 
os.makedirs("outputs/overlays",     exist_ok=True)
os.makedirs("outputs/comparisons",  exist_ok=True)
os.makedirs("outputs/heatmaps",     exist_ok=True)
os.makedirs("outputs/full_figures", exist_ok=True)

# ── Config 
NUM_SAMPLES    = 10       # how many test images to visualize
OVERLAY_ALPHA  = 0.45     # opacity of tumor highlight
HEATMAP_ALPHA  = 0.55     # opacity of attention heatmap
TUMOR_COLOR    = (220, 40, 40)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def load_sample(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask  = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (Config.IMG_HEIGHT, Config.IMG_WIDTH))
    mask  = cv2.resize(mask,  (Config.IMG_HEIGHT, Config.IMG_WIDTH))

    image = image.astype(np.float32) / 255.0
    mask  = (mask > 127).astype(np.float32)
    return image, mask


def get_test_pairs(images_dir, masks_dir, n=NUM_SAMPLES):
    img_dir  = Path(images_dir)
    mask_dir = Path(masks_dir)

    pairs = []

    mask_dict = {}
    for m in mask_dir.iterdir():
        if m.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            key = m.stem.replace("mask_", "")
            mask_dict[key] = m

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        key = img_path.stem.replace("img_", "")
        if key in mask_dict:
            pairs.append((img_path, mask_dict[key]))
        if len(pairs) >= n:
            break

    return pairs


def make_overlay(mri_np, pred_mask_np, alpha=OVERLAY_ALPHA, color=TUMOR_COLOR):
    mri_rgb     = (np.stack([mri_np] * 3, axis=-1) * 255).astype(np.uint8)
    color_layer = np.zeros_like(mri_rgb)
    color_layer[pred_mask_np == 1] = color
    mask_3ch    = np.stack([pred_mask_np] * 3, axis=-1).astype(bool)
    return np.where(
        mask_3ch,
        (mri_rgb * (1 - alpha) + color_layer * alpha).astype(np.uint8),
        mri_rgb
    )


def make_heatmap_overlay(mri_np, heatmap_np, alpha=HEATMAP_ALPHA):
    mri_u8   = (mri_np * 255).astype(np.uint8)
    heat_u8  = (heatmap_np * 255).astype(np.uint8)
    colored  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    mri_rgb  = cv2.cvtColor(mri_u8, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(mri_rgb, 1 - alpha, colored, alpha, 0)


def run_evaluation(predictor, label):
    print(f"\nEvaluating {label}...")
    evaluator = Evaluator(predictor)
    metrics   = evaluator.evaluate_dataset(Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR)
    print(f"\n{label} Results:")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")
    return metrics


def run_visualization(predictor_unet, predictor_att, test_pairs):
    print(f"\nGenerating visualizations for {len(test_pairs)} samples...\n")

    for i, (img_path, mask_path) in enumerate(test_pairs):
        mri, gt = load_sample(img_path, mask_path)

        # ── run inference
        pred_unet = predictor_unet.predict(str(img_path))    # (H,W) binary numpy
        pred_att  = predictor_att.predict(str(img_path))

        overlay_unet = make_overlay(mri, pred_unet)
        overlay_att  = make_overlay(mri, pred_att)

        # ── 1. Overlays: Original | U-Net overlay | Attention U-Net overlay
        # FIX: Added U-Net overlay alongside Attention U-Net overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mri,          cmap="gray"); axes[0].set_title("Original MRI")
        axes[1].imshow(overlay_unet);              axes[1].set_title("Predicted tumor (U-Net)")
        axes[2].imshow(overlay_att);               axes[2].set_title("Predicted tumor (Attention U-Net)")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"outputs/overlays/sample_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── 2. Side-by-side comparison (unchanged — already working well)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(mri,       cmap="gray");                      axes[0].set_title("(a) Input MRI")
        axes[1].imshow(gt,        cmap="Reds", vmin=0, vmax=1);      axes[1].set_title("(b) Ground truth")
        axes[2].imshow(pred_unet, cmap="Reds", vmin=0, vmax=1);      axes[2].set_title("(c) U-Net prediction")
        axes[3].imshow(pred_att,  cmap="Reds", vmin=0, vmax=1);      axes[3].set_title("(d) Attention U-Net")
        tumor_patch = mpatches.Patch(color="red", alpha=0.6, label="Tumor")
        fig.legend(handles=[tumor_patch], loc="lower center", fontsize=10)
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Sample {i:02d} — {img_path.name}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"outputs/comparisons/sample_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── 3. Attention heatmap
        # FIX: Moved get_attention_map() call OUTSIDE the inner if-block so heatmap
        #      is computed once and reused. Also added a fallback warning so you know
        #      exactly why heatmaps are skipped if the method is missing.
        attn_map = None
        heat_img = None
        if hasattr(predictor_att, "get_attention_map"):
            try:
                attn_map = predictor_att.get_attention_map(str(img_path))   # (H,W) [0,1]
                heat_img = make_heatmap_overlay(mri, attn_map)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(mri,      cmap="gray"); axes[0].set_title("(a) Input MRI")
                axes[1].imshow(heat_img);              axes[1].set_title("(b) Attention heatmap")
                axes[2].imshow(pred_att, cmap="Reds"); axes[2].set_title("(c) Prediction")
                for ax in axes: ax.axis("off")
                plt.tight_layout()
                plt.savefig(f"outputs/heatmaps/sample_{i:02d}.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"  [WARNING] Heatmap failed for sample {i:02d}: {e}")
        else:
            if i == 0:  # Print warning only once
                print("  [WARNING] predictor_att has no get_attention_map() method — "
                      "heatmaps will be skipped. Implement this method in your Predictor "
                      "to aggregate and return attention gate weights as a (H,W) map.")

        # ── 4. Full 5-panel paper figure
        # FIX: Attention U-Net panel now shows pred_att on white background with red
        #      (cmap="Reds") instead of the coloured overlay, matching U-Net style.
        #      Heatmap panel (panel e) is only added when attn_map was computed above.
        has_attn = attn_map is not None
        ncols    = 5 if has_attn else 4
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        axes[0].imshow(mri,       cmap="gray");                      axes[0].set_title("(a) MRI input")
        axes[1].imshow(gt,        cmap="Reds", vmin=0, vmax=1);      axes[1].set_title("(b) Ground truth")
        axes[2].imshow(pred_unet, cmap="Reds", vmin=0, vmax=1);      axes[2].set_title("(c) U-Net")
        # FIX: was `overlay_att` (brain background); now `pred_att` with cmap="Reds"
        axes[3].imshow(pred_att,  cmap="Reds", vmin=0, vmax=1);      axes[3].set_title("(d) Att. U-Net")

        if has_attn:
            axes[4].imshow(heat_img);                                axes[4].set_title("(e) Attention heatmap")

        for ax in axes: ax.axis("off")
        plt.suptitle(f"Figure — Sample {i:02d}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"outputs/full_figures/sample_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  [{i+1:02d}/{len(test_pairs)}] saved  →  {img_path.name}")


def print_comparison(metrics_unet, metrics_att):
    print("\n" + "═" * 52)
    print(f"  {'Metric':<28} {'U-Net':>8}  {'Att-UNet':>8}")
    print("─" * 52)
    all_keys = set(metrics_unet) | set(metrics_att)
    for k in sorted(all_keys):
        u = metrics_unet.get(k, float("nan"))
        a = metrics_att.get(k,  float("nan"))
        winner = "  ◀" if a > u else ""
        print(f"  {k:<28} {u:>8.4f}  {a:>8.4f}{winner}")
    print("═" * 52)
    print("  ◀ = Attention U-Net wins\n")


if __name__ == "__main__":

    for path in ["models/unet.pth", "models/attention_unet.pth"]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run main.py first to train and save the models."
            )

    print("\nLoading saved models...")
    predictor_unet = Predictor("models/unet.pth",           model=UNetModel())
    predictor_att  = Predictor("models/attention_unet.pth",  model=AttentionUNetModel())
    print("  models/unet.pth             ✓")
    print("  models/attention_unet.pth   ✓")

    # metrics_unet = run_evaluation(predictor_unet, "U-Net")
    # metrics_att  = run_evaluation(predictor_att,  "Attention U-Net")
    # print_comparison(metrics_unet, metrics_att)

    test_pairs = get_test_pairs(Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)
    if not test_pairs:
        print("WARNING: No test image/mask pairs found. Check your Config paths.")
    else:
        run_visualization(predictor_unet, predictor_att, test_pairs)
        print(f"\nAll outputs saved to outputs/")
        print("  overlays/        →  MRI | U-Net overlay | Att-UNet overlay")
        print("  comparisons/     →  input | GT | U-Net | Att-UNet")
        print("  heatmaps/        →  attention heatmap overlay")
        print("  full_figures/    →  4 or 5-panel paper figure")