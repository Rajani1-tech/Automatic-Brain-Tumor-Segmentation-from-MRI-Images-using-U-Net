"""
evaluate_classifiers.py  —  NeuroScan AI
Full evaluation script for tumor profile and grade classifiers.

Usage:
    python evaluate_classifiers.py
"""

import os, sys, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))
from configs.config import Config
from src.models.tumor_classifier     import TumorClassifier
from src.models.grade_classifier     import GradeClassifier
from src.datasets.classifier_dataset import TumorProfileDataset

OUT    = pathlib.Path("outputs/evaluation")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Import GradeDataset — it lives inside grade_trainer.py
# ─────────────────────────────────────────────────────────────────────────────
def import_grade_dataset():
    # Try grade_trainer first (that's where GradeTrainer uses it)
    try:
        from src.training.grade_trainer import GradeDataset
        print("  GradeDataset  ← src.training.grade_trainer")
        return GradeDataset
    except ImportError:
        pass
    # Try datasets module
    try:
        from src.datasets.dataset import GradeDataset
        print("  GradeDataset  ← src.datasets.dataset")
        return GradeDataset
    except ImportError:
        pass
    # Try classifier_dataset
    try:
        from src.datasets.classifier_dataset import GradeDataset
        print("  GradeDataset  ← src.datasets.classifier_dataset")
        return GradeDataset
    except ImportError:
        pass
    raise ImportError(
        "Cannot find GradeDataset.\n"
        "Run: grep -rn 'class GradeDataset' src/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnose grade val folder
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_grade_folder():
    grade_val = pathlib.Path(Config.GRADE_VAL_DIR)
    print(f"\n  Grade val dir : {grade_val.resolve()}")
    print(f"  Exists        : {grade_val.exists()}")

    if not grade_val.exists():
        print(f"  ✗ Directory does not exist!")
        print(f"    Check Config.GRADE_VAL_DIR = '{Config.GRADE_VAL_DIR}'")
        return False

    subdirs = sorted([d for d in grade_val.iterdir() if d.is_dir()])
    files   = list(grade_val.glob("*.*"))
    print(f"  Subdirs       : {[d.name for d in subdirs]}")
    print(f"  Direct files  : {len(files)}")

    for sub in subdirs:
        imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg")) + \
               list(sub.glob("*.jpeg"))
        print(f"    {sub.name}/  → {len(imgs)} images")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, title, save_path, cmap='Blues'):
    fig, ax = plt.subplots(
        figsize=(max(5, len(class_names)*1.6), max(4, len(class_names)*1.4)),
        facecolor='white')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True',      fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved → {save_path}")


def plot_roc(y_true, y_probs, class_names, title, save_path):
    """Safe ROC — handles missing classes and size mismatches."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    colors  = ['#c0392b', '#27ae60', '#2980b9', '#8e44ad']
    n       = len(class_names)

    try:
        if n == 2:
            # Binary — use column 1 (positive class)
            scores = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs[:, 0]
            fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
            ax.plot(fpr, tpr, color=colors[0], lw=2,
                    label=f'AUC = {auc(fpr, tpr):.3f}')
        else:
            from sklearn.preprocessing import label_binarize
            all_labels = list(range(y_probs.shape[1]))
            y_bin      = label_binarize(y_true, classes=all_labels)
            for i, (name, col) in enumerate(zip(class_names, colors)):
                if i >= y_probs.shape[1]:
                    continue
                if y_bin.ndim > 1 and i < y_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                    ax.plot(fpr, tpr, color=col, lw=2,
                            label=f'{name}  AUC={auc(fpr, tpr):.3f}')
    except Exception as e:
        ax.text(0.5, 0.5, f'ROC unavailable:\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)

    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate',  fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved → {save_path}")


def plot_per_class_bars(report_dict, class_names, title, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    x       = np.arange(len(class_names))
    width   = 0.25
    colors  = ['#2980b9', '#27ae60', '#c0392b']
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*1.8), 4),
                           facecolor='white')
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(name, {}).get(metric, 0) for name in class_names]
        bars = ax.bar(x + i*width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation — handles missing classes gracefully
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(model, loader, class_names, tag, cmap):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    print(f"    Inference on {len(loader.dataset)} samples ...")

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            logits = model(imgs.to(DEVICE))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.vstack(all_probs)

    # Only evaluate classes present in val set
    present_labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    present_names  = [class_names[i] for i in present_labels
                      if i < len(class_names)]

    print(f"    Classes present : {present_names}")
    missing = [class_names[i] for i in range(len(class_names))
               if i not in present_labels]
    if missing:
        print(f"    ⚠ Absent in val : {missing}")

    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    print(f"\n{classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0)}")

    slug = tag.lower().replace(' ', '_')
    plot_confusion_matrix(cm, present_names,
                          f'{tag} — Confusion Matrix',
                          OUT / f'{slug}_confusion_matrix.png', cmap)
    plot_roc(y_true, y_probs, present_names,
             f'{tag} — ROC Curve',
             OUT / f'{slug}_roc_curve.png')
    plot_per_class_bars(report, present_names,
                        f'{tag} — Per-Class Metrics',
                        OUT / f'{slug}_per_class_metrics.png')

    return {
        "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred,
                                          average='weighted', zero_division=0)), 4),
        "recall_weighted":    round(float(recall_score(y_true, y_pred,
                                          average='weighted', zero_division=0)), 4),
        "f1_weighted":        round(float(f1_score(y_true, y_pred,
                                          average='weighted', zero_division=0)), 4),
        "classes_in_val":     present_names,
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall":    round(report[name]["recall"],    4),
                "f1":        round(report[name]["f1-score"],  4),
                "support":   int(report[name]["support"])
            } for name in present_names
        },
        "confusion_matrix": cm.tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NeuroScan AI — Classifier Evaluation")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    results = {}

    # ─────────────────────────────────────────────────────────────────
    # 1. Tumor Profile Classifier
    #    TumorClassifier(pretrained=False) — matches main.py
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[1/2] Tumor Profile Classifier")
    print("="*60)
    try:
        profile_ds = TumorProfileDataset(
            images_dir = Config.CLASSIFIER_VAL_IMAGES,
            masks_dir  = Config.CLASSIFIER_VAL_MASKS,
        )
        profile_loader = DataLoader(profile_ds, batch_size=32,
                                    shuffle=False, num_workers=2,
                                    pin_memory=True)

        profile_model = TumorClassifier(pretrained=False).to(DEVICE)
        ckpt  = torch.load(Config.CLASSIFIER_PATH,
                           map_location=DEVICE, weights_only=False)
        profile_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"  Loaded  : {Config.CLASSIFIER_PATH}")
        print(f"  Samples : {len(profile_ds)}")

        s = run_eval(profile_model, profile_loader,
                     list(Config.TUMOR_CLASS_NAMES),
                     'Tumor Profile Classifier', 'Blues')
        results['tumor_profile_classifier'] = s
        print(f"\n  Accuracy : {s['accuracy']:.4f}")
        print(f"  F1       : {s['f1_weighted']:.4f}")

    except Exception as e:
        import traceback
        print(f"\n  [FAILED] {e}")
        print(traceback.format_exc())

    # ─────────────────────────────────────────────────────────────────
    # 2. Grade Classifier  (LGG / HGG)
    #    GradeClassifier(pretrained=False) — matches main.py
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[2/2] Grade Classifier  (LGG / HGG)")
    print("="*60)
    try:
        # Diagnose folder structure first
        folder_ok = diagnose_grade_folder()
        if not folder_ok:
            raise RuntimeError(
                f"Grade val folder not found: {Config.GRADE_VAL_DIR}\n"
                f"Run: ls brain-tumour_seg_dataset/dataset/classifier/"
            )

        # Import GradeDataset
        GradeDataset = import_grade_dataset()

        # Instantiate — try with/without augment kwarg
        grade_ds = None
        for kwargs in [{'augment': False}, {}]:
            try:
                grade_ds = GradeDataset(Config.GRADE_VAL_DIR, **kwargs)
                break
            except TypeError:
                continue
        if grade_ds is None:
            raise RuntimeError("Cannot instantiate GradeDataset")

        print(f"  Samples : {len(grade_ds)}")

        # Show label distribution
        print("  Checking label distribution (first 200 samples)...")
        sample_size = min(200, len(grade_ds))
        sample_labels = [int(grade_ds[i][1]) for i in range(sample_size)]
        unique, counts = np.unique(sample_labels, return_counts=True)
        for u, c in zip(unique, counts):
            name = Config.GRADE_CLASS_NAMES[u] \
                   if u < len(Config.GRADE_CLASS_NAMES) else f"class_{u}"
            print(f"    label {u} ({name}): {c}/{sample_size} in sample")

        grade_loader = DataLoader(grade_ds, batch_size=32,
                                  shuffle=False, num_workers=2,
                                  pin_memory=True)

        grade_model = GradeClassifier(pretrained=False).to(DEVICE)
        ckpt  = torch.load(Config.GRADE_PATH,
                           map_location=DEVICE, weights_only=False)
        grade_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"  Loaded  : {Config.GRADE_PATH}")

        s = run_eval(grade_model, grade_loader,
                     list(Config.GRADE_CLASS_NAMES),
                     'Grade Classifier', 'Reds')
        results['grade_classifier'] = s
        print(f"\n  Accuracy : {s['accuracy']:.4f}")
        print(f"  F1       : {s['f1_weighted']:.4f}")

    except Exception as e:
        import traceback
        print(f"\n  [FAILED] {e}")
        print(traceback.format_exc())
        print("\n  DEBUG: run these commands and paste output:")
        print("    find brain-tumour_seg_dataset/dataset/classifier/val -type d")
        print("    find brain-tumour_seg_dataset/dataset/classifier/val -type f | head -10")
        print("    grep -n 'class GradeDataset\\|def __init__\\|label\\|lgg\\|hgg' src/training/grade_trainer.py | head -30")

    # ── Save JSON ─────────────────────────────────────────────────────
    out_json = OUT / 'classifier_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, s in results.items():
        print(f"\n  {name}")
        print(f"    Accuracy  : {s['accuracy']:.4f}")
        print(f"    Precision : {s['precision_weighted']:.4f}")
        print(f"    Recall    : {s['recall_weighted']:.4f}")
        print(f"    F1        : {s['f1_weighted']:.4f}")
        for cls, m in s['per_class'].items():
            print(f"    {cls:<22}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")

    print(f"\n  JSON  → {out_json}")
    print(f"  Plots → {OUT}/")
    print("="*60)


if __name__ == '__main__':
    main()