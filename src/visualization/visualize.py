import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import Config

_COLORS_BGR = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

TRAIN_COLOR   = "#4fc3f7"
VAL_COLOR     = "#ef5350"
IOU_COLOR     = "#66bb6a"
VAL_IOU_COLOR = "#ffa726"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#444",
        "axes.labelcolor":  "#ccc",
        "xtick.color":      "#aaa",
        "ytick.color":      "#aaa",
        "text.color":       "#eee",
        "grid.color":       "#333",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "lines.linewidth":  2.0,
        "font.size":        10,
        "legend.framealpha": 0.3,
        "legend.edgecolor": "#555",
    })


def _save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved → {path}")


def _get(history, *keys):
    """
    Flexible history key lookup.
    Tries each key in order and returns the first match.
    Handles both naming conventions:
      "loss" / "train_loss"
      "val_loss"
      "acc"  / "train_acc"
      "val_acc"
    """
    for k in keys:
        if k in history:
            return history[k]
    raise KeyError(f"None of {keys} found in history. "
                   f"Available keys: {list(history.keys())}")


class Visualizer:

    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        h, w  = mask.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        if mask.max() > 3:
            color[mask > 127] = _COLORS_BGR[1]
            return color
        for idx, bgr in enumerate(_COLORS_BGR):
            color[mask == idx] = bgr
        return color

    def overlay(self, gray: np.ndarray, color_mask: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
        if gray.dtype != np.uint8:
            gray = (gray * 255).clip(0, 255).astype(np.uint8)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if gray_bgr.shape[:2] != color_mask.shape[:2]:
            color_mask = cv2.resize(color_mask,
                                    (gray_bgr.shape[1], gray_bgr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        tumor   = np.any(color_mask > 0, axis=2)
        blended = gray_bgr.copy()
        blended[tumor] = cv2.addWeighted(
            gray_bgr, 1 - alpha, color_mask, alpha, 0)[tumor]
        return blended

    def compute_area_stats(self, mask: np.ndarray) -> dict:
        total = mask.size
        return {
            name: {"pixels":     int((mask == idx).sum()),
                   "percentage": round(int((mask == idx).sum()) / total * 100, 3)}
            for name, idx in [("NCR/NET", 1), ("Edema", 2), ("Enhancing Tumor", 3)]
        }

    def plot_segmentation_history(self, history: dict, model_name: str,
                                   save_path: str) -> None:
        _style()
        epochs = range(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"{model_name} — Training History", fontsize=14, y=1.01)

        ax = axes[0, 0]
        ax.plot(epochs, history["loss"],     color=TRAIN_COLOR, label="Train Loss")
        ax.plot(epochs, history["val_loss"], color=VAL_COLOR,   label="Val Loss",
                linestyle="--")
        ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True)

        ax = axes[0, 1]
        ax.plot(epochs, history["dice"],     color=TRAIN_COLOR, label="Train Dice")
        ax.plot(epochs, history["val_dice"], color=VAL_COLOR,   label="Val Dice",
                linestyle="--")
        ax.set_title("Dice Coefficient"); ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(epochs, history["iou"],     color=IOU_COLOR,     label="Train IoU")
        ax.plot(epochs, history["val_iou"], color=VAL_IOU_COLOR, label="Val IoU",
                linestyle="--")
        ax.set_title("IoU Score"); ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True)

        ax   = axes[1, 1]
        t_a  = np.array(history["loss"])
        v_a  = np.array(history["val_loss"])
        ax.fill_between(epochs, t_a, v_a, where=(v_a >= t_a),
                        alpha=0.25, color=VAL_COLOR,   label="Overfit region")
        ax.fill_between(epochs, t_a, v_a, where=(v_a < t_a),
                        alpha=0.25, color=TRAIN_COLOR, label="Underfit region")
        ax.plot(epochs, t_a, color=TRAIN_COLOR, label="Train Loss")
        ax.plot(epochs, v_a, color=VAL_COLOR,   label="Val Loss", linestyle="--")
        ax.set_title("Overfitting Diagnostic"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

        _save(fig, save_path)

    def plot_classifier_history(self, history: dict, model_name: str,
                                 save_path: str) -> None:
        """
        Flexible key lookup handles both naming conventions:
          Standard trainer : "loss", "val_loss", "acc",        "val_acc"
          Grade trainer    : "train_loss", "val_loss", "train_acc", "val_acc"
        """
        _style()

        train_loss = _get(history, "loss",      "train_loss")
        val_loss   = _get(history, "val_loss")
        train_acc  = _get(history, "acc",       "train_acc")
        val_acc    = _get(history, "val_acc")

        epochs = range(1, len(train_loss) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{model_name} — Training History", fontsize=13)

        axes[0].plot(epochs, train_loss, color=TRAIN_COLOR, label="Train Loss")
        axes[0].plot(epochs, val_loss,   color=VAL_COLOR,   label="Val Loss",
                     linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs, train_acc, color=TRAIN_COLOR, label="Train Acc")
        axes[1].plot(epochs, val_acc,   color=VAL_COLOR,   label="Val Acc",
                     linestyle="--")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(True)

        _save(fig, save_path)

    def plot_all_histories(self, histories: dict, save_path: str) -> None:
        _style()
        palette = ["#4fc3f7", "#ef5350", "#66bb6a", "#ffa726",
                   "#ce93d8", "#80cbc4", "#ffcc02", "#f48fb1"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("All Models — Training Comparison", fontsize=14)

        for metric, title, ax in [("val_loss", "Val Loss",  axes[0]),
                                   ("val_dice", "Val Dice",  axes[1]),
                                   ("val_iou",  "Val IoU",   axes[2])]:
            for i, (name, hist) in enumerate(histories.items()):
                if metric not in hist:
                    continue
                ax.plot(range(1, len(hist[metric]) + 1), hist[metric],
                        color=palette[i % len(palette)], label=name)
            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.legend(fontsize=8); ax.grid(True)

        _save(fig, save_path)

    def plot_comparison(self, bin_metrics: dict, multi_metrics: dict,
                         save_path: str) -> None:
        _style()
        metrics    = ["Dice / Mean Dice", "IoU / Mean IoU", "Accuracy", "F1 Score"]
        bin_vals   = [bin_metrics.get("dice",      0), bin_metrics.get("iou",      0),
                      bin_metrics.get("accuracy",  0), bin_metrics.get("f1_score", 0)]
        multi_vals = [multi_metrics.get("mean_dice", 0), multi_metrics.get("mean_iou",  0),
                      multi_metrics.get("accuracy",  0), multi_metrics.get("f1_score",  0)]

        x   = np.arange(len(metrics))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle("Binary vs Multiclass Segmentation", fontsize=13)

        bars1 = ax.bar(x - w/2, bin_vals,   w, label="Binary",
                       color=TRAIN_COLOR, alpha=0.85)
        bars2 = ax.bar(x + w/2, multi_vals, w, label="Multiclass",
                       color=VAL_COLOR,   alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
        ax.legend(); ax.grid(True, axis="y")

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        _save(fig, save_path)

    def plot_per_class_dice(self, multi_metrics: dict, save_path: str) -> None:
        _style()
        per_class = multi_metrics.get("per_class", {})
        if not per_class:
            return

        names  = list(per_class.keys())
        dice_v = [per_class[n]["dice"] for n in names]
        iou_v  = [per_class[n]["iou"]  for n in names]
        y      = np.arange(len(names))
        w      = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Per-Class Dice & IoU (Multiclass)", fontsize=13)

        ax.barh(y + w/2, dice_v, w, label="Dice", color=TRAIN_COLOR, alpha=0.85)
        ax.barh(y - w/2, iou_v,  w, label="IoU",  color=IOU_COLOR,   alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlim(0, 1.1); ax.set_xlabel("Score")
        ax.legend(); ax.grid(True, axis="x")

        for i, (d, u) in enumerate(zip(dice_v, iou_v)):
            ax.text(d + 0.01, i + w/2, f"{d:.3f}", va="center", fontsize=8)
            ax.text(u + 0.01, i - w/2, f"{u:.3f}", va="center", fontsize=8)

        _save(fig, save_path)

    def plot_grade_classifier_history(self, history: dict,
                                       save_path: str) -> None:
        """
        Dedicated plot for LGG/HGG grade classifier.
        Shows loss, accuracy, and class-level precision/recall if available.
        Handles key names from GradeClassifier trainer.
        """
        _style()

        train_loss = _get(history, "train_loss", "loss")
        val_loss   = _get(history, "val_loss")
        train_acc  = _get(history, "train_acc",  "acc")
        val_acc    = _get(history, "val_acc")

        epochs     = range(1, len(train_loss) + 1)
        has_auc    = "val_auc" in history

        ncols = 3 if has_auc else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        fig.suptitle("Grade Classifier (LGG vs HGG) — Training History",
                     fontsize=13)

        axes[0].plot(epochs, train_loss, color=TRAIN_COLOR, label="Train Loss")
        axes[0].plot(epochs, val_loss,   color=VAL_COLOR,   label="Val Loss",
                     linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs, train_acc, color=TRAIN_COLOR, label="Train Acc")
        axes[1].plot(epochs, val_acc,   color=VAL_COLOR,   label="Val Acc",
                     linestyle="--")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(True)

        if has_auc:
            axes[2].plot(epochs, history["val_auc"], color=IOU_COLOR,
                         label="Val AUC")
            axes[2].set_title("ROC-AUC"); axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("AUC"); axes[2].set_ylim(0, 1)
            axes[2].legend(); axes[2].grid(True)

        _save(fig, save_path)