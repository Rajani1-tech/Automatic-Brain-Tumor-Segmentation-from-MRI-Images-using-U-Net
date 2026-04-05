import os
import sys
import json

import torch

sys.path.append(os.path.abspath("."))

from configs.config import Config
from src.models.unet import UNetModel
from src.models.attention_unet import AttentionUNetModel
from src.models.tumor_classifier import TumorClassifier
from src.models.grade_classifier import GradeClassifier
from src.training.trainer import SegmentationTrainer, ClassifierTrainer
from src.training.grade_trainer import GradeTrainer
from src.inference.predict import SegmentationPredictor, ClassifierPredictor
from src.evaluation.metrics import SegmentationEvaluator
from src.visualization.visualize import Visualizer

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

viz           = Visualizer()
all_histories = {}


print("\n" + "=" * 60)
print("  Phase 1 — Binary Segmentation")
print("=" * 60)

unet_bin      = UNetModel(mode="binary")
trainer       = SegmentationTrainer(unet_bin, mode="binary", augment=True)
hist_unet_bin = trainer.train(save_path=Config.UNET_BINARY_PATH)
all_histories["UNet Binary"] = hist_unet_bin
viz.plot_segmentation_history(hist_unet_bin, "U-Net (Binary)",
                               "outputs/unet_binary_history.png")

attn_bin      = AttentionUNetModel(mode="binary")
trainer       = SegmentationTrainer(attn_bin, mode="binary", augment=True)
hist_attn_bin = trainer.train(save_path=Config.ATTN_UNET_BINARY_PATH)
all_histories["AttentionUNet Binary"] = hist_attn_bin
viz.plot_segmentation_history(hist_attn_bin, "Attention U-Net (Binary)",
                               "outputs/attn_unet_binary_history.png")

pred_unet_bin    = SegmentationPredictor(UNetModel(mode="binary"),
                                          Config.UNET_BINARY_PATH, mode="binary")
pred_attn_bin    = SegmentationPredictor(AttentionUNetModel(mode="binary"),
                                          Config.ATTN_UNET_BINARY_PATH, mode="binary")
metrics_unet_bin = SegmentationEvaluator(pred_unet_bin).evaluate_dataset(
    Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)
metrics_attn_bin = SegmentationEvaluator(pred_attn_bin).evaluate_dataset(
    Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)

print("\nBinary Results:")
for name, m in [("U-Net", metrics_unet_bin), ("Attn U-Net", metrics_attn_bin)]:
    print(f"  {name:<14} | dice {m['dice']:.4f} | iou {m['iou']:.4f} | "
          f"acc {m['accuracy']:.4f}")


print("\n" + "=" * 60)
print("  Phase 2 — Multiclass Segmentation")
print("=" * 60)

unet_multi      = UNetModel(mode="multiclass")
trainer         = SegmentationTrainer(unet_multi, mode="multiclass", augment=True)
hist_unet_multi = trainer.train(save_path=Config.UNET_MULTI_PATH)
all_histories["UNet Multiclass"] = hist_unet_multi
viz.plot_segmentation_history(hist_unet_multi, "U-Net (Multiclass)",
                               "outputs/unet_multiclass_history.png")

attn_multi      = AttentionUNetModel(mode="multiclass")
trainer         = SegmentationTrainer(attn_multi, mode="multiclass", augment=True)
hist_attn_multi = trainer.train(save_path=Config.ATTN_UNET_MULTI_PATH)
all_histories["AttentionUNet Multiclass"] = hist_attn_multi
viz.plot_segmentation_history(hist_attn_multi, "Attention U-Net (Multiclass)",
                               "outputs/attn_unet_multiclass_history.png")

pred_unet_multi    = SegmentationPredictor(UNetModel(mode="multiclass"),
                                            Config.UNET_MULTI_PATH, mode="multiclass")
pred_attn_multi    = SegmentationPredictor(AttentionUNetModel(mode="multiclass"),
                                            Config.ATTN_UNET_MULTI_PATH, mode="multiclass")
metrics_unet_multi = SegmentationEvaluator(pred_unet_multi).evaluate_dataset(
    Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)
metrics_attn_multi = SegmentationEvaluator(pred_attn_multi).evaluate_dataset(
    Config.TEST_IMAGES_DIR, Config.TEST_MASKS_DIR)

print("\nMulticlass Results:")
for name, m in [("U-Net", metrics_unet_multi), ("Attn U-Net", metrics_attn_multi)]:
    print(f"  {name:<14} | mean_dice {m['mean_dice']:.4f} | mean_iou {m['mean_iou']:.4f}")
    for cls_name, cls_m in m["per_class"].items():
        print(f"    {cls_name:<20} dice {cls_m['dice']:.4f} | iou {cls_m['iou']:.4f}")


print("\n" + "=" * 60)
print("  Phase 3 — Comparison")
print("=" * 60)

viz.plot_comparison(metrics_attn_bin, metrics_attn_multi,
                    "outputs/binary_vs_multiclass.png")
viz.plot_per_class_dice(metrics_attn_multi, "outputs/per_class_dice.png")
viz.plot_all_histories(all_histories, "outputs/all_models_training_curves.png")

all_metrics = {
    "unet_binary":      metrics_unet_bin,
    "attn_unet_binary": metrics_attn_bin,
    "unet_multiclass":  metrics_unet_multi,
    "attn_unet_multi":  metrics_attn_multi,
}
with open("outputs/metrics_summary.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("  Metrics saved → outputs/metrics_summary.json")



print("\n" + "=" * 60)
print("  Phase 4 — Tumor Profile Classifier")
print("=" * 60)

classifier  = TumorClassifier(pretrained=True)
clf_trainer = ClassifierTrainer(classifier, augment=True)
hist_clf    = clf_trainer.train(epochs=30, save_path=Config.CLASSIFIER_PATH)
viz.plot_classifier_history(hist_clf, "Tumor Profile Classifier",
                             "outputs/classifier_history.png")


print("\n" + "=" * 60)
print("  Phase 5 — LGG / HGG Grade Classifier")
print("=" * 60)

grade_clf     = GradeClassifier(pretrained=True)
grade_trainer = GradeTrainer(grade_clf, augment=True)
hist_grade    = grade_trainer.train(
    train_dir  = Config.GRADE_TRAIN_DIR,
    val_dir    = Config.GRADE_VAL_DIR,
    epochs     = Config.GRADE_EPOCHS,
    save_path  = Config.GRADE_PATH,
)
viz.plot_classifier_history(hist_grade, "LGG/HGG Grade Classifier",
                             "outputs/grade_classifier_history.png")

# Save grade metrics to summary
all_metrics["grade_classifier"] = {
    "best_val_acc": max(hist_grade["val_acc"]),
    "final_train_acc": hist_grade["train_acc"][-1],
    "final_val_acc":   hist_grade["val_acc"][-1],
}
with open("outputs/metrics_summary.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("  Grade metrics saved → outputs/metrics_summary.json")


print("\n" + "=" * 60)
print("  Training complete")
print("=" * 60)
print(f"\n  Models  : {Config.UNET_BINARY_PATH}, {Config.ATTN_UNET_BINARY_PATH}")
print(f"            {Config.UNET_MULTI_PATH}, {Config.ATTN_UNET_MULTI_PATH}")
print(f"            {Config.CLASSIFIER_PATH}")
print(f"            {Config.GRADE_PATH}  ← NEW: LGG/HGG grade classifier")
print(f"\n  Outputs : outputs/")
print(f"\n  Run     : streamlit run app.py")