import os
from collections import defaultdict

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from configs.config import Config
from src.inference.predict import SegmentationPredictor

MOD_DETECT_ORDER = ["t1ce", "t1", "t2", "flair"]


def _detect_modality(filename):
    stem = os.path.splitext(filename)[0].lower()
    for mod in MOD_DETECT_ORDER:
        if f"_{mod}_" in stem:
            return mod
    return None


def _build_slice_to_image(images_dir, target_mod="flair"):
    """
    Build lookup: slice_key → image filename for a given modality.

    slice_key is the mask stem, e.g. "BraTS2021_00000_000"
    image file is e.g.  "BraTS2021_00000_flair_000.png"

    For binary evaluation we use flair as the representative modality
    since the binary model was trained on every-4th image (flair comes
    first alphabetically, so it was the representative during training).
    """
    lookup = {}
    for f in os.listdir(images_dir):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mod = _detect_modality(f)
        if mod != target_mod:
            continue
        stem      = os.path.splitext(f)[0]
        slice_key = stem.replace(f"_{mod}_", "_")
        lookup[slice_key] = f
    return lookup


class SegmentationEvaluator:

    def __init__(self, predictor: SegmentationPredictor):
        self.predictor = predictor
        self.mode      = predictor.mode

    @staticmethod
    def _dice(y_true, y_pred, smooth=1.0):
        y_t = y_true.flatten().astype(np.float32)
        y_p = y_pred.flatten().astype(np.float32)
        return (2 * np.dot(y_t, y_p) + smooth) / (y_t.sum() + y_p.sum() + smooth)

    @staticmethod
    def _iou(y_true, y_pred, smooth=1.0):
        y_t   = y_true.flatten().astype(np.float32)
        y_p   = y_pred.flatten().astype(np.float32)
        inter = np.dot(y_t, y_p)
        union = y_t.sum() + y_p.sum() - inter
        return (inter + smooth) / (union + smooth)

    def evaluate_dataset(self, images_dir, masks_dir):
        if self.mode == "binary":
            return self._evaluate_binary(images_dir, masks_dir)
        return self._evaluate_multiclass(images_dir, masks_dir)

    def _evaluate_binary(self, images_dir, masks_dir):
        """
        Binary evaluation using stem-based image-mask pairing.

        Problem with naive zip(sorted_images, sorted_masks):
          images has 4× more files than masks (4 modalities per slice).
          Alphabetical sort puts ALL flair slices first, then ALL t1, etc.
          zip pairs flair_slice_001 with mask_000 of a different patient → dice ~0.14.

        Fix: build slice_key → flair_filename lookup, then iterate masks
        and look up the correct flair image for each mask.
        Uses flair as representative because it was every-4th image during training
        (flair sorts first alphabetically → index 0, 4, 8, ... → picked by all_images[::4]).
        """
        all_masks = sorted(f for f in os.listdir(masks_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))

        slice_to_img = _build_slice_to_image(images_dir, target_mod="flair")

        y_true_all, y_pred_all = [], []
        dice_scores, iou_scores = [], []
        skipped = 0

        for mask_name in all_masks:
            mask_gt = cv2.imread(os.path.join(masks_dir, mask_name),
                                 cv2.IMREAD_GRAYSCALE)
            if mask_gt is None:
                continue

            mask_stem = os.path.splitext(mask_name)[0]
            img_file  = slice_to_img.get(mask_stem)
            if img_file is None:
                skipped += 1
                continue

            mask_pred = self.predictor.predict(os.path.join(images_dir, img_file))
            if mask_pred.shape != mask_gt.shape:
                mask_pred = cv2.resize(mask_pred,
                                       (mask_gt.shape[1], mask_gt.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            gt_bin   = (mask_gt   > 0).astype(np.int32)
            pred_bin = (mask_pred > 127).astype(np.int32)

            y_true_all.extend(gt_bin.flatten())
            y_pred_all.extend(pred_bin.flatten())
            dice_scores.append(self._dice(gt_bin, pred_bin))
            iou_scores.append(self._iou(gt_bin, pred_bin))

        if skipped:
            print(f"[SegmentationEvaluator] binary: {skipped} masks had no matching image")

        return {
            "mode":      "binary",
            "accuracy":  accuracy_score(y_true_all, y_pred_all),
            "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
            "recall":    recall_score(y_true_all, y_pred_all, zero_division=0),
            "f1_score":  f1_score(y_true_all, y_pred_all, zero_division=0),
            "dice":      float(np.mean(dice_scores)),
            "iou":       float(np.mean(iou_scores)),
        }

    def _evaluate_multiclass(self, images_dir, masks_dir):
        """
        Multiclass evaluation using t1ce as representative image.
        predictor.predict() for multiclass auto-loads all 4 modalities
        internally given any single modality file from that slice.
        """
        num_classes            = Config.NUM_CLASSES
        class_dice             = [[] for _ in range(num_classes)]
        class_iou              = [[] for _ in range(num_classes)]
        y_true_all, y_pred_all = [], []

        all_masks     = sorted(f for f in os.listdir(masks_dir)
                               if f.lower().endswith((".png", ".jpg", ".jpeg")))
        slice_to_t1ce = _build_slice_to_image(images_dir, target_mod="t1ce")
        skipped       = 0

        for mask_name in all_masks:
            mask_gt = cv2.imread(os.path.join(masks_dir, mask_name),
                                 cv2.IMREAD_GRAYSCALE)
            if mask_gt is None:
                continue

            gt        = mask_gt.astype(np.int32)
            mask_stem = os.path.splitext(mask_name)[0]
            t1ce_file = slice_to_t1ce.get(mask_stem)

            if t1ce_file is None:
                skipped += 1
                continue

            pred = self.predictor.predict(
                os.path.join(images_dir, t1ce_file)).astype(np.int32)

            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8),
                                  (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)

            y_true_all.extend(gt.flatten())
            y_pred_all.extend(pred.flatten())

            for c in range(num_classes):
                class_dice[c].append(self._dice((gt == c).astype(np.float32),
                                                 (pred == c).astype(np.float32)))
                class_iou[c].append(self._iou((gt == c).astype(np.float32),
                                               (pred == c).astype(np.float32)))

        if skipped:
            print(f"[SegmentationEvaluator] multiclass: {skipped} masks had no matching t1ce")

        per_class = {name: {"dice": float(np.mean(class_dice[c])),
                            "iou":  float(np.mean(class_iou[c]))}
                     for c, name in enumerate(Config.CLASS_NAMES)}

        fg_dice = [float(np.mean(class_dice[c])) for c in range(1, num_classes)]
        fg_iou  = [float(np.mean(class_iou[c]))  for c in range(1, num_classes)]

        return {
            "mode":      "multiclass",
            "accuracy":  accuracy_score(y_true_all, y_pred_all),
            "precision": precision_score(y_true_all, y_pred_all,
                                         average="macro", zero_division=0),
            "recall":    recall_score(y_true_all, y_pred_all,
                                      average="macro", zero_division=0),
            "f1_score":  f1_score(y_true_all, y_pred_all,
                                  average="macro", zero_division=0),
            "mean_dice": float(np.mean(fg_dice)),
            "mean_iou":  float(np.mean(fg_iou)),
            "per_class": per_class,
        }