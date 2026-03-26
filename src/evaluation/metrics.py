# src/evaluation/metrics.py
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from configs.config import Config
from src.inference.predict import Predictor


class Evaluator:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def _dice_coefficient(self, y_true, y_pred):
        smooth = 1.0
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def _iou_score(self, y_true, y_pred):
        smooth = 1.0
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    def evaluate_dataset(self, images_dir, masks_dir):
        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))

        y_true_all, y_pred_all = [], []
        dice_scores, iou_scores = [], []

        for img_name, mask_name in zip(images_list, masks_list):
            mask_gt = cv2.imread(os.path.join(masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            if mask_gt is None:
                continue

            mask_pred = self.predictor.predict(os.path.join(images_dir, img_name))

            if mask_pred.shape != mask_gt.shape:
                mask_pred = cv2.resize(mask_pred, (mask_gt.shape[1], mask_gt.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            mask_gt_bin = (mask_gt > 127).astype(np.int32)
            mask_pred_bin = (mask_pred > 127).astype(np.int32)

            y_true_all.extend(mask_gt_bin.flatten())
            y_pred_all.extend(mask_pred_bin.flatten())

            dice_scores.append(self._dice_coefficient(mask_gt_bin, mask_pred_bin))
            iou_scores.append(self._iou_score(mask_gt_bin, mask_pred_bin))

        return {
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
            "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
            "f1_score": f1_score(y_true_all, y_pred_all, zero_division=0),
            "dice_coefficient": np.mean(dice_scores),
            "iou": np.mean(iou_scores)
        }