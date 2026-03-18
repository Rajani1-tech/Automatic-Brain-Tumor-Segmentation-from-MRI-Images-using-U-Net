# src/evaluation/evaluator.py
import os
import cv2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from configs.config import Config
from src.inference.predict import Predictor

class Evaluator:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def evaluate_dataset(self, images_dir, masks_dir):
        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))

        y_true, y_pred = [], []

        for img_name, mask_name in zip(images_list, masks_list):
            mask_gt = cv2.imread(os.path.join(masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            mask_gt = cv2.resize(mask_gt, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            mask_gt = (mask_gt > 127).astype(int)

            mask_pred = self.predictor.predict(os.path.join(images_dir, img_name))
            mask_pred = (mask_pred > 127).astype(int)

            y_true.extend(mask_gt.flatten())
            y_pred.extend(mask_pred.flatten())

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }