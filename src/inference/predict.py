# src/inference/predictor.py
import os
import cv2
import numpy as np
import torch
from configs.config import Config
from src.models.unet import UNetModel


class Predictor:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = UNetModel().to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_path, threshold=0.5, return_resized=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        original_shape = img.shape
        img_resized = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT)).astype(np.float32) / 255.0

        tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        with torch.no_grad():
            pred = self.model(tensor)[0, 0].cpu().numpy()  # (H, W)

        mask_pred = (pred > threshold).astype(np.uint8) * 255

        if return_resized:
            mask_pred = cv2.resize(mask_pred, (original_shape[1], original_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        return mask_pred