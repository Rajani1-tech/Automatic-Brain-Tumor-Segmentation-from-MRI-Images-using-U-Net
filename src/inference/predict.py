# src/inference/predictor.py
import cv2
import numpy as np
from src.models.unet import UNetModel
from configs.config import Config

class Predictor:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH):
        self.model = UNetModel().model
        self.model.load_weights(model_path)

    def predict(self, image_path, threshold=0.5, return_resized=True):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_shape = img.shape
        img_resized = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=(0,-1))
        mask = self.model.predict(img_resized)[0,...,0]
        mask = (mask > threshold).astype(np.uint8)
        if return_resized:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask