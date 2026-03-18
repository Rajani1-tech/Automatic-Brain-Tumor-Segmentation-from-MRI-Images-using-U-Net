# src/inference/predictor.py
import cv2
import numpy as np
from src.models.unet import UNetModel
from configs.config import Config

class Predictor:
    def __init__(self, model_path=Config.MODEL_SAVE_PATH):
        self.model = UNetModel().model
        self.model.load_weights(model_path)

    def predict(self, image_path, threshold=0.5):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT)) / 255.0
        img = np.expand_dims(img, axis=(0,-1))
        mask = self.model.predict(img)[0,...,0]
        mask = (mask > threshold).astype(np.uint8) * 255
        return mask