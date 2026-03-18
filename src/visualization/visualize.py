# src/visualization/visualizer.py
import matplotlib.pyplot as plt
import cv2
from src.inference.predict import Predictor

class Visualizer:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def show_prediction(self, image_path):
        mask = self.predictor.predict(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')

        plt.subplot(1,2,2)
        plt.title("Predicted Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()