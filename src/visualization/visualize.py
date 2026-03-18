import matplotlib.pyplot as plt
import cv2

class Visualizer:
    def __init__(self, predictor):
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

    def plot_training_history(self, history, save_path="training_plot.png"):
        hist = history.history

        plt.figure(figsize=(12,8))

        plt.subplot(2,2,1)
        plt.plot(hist['loss'], label='Train Loss')
        plt.plot(hist['val_loss'], label='Val Loss')
        plt.title("Loss")
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot(hist['accuracy'], label='Train Acc')
        plt.plot(hist['val_accuracy'], label='Val Acc')
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot(hist['dice_coef'], label='Train Dice')
        plt.plot(hist['val_dice_coef'], label='Val Dice')
        plt.title("Dice Score")
        plt.legend()

        plt.subplot(2,2,4)
        plt.plot(hist['iou_coef'], label='Train IoU')
        plt.plot(hist['val_iou_coef'], label='Val IoU')
        plt.title("IoU Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()