# main.py
import os
import sys
sys.path.append(os.path.abspath("."))

from configs.config import Config
from src.models.unet import UNetModel
from src.training.trainer import Trainer
from src.inference.predict import Predictor
from src.evaluation.metrics import Evaluator
from src.visualization.visualize import Visualizer

unet = UNetModel()
unet.compile()
unet.summary()

trainer = Trainer(unet, augment=True)
model_history = trainer.train()

predictor = Predictor(Config.MODEL_SAVE_PATH)
visualizer = Visualizer(predictor)
visualizer.plot_training_history(model_history, save_path="training_loss_accuracy.png")
visualizer.show_prediction(os.path.join(Config.VAL_IMAGES_DIR, "image_001.png"))

evaluator = Evaluator(predictor)
metrics = evaluator.evaluate_dataset(Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR)
print(metrics)