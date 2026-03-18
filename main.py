# main.py
import os
import sys
sys.path.append(os.path.abspath("."))


from src.models.unet import UNetModel
from src.training.trainer import Trainer
from src.inference.predict import Predictor
from src.evaluation.metrics import Evaluator
from src.visualization.visualize import Visualizer

# Build and compile model
unet = UNetModel()
unet.compile()
unet.summary()

# Train
trainer = Trainer(unet)
trainer.train()

# Predict & Visualize
predictor = Predictor()
visualizer = Visualizer(predictor)
visualizer.show_prediction("../data/test/images/test1.png")

# Evaluate
evaluator = Evaluator(predictor)
metrics = evaluator.evaluate_dataset("../data/val/images", "../data/val/masks")
print(metrics)