# main.py
import os
import sys
import torch
sys.path.append(os.path.abspath("."))

from configs.config import Config
from src.models.unet import UNetModel
from src.models.attention_unet import AttentionUNetModel
from src.training.trainer import Trainer
from src.inference.predict import Predictor
from src.evaluation.metrics import Evaluator

os.makedirs("models", exist_ok=True)

print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

print("Training U-Net...\n")

unet = UNetModel()
trainer_unet = Trainer(unet, augment=True)
history_unet = trainer_unet.train(save_path="models/unet.pth")

predictor_unet = Predictor("models/unet.pth", model=UNetModel())
evaluator_unet = Evaluator(predictor_unet)
metrics_unet = evaluator_unet.evaluate_dataset(Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR)

print("\nU-Net Results:")
for k, v in metrics_unet.items():
    print(f"  {k}: {v:.4f}")

print("\nTraining Attention U-Net...\n")

att_unet = AttentionUNetModel()
trainer_att = Trainer(att_unet, augment=True)
history_att = trainer_att.train(save_path="models/attention_unet.pth")

predictor_att = Predictor("models/attention_unet.pth", model=AttentionUNetModel())
evaluator_att = Evaluator(predictor_att)
metrics_att = evaluator_att.evaluate_dataset(Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR)

print("\nAttention U-Net Results:")
for k, v in metrics_att.items():
    print(f"  {k}: {v:.4f}")
print("\nModel Comparison:")
print(f"  U-Net          Dice: {metrics_unet['dice_coefficient']:.4f} | IoU: {metrics_unet['iou']:.4f}")
print(f"  Attention U-Net Dice: {metrics_att['dice_coefficient']:.4f} | IoU: {metrics_att['iou']:.4f}")