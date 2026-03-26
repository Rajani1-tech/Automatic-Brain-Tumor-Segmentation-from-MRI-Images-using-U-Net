# src/training/trainer.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from configs.config import Config
from src.datasets.dataset import BrainTumorDataset
from src.training.losses import bce_dice_loss, dice_coef, iou_coef


class Trainer:
    def __init__(self, model,
                 train_dir=Config.TRAIN_IMAGES_DIR,
                 train_mask=Config.TRAIN_MASKS_DIR,
                 val_dir=Config.VAL_IMAGES_DIR,
                 val_mask=Config.VAL_MASKS_DIR,
                 augment=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        train_dataset = BrainTumorDataset(train_dir, train_mask, augment=augment)
        val_dataset = BrainTumorDataset(val_dir, val_mask, augment=False)

        self.train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                       shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                     shuffle=False, num_workers=2, pin_memory=True)

    def train(self, epochs=Config.EPOCHS, save_path=Config.MODEL_SAVE_PATH):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        history = {"loss": [], "val_loss": [], "dice": [], "val_dice": [], "iou": [], "val_iou": []}

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss, train_dice, train_iou = 0.0, 0.0, 0.0

            for imgs, masks in self.train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss = bce_dice_loss(masks, preds)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_dice += dice_coef(masks, preds).item()
                train_iou += iou_coef(masks, preds).item()

            n = len(self.train_loader)
            train_loss /= n
            train_dice /= n
            train_iou /= n

            # --- Validation ---
            self.model.eval()
            val_loss, val_dice, val_iou = 0.0, 0.0, 0.0

            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    preds = self.model(imgs)
                    val_loss += bce_dice_loss(masks, preds).item()
                    val_dice += dice_coef(masks, preds).item()
                    val_iou += iou_coef(masks, preds).item()

            nv = len(self.val_loader)
            val_loss /= nv
            val_dice /= nv
            val_iou /= nv

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            history["iou"].append(train_iou)
            history["val_iou"].append(val_iou)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"loss: {train_loss:.4f} | dice: {train_dice:.4f} | iou: {train_iou:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_dice: {val_dice:.4f} | val_iou: {val_iou:.4f}")

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved best model to {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history