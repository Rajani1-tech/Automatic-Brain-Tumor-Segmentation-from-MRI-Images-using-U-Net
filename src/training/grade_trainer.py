# src/training/grade_trainer.py
"""
LGG / HGG Grade Classifier — Dataset + Trainer
───────────────────────────────────────────────
Reads from:
  dataset/classifier/train/LGG/*.png
  dataset/classifier/train/HGG/*.png
  dataset/classifier/val/LGG/*.png
  dataset/classifier/val/HGG/*.png

Each PNG is a T1CE grayscale slice extracted from BraTS2020 h5 files.
Label: LGG=0, HGG=1
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from configs.config import Config



class GradeDataset(Dataset):
    """
    Folder-based binary dataset.
    Expects:
      root/LGG/*.png  → label 0
      root/HGG/*.png  → label 1
    """

    def __init__(self, root_dir, augment=False):
        self.samples  = []
        self.augment  = augment

        for label, grade in enumerate(["LGG", "HGG"]):
            folder = os.path.join(root_dir, grade)
            if not os.path.exists(folder):
                print(f"[GradeDataset] WARNING: {folder} not found — skipping")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(".png"):
                    self.samples.append(
                        (os.path.join(folder, fname), label))

        print(f"[GradeDataset] {root_dir}: "
              f"LGG={sum(1 for _,l in self.samples if l==0)} | "
              f"HGG={sum(1 for _,l in self.samples if l==1)} | "
              f"total={len(self.samples)}")

        # Augmentation transforms (training only)
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        tensor = self.transform(img)        # (1, H, W) float32 in [0,1]
        return tensor, torch.tensor(label, dtype=torch.long)


class GradeTrainer:
    """
    Trains the GradeClassifier for LGG vs HGG binary classification.

    Handles class imbalance automatically via weighted CrossEntropy.
    """

    def __init__(self, model, augment=True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.augment = augment

    def train(self,
              train_dir=Config.GRADE_TRAIN_DIR,
              val_dir=Config.GRADE_VAL_DIR,
              epochs=Config.GRADE_EPOCHS,
              lr=Config.GRADE_LEARNING_RATE,
              batch_size=Config.GRADE_BATCH_SIZE,
              save_path=Config.GRADE_PATH):

        train_ds = GradeDataset(train_dir, augment=self.augment)
        val_ds   = GradeDataset(val_dir,   augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

        # Weighted loss to handle any remaining imbalance
        n_lgg  = sum(1 for _, l in train_ds.samples if l == 0)
        n_hgg  = sum(1 for _, l in train_ds.samples if l == 1)
        total  = n_lgg + n_hgg
        weights = torch.tensor(
            [total / (2 * n_lgg), total / (2 * n_hgg)],
            dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss":   [], "val_acc":   []}
        best_val_acc = 0.0

        print(f"\nTraining GradeClassifier on {self.device}")
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
        print(f"Class weights — LGG: {weights[0]:.3f} | HGG: {weights[1]:.3f}")

        for epoch in range(1, epochs + 1):

            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                t_loss    += loss.item() * imgs.size(0)
                preds      = logits.argmax(dim=1)
                t_correct += (preds == labels).sum().item()
                t_total   += imgs.size(0)

            self.model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    logits = self.model(imgs)
                    loss   = criterion(logits, labels)
                    v_loss    += loss.item() * imgs.size(0)
                    preds      = logits.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total   += imgs.size(0)

            t_loss /= t_total
            v_loss /= v_total
            t_acc   = t_correct / t_total
            v_acc   = v_correct / v_total

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            scheduler.step()

            # Save best model
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(self.model.state_dict(), save_path)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
                      f"val loss {v_loss:.4f} acc {v_acc:.4f}"
                      + (" ✅ best" if v_acc == best_val_acc else ""))

        print(f"\nBest val accuracy: {best_val_acc:.4f}")
        print(f"Model saved → {save_path}")
        return history
