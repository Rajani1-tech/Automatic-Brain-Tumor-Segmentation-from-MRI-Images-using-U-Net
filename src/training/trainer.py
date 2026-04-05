import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from configs.config import Config
from src.datasets.dataset import BrainTumorDataset
from src.datasets.classifier_dataset import TumorProfileDataset
from src.training.losses import (
    bce_dice_loss, dice_coef, iou_coef,
    ce_dice_loss, multiclass_dice_coef, multiclass_iou_coef,
    per_class_dice, per_class_iou,
)


class SegmentationTrainer:

    def __init__(self, model, mode="binary",
                 train_dir=Config.TRAIN_IMAGES_DIR,
                 train_mask=Config.TRAIN_MASKS_DIR,
                 val_dir=Config.VAL_IMAGES_DIR,
                 val_mask=Config.VAL_MASKS_DIR,
                 augment=True):

        assert mode in ("binary", "multiclass")
        self.mode         = mode
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._class_names = ["NCR/NET", "Edema", "Enh.Tumor"]

        print(f"[SegmentationTrainer] mode={mode} | device={self.device}")

        self.model = model.to(self.device)

        if mode == "binary":
            self.optimizer = Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                               patience=5, factor=0.5, verbose=True)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-5)
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               T_max=Config.EPOCHS, eta_min=1e-6)

        train_ds = BrainTumorDataset(train_dir, train_mask, augment=augment, mode=mode)
        val_ds   = BrainTumorDataset(val_dir,   val_mask,   augment=False,   mode=mode)

        # shuffle=True for both modes — bg_only=0% means all slices are informative,
        # WeightedRandomSampler not needed (ET present in 72% of slices)
        self.train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                       shuffle=True,  num_workers=2, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                                       shuffle=False, num_workers=2, pin_memory=True)

    def _step(self, imgs, masks):
        imgs  = imgs.to(self.device)
        masks = masks.to(self.device)
        preds = self.model(imgs)

        if self.mode == "binary":
            loss = bce_dice_loss(masks, preds)
            dice = dice_coef(masks, preds)
            iou  = iou_coef(masks, preds)
            return loss, dice, iou, None, None

        loss    = ce_dice_loss(masks, preds)
        soft    = F.softmax(preds, dim=1)
        dice    = multiclass_dice_coef(masks, soft)
        iou     = multiclass_iou_coef(masks, soft)
        pc_dice = per_class_dice(masks, soft)
        pc_iou  = per_class_iou(masks, soft)
        return loss, dice, iou, pc_dice, pc_iou

    def train(self, epochs=Config.EPOCHS, save_path=None):
        if save_path is None:
            save_path = (Config.UNET_BINARY_PATH if self.mode == "binary"
                         else Config.UNET_MULTI_PATH)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        best_val     = float("inf") if self.mode == "binary" else 0.0
        patience_ctr = 0
        patience     = 10 if self.mode == "binary" else 15

        history = {k: [] for k in ("loss", "val_loss", "dice", "val_dice", "iou", "val_iou")}
        if self.mode == "multiclass":
            history["val_pc_dice"] = []
            history["val_pc_iou"]  = []

        for epoch in range(epochs):
            self.model.train()
            t_loss = t_dice = t_iou = 0.0

            for imgs, masks in self.train_loader:
                self.optimizer.zero_grad()
                loss, dice, iou, _, _ = self._step(imgs, masks)
                loss.backward()
                if self.mode == "multiclass":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                t_loss += loss.item()
                t_dice += dice.item()
                t_iou  += iou.item()

            n      = len(self.train_loader)
            t_loss /= n
            t_dice /= n
            t_iou  /= n

            self.model.eval()
            v_loss    = v_dice = v_iou = 0.0
            v_pc_dice = [0.0] * (Config.NUM_CLASSES - 1)
            v_pc_iou  = [0.0] * (Config.NUM_CLASSES - 1)

            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    loss, dice, iou, pc_d, pc_i = self._step(imgs, masks)
                    v_loss += loss.item()
                    v_dice += dice.item()
                    v_iou  += iou.item()
                    if pc_d is not None:
                        for c in range(len(v_pc_dice)):
                            v_pc_dice[c] += pc_d[c]
                            v_pc_iou[c]  += pc_i[c]

            nv        = len(self.val_loader)
            v_loss   /= nv
            v_dice   /= nv
            v_iou    /= nv
            v_pc_dice = [x / nv for x in v_pc_dice]
            v_pc_iou  = [x / nv for x in v_pc_iou]

            if self.mode == "binary":
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            for k, v in zip(
                ("loss", "val_loss", "dice", "val_dice", "iou", "val_iou"),
                (t_loss, v_loss, t_dice, v_dice, t_iou, v_iou),
            ):
                history[k].append(v)

            if self.mode == "multiclass":
                history["val_pc_dice"].append(v_pc_dice)
                history["val_pc_iou"].append(v_pc_iou)

            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"loss {t_loss:.4f}  dice {t_dice:.4f}  iou {t_iou:.4f} | "
                  f"val_loss {v_loss:.4f}  val_dice {v_dice:.4f}  val_iou {v_iou:.4f}")

            if self.mode == "multiclass":
                parts = [f"{n} dice={d:.3f} iou={i:.3f}"
                         for n, d, i in zip(self._class_names, v_pc_dice, v_pc_iou)]
                print("    Per-class → " + " | ".join(parts))

            improved = (v_loss < best_val) if self.mode == "binary" else (v_dice > best_val)
            if improved:
                best_val     = v_loss if self.mode == "binary" else v_dice
                patience_ctr = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved best model → {save_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        return history


class ClassifierTrainer:

    def __init__(self, model,
                 train_images=Config.CLASSIFIER_TRAIN_IMAGES,
                 train_masks=Config.CLASSIFIER_TRAIN_MASKS,
                 val_images=Config.CLASSIFIER_VAL_IMAGES,
                 val_masks=Config.CLASSIFIER_VAL_MASKS,
                 augment=True):

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5,
                                           factor=0.5, verbose=True)

        print(f"[ClassifierTrainer] device={self.device}")

        train_ds = TumorProfileDataset(train_images, train_masks, augment=augment)
        val_ds   = TumorProfileDataset(val_images,   val_masks,   augment=False)

        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                       num_workers=2, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                                       num_workers=2, pin_memory=True)

    def train(self, epochs=30, save_path=Config.CLASSIFIER_PATH):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        best_acc     = 0.0
        patience_ctr = 0
        patience     = 8
        history      = {k: [] for k in ("loss", "val_loss", "acc", "val_acc")}

        for epoch in range(epochs):
            self.model.train()
            t_loss = correct = total = 0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                out  = self.model(imgs)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                t_loss  += loss.item()
                preds    = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            t_loss /= len(self.train_loader)
            t_acc   = correct / total

            self.model.eval()
            v_loss = v_correct = v_total = 0

            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    out        = self.model(imgs)
                    v_loss    += self.criterion(out, labels).item()
                    preds      = out.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total   += labels.size(0)

            v_loss /= len(self.val_loader)
            v_acc   = v_correct / v_total
            self.scheduler.step(v_loss)

            for k, val in zip(("loss", "val_loss", "acc", "val_acc"),
                              (t_loss, v_loss, t_acc, v_acc)):
                history[k].append(val)

            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"loss {t_loss:.4f}  acc {t_acc:.4f} | "
                  f"val_loss {v_loss:.4f}  val_acc {v_acc:.4f}")

            if v_acc > best_acc:
                best_acc     = v_acc
                patience_ctr = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved best classifier → {save_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        return history