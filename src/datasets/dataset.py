# src/datasets/brain_tumor_dataset.py
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from configs.config import Config
import albumentations as A


class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir,
                 img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                 augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment
        self.images_list = sorted(os.listdir(images_dir))
        self.masks_list = sorted(os.listdir(masks_dir))

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
            ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images_list[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_list[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
        mask = cv2.resize(mask, self.img_size).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # PyTorch expects (C, H, W)
        img = torch.from_numpy(img).unsqueeze(0)    # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img, mask