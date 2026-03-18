# src/datasets/brain_tumor_dataset.py
import os
import numpy as np
import cv2
from keras.utils import Sequence
from configs.config import Config
import albumentations as A

class BrainTumorDataset(Sequence):
    def __init__(self, images_dir, masks_dir, batch_size=Config.BATCH_SIZE,
                 img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH), augment=False, **kwargs):
        super().__init__(**kwargs)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
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
        return len(self.images_list) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.masks_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype=np.float32)

        for i, (img_name, mask_name) in enumerate(zip(batch_images, batch_masks)):
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, mask_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
            mask = cv2.resize(mask, self.img_size).astype(np.float32)
            mask = (mask > 127).astype(np.float32)

            if self.augment:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            # Ensure channel dimension for both image and mask
            img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
            mask = np.expand_dims(mask, axis=-1) if mask.ndim == 2 else mask

            X[i] = img
            Y[i] = mask

        return X, Y

    def on_epoch_end(self):
        indices = np.arange(len(self.images_list))
        np.random.shuffle(indices)
        self.images_list = [self.images_list[i] for i in indices]
        self.masks_list = [self.masks_list[i] for i in indices]