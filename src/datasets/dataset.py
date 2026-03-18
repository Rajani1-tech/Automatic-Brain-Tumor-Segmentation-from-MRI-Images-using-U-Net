# src/datasets/brain_tumor_dataset.py
import os
import numpy as np
import cv2
from keras.utils import Sequence
from configs.config import Config

class BrainTumorDataset(Sequence):
    def __init__(self, images_dir, masks_dir, batch_size=Config.BATCH_SIZE, img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_list = sorted(os.listdir(images_dir))
        self.masks_list = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images_list) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_masks = self.masks_list[idx*self.batch_size:(idx+1)*self.batch_size]

        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 1), dtype=np.float32)

        for i, (img_name, mask_name) in enumerate(zip(batch_images, batch_masks)):
            img = cv2.imread(os.path.join(self.images_dir, img_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, self.img_size) / 255.0
            mask = cv2.resize(mask, self.img_size) / 255.0

            X[i, ..., 0] = img
            Y[i, ..., 0] = mask

        return X, Y