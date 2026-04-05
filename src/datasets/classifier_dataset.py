import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from configs.config import Config

MOD_DETECT_ORDER = ["t1ce", "t1", "t2", "flair"]


def _detect_modality(filename):
    stem = os.path.splitext(filename)[0].lower()
    for mod in MOD_DETECT_ORDER:
        if f"_{mod}_" in stem:
            return mod
    return None


def derive_tumor_profile(mask: np.ndarray) -> int:
    unique = set(np.unique(mask).tolist())
    unique.discard(0)
    if not unique:
        return 0
    if unique == {2}:
        return 1
    if 3 not in unique:
        return 2
    return 3


class TumorProfileDataset(Dataset):

    def __init__(self, images_dir: str, masks_dir: str,
                 img_size: tuple = (Config.IMG_HEIGHT, Config.IMG_WIDTH),
                 augment: bool = False):

        self.img_size = img_size
        self.augment  = augment

        all_images = sorted(f for f in os.listdir(images_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        all_masks  = sorted(f for f in os.listdir(masks_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))

        n_img  = len(all_images)
        n_mask = len(all_masks)

        if n_img == n_mask:
            
            img_for_mask = {os.path.splitext(f)[0]: f for f in all_images}
            self.samples = []
            for mask_file in all_masks:
                mask_stem = os.path.splitext(mask_file)[0]
                img_file  = img_for_mask.get(mask_stem)
                if img_file:
                    self.samples.append((os.path.join(images_dir, img_file),
                                         os.path.join(masks_dir,  mask_file)))

        else:
            # 4 images per mask — pick t1ce as representative
            # t1ce chosen because it shows tumor contrast best
            # same modality used in classifier training from data.py HGG/LGG folders
            slice_to_t1ce = {}
            for img_file in all_images:
                mod = _detect_modality(img_file)
                if mod != "t1ce":
                    continue
                stem      = os.path.splitext(img_file)[0]
                slice_key = stem.replace("_t1ce_", "_")
                slice_to_t1ce[slice_key] = img_file

            self.samples = []
            missing = 0
            for mask_file in all_masks:
                mask_stem = os.path.splitext(mask_file)[0]
                img_file  = slice_to_t1ce.get(mask_stem)
                if img_file is None:
                    missing += 1
                    continue
                self.samples.append((os.path.join(images_dir, img_file),
                                     os.path.join(masks_dir,  mask_file)))

            if missing:
                print(f"[TumorProfileDataset] {missing} masks had no matching t1ce image")

        print(f"[TumorProfileDataset] images={n_img} | masks={n_mask} | "
              f"pairs={len(self.samples)} | using t1ce modality")

        self._log_distribution()

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=20, p=0.4),
            ])

    def _log_distribution(self):
        counts = [0] * Config.NUM_TUMOR_CLASSES
        for _, mask_path in self.samples:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            counts[derive_tumor_profile(mask)] += 1
        total = sum(counts)
        print("  [TumorProfileDataset] class distribution:")
        for i, (name, cnt) in enumerate(zip(Config.TUMOR_CLASS_NAMES, counts)):
            pct = cnt / total * 100 if total else 0
            print(f"    {i} — {name:<18}: {cnt:>6}  ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        img   = cv2.resize(img,  self.img_size).astype(np.float32) / 255.0
        mask  = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        label = derive_tumor_profile(mask)

        if self.augment:
            img = self.transform(image=img)["image"]

        return torch.from_numpy(img).unsqueeze(0), torch.tensor(label, dtype=torch.long)