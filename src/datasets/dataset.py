import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from configs.config import Config


class BrainTumorDataset(Dataset):

    MOD_DETECT_ORDER = ["t1ce", "t1", "t2", "flair"]
    CHANNEL_ORDER    = ["t1", "t1ce", "t2", "flair"]

    def __init__(self, images_dir, masks_dir,
                 img_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                 augment=False, mode="binary"):

        assert mode in ("binary", "multiclass"), f"Unknown mode: {mode}"

        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size
        self.augment    = augment
        self.mode       = mode

        all_images = sorted(f for f in os.listdir(images_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        all_masks  = sorted(f for f in os.listdir(masks_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))

        n_img  = len(all_images)
        n_mask = len(all_masks)

        if mode == "multiclass" and n_img == 4 * n_mask:
            self.pairs       = self._build_multimodal_pairs(all_images, all_masks)
            self._multimodal = True
        elif n_img == n_mask:
            self.pairs       = list(zip(all_images, all_masks))
            self._multimodal = False
        elif n_img == 4 * n_mask:
            self.pairs       = list(zip(all_images[::4], all_masks))
            self._multimodal = False
        else:
            mask_stems = {os.path.splitext(m)[0]: m for m in all_masks}
            pairs, seen = [], set()
            for img_file in all_images:
                stem = os.path.splitext(img_file)[0]
                for ms, mf in mask_stems.items():
                    if stem.startswith(ms) and ms not in seen:
                        pairs.append((img_file, mf))
                        seen.add(ms)
                        break
            n            = min(n_img, n_mask)
            self.pairs   = pairs if pairs else list(zip(all_images[:n], all_masks[:n]))
            self._multimodal = False

        print(f"[BrainTumorDataset] mode={mode} | images={n_img} | "
              f"masks={n_mask} | pairs={len(self.pairs)}"
              + (" | 4-modality: T1 T1CE T2 FLAIR" if self._multimodal else ""))

        if self.augment:
            if mode == "multiclass":
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                             rotate=(-15, 15), p=0.3),
                ])
            else:
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                             rotate=(-15, 15), p=0.3),
                ])

    def _detect_modality(self, filename):
        stem = os.path.splitext(filename)[0].lower()
        for mod in self.MOD_DETECT_ORDER:
            if f"_{mod}_" in stem:
                return mod
        return None

    def _build_multimodal_pairs(self, all_images, all_masks):
        slice_to_mods = defaultdict(dict)
        for img_file in all_images:
            mod = self._detect_modality(img_file)
            if mod is None:
                continue
            stem      = os.path.splitext(img_file)[0]
            slice_key = stem.replace(f"_{mod}_", "_")
            slice_to_mods[slice_key][mod] = img_file

        pairs, missing = [], 0
        for mask_file in all_masks:
            mask_stem = os.path.splitext(mask_file)[0]
            mods      = slice_to_mods.get(mask_stem, {})
            ordered   = [mods.get(mod) for mod in self.CHANNEL_ORDER]
            if all(f is not None for f in ordered):
                pairs.append((ordered, mask_file))
            else:
                missing += 1

        if missing:
            print(f"[BrainTumorDataset] {missing} slices skipped (incomplete modalities)")

        if pairs:
            print(f"[BrainTumorDataset] Grouping verified (first slice):")
            print(f"  Mask : {pairs[0][1]}")
            for ch, (mod, fname) in enumerate(zip(self.CHANNEL_ORDER, pairs[0][0])):
                print(f"  Ch{ch} {mod:5s}: {fname}")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.resize(img, self.img_size,
                          interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    def _zscore(self, img_stack):
        out = np.zeros_like(img_stack, dtype=np.float32)
        for c in range(img_stack.shape[0]):
            ch    = img_stack[c]
            brain = ch > 0.01
            if brain.sum() > 100:
                out[c] = (ch - ch[brain].mean()) / (ch[brain].std() + 1e-8)
            else:
                out[c] = ch
        return out

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        # data.py already remaps label 4 → 3 during extraction
        # mask values are already in {0, 1, 2, 3} — no remap needed
        return mask

    def __getitem__(self, idx):
        img_files, mask_file = self.pairs[idx]
        mask = self._load_mask(os.path.join(self.masks_dir, mask_file))

        if self.mode == "binary":
            mask = (mask > 0).astype(np.float32)
        else:
            mask = mask.astype(np.int64)

        if self._multimodal:
            img_stack = np.stack([self._load_image(os.path.join(self.images_dir, f))
                                   for f in img_files], axis=0)
            img_stack = self._zscore(img_stack)

            if self.augment:
                aug       = self.transform(image=img_stack.transpose(1, 2, 0),
                                           mask=mask.astype(np.float32))
                img_stack = aug["image"].transpose(2, 0, 1)
                mask      = aug["mask"].astype(np.int64)

            return (torch.from_numpy(img_stack.astype(np.float32)),
                    torch.from_numpy(mask.astype(np.int64)))

        img = self._load_image(os.path.join(self.images_dir, img_files))

        if self.augment:
            aug  = self.transform(image=img, mask=mask.astype(np.float32))
            img  = aug["image"]
            mask = (aug["mask"].astype(np.float32) if self.mode == "binary"
                    else aug["mask"].astype(np.int64))

        img_tensor  = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = (torch.from_numpy(mask).unsqueeze(0) if self.mode == "binary"
                       else torch.from_numpy(mask.astype(np.int64)))

        return img_tensor, mask_tensor