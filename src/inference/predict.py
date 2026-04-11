import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config

MOD_DETECT_ORDER = ["t1ce", "t1", "t2", "flair"]
CHANNEL_ORDER    = ["t1", "t1ce", "t2", "flair"]


def _detect_modality(filename):
    stem = os.path.splitext(filename)[0].lower()
    for mod in MOD_DETECT_ORDER:
        if f"_{mod}_" in stem:
            return mod
    return None


def _zscore(img_stack):
    out = np.zeros_like(img_stack, dtype=np.float32)
    for c in range(img_stack.shape[0]):
        ch    = img_stack[c]
        brain = ch > 0.01
        if brain.sum() > 100:
            out[c] = (ch - ch[brain].mean()) / (ch[brain].std() + 1e-8)
        else:
            out[c] = ch
    return out


def _load_gray(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(img, img_size,
                      interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


class SegmentationPredictor:

    def __init__(self, model, model_path, mode="binary"):
        assert mode in ("binary", "multiclass")
        self.mode   = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()

    def _preprocess_binary(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        original_shape = img.shape
        resized = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        tensor  = (torch.from_numpy(resized.astype(np.float32) / 255.0)
                   .unsqueeze(0).unsqueeze(0).to(self.device))
        return tensor, original_shape

    def _preprocess_multiclass_from_file(self, image_path):
        """
        For training/evaluation: image_path has BraTS filename format
        e.g. BraTS2021_00000_t1ce_000.png
        Derives slice_stem and loads all 4 modalities from same folder.
        """
        image_dir = os.path.dirname(image_path)
        filename  = os.path.basename(image_path)
        mod       = _detect_modality(filename)

        if mod is None:
            raise ValueError(
                f"Cannot detect modality from filename: {filename}\n"
                f"Expected format: {{patient}}_{{modality}}_{{slice}}.png\n"
                f"Use predict_from_arrays() for app/demo inference.")

        stem       = os.path.splitext(filename)[0]
        slice_stem = stem.replace(f"_{mod}_", "_")

        all_files     = os.listdir(image_dir)
        slice_to_mods = defaultdict(dict)
        for f in all_files:
            m = _detect_modality(f)
            if m is None:
                continue
            s         = os.path.splitext(f)[0]
            slice_key = s.replace(f"_{m}_", "_")
            slice_to_mods[slice_key][m] = f

        mods    = slice_to_mods.get(slice_stem, {})
        ordered = [mods.get(m) for m in CHANNEL_ORDER]

        if not all(f is not None for f in ordered):
            raise ValueError(
                f"Could not find all 4 modalities for slice '{slice_stem}'. "
                f"Found: {list(mods.keys())}")

        img_size  = (Config.IMG_WIDTH, Config.IMG_HEIGHT)
        channels  = [_load_gray(os.path.join(image_dir, f), img_size) for f in ordered]
        img_stack = _zscore(np.stack(channels, axis=0))
        original  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).shape
        return torch.from_numpy(img_stack).unsqueeze(0).to(self.device), original

    def _preprocess_multiclass_from_arrays(self, modality_arrays):
        """
        For app/demo inference: accepts a dict of numpy arrays.
        modality_arrays: {"t1": arr, "t1ce": arr, "t2": arr, "flair": arr}
        Each array should be (H, W) uint8 grayscale.
        """
        img_size = (Config.IMG_WIDTH, Config.IMG_HEIGHT)
        channels = []
        for mod in CHANNEL_ORDER:
            arr = modality_arrays[mod]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32) / 255.0
            resized = cv2.resize(arr, img_size, interpolation=cv2.INTER_LINEAR)
            channels.append(resized)
        img_stack = _zscore(np.stack(channels, axis=0))
        return torch.from_numpy(img_stack).unsqueeze(0).to(self.device)

    def predict(self, image_path, threshold=0.5):
        """
        File-based prediction. Works for:
          binary mode    : any image file
          multiclass mode: BraTS-formatted filename only
                           (use predict_from_arrays for app uploads)
        """
        if self.mode == "binary":
            tensor, original_shape = self._preprocess_binary(image_path)
            with torch.no_grad():
                out = self.model(tensor)
            mask = (out[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
            return cv2.resize(mask, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        tensor, original_shape = self._preprocess_multiclass_from_file(image_path)
        with torch.no_grad():
            out = self.model(tensor)
        mask = F.softmax(out, dim=1)[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        return cv2.resize(mask, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def predict_from_arrays(self, modality_arrays, original_shape=None, threshold=0.5):
        """
        Array-based prediction for Streamlit app uploads.
        Accepts any image regardless of filename.

        binary mode:
          modality_arrays: single (H, W) numpy array or {"any_key": arr}
          returns (H, W) uint8 mask with values 0/255

        multiclass mode:
          modality_arrays: {"t1": arr, "t1ce": arr, "t2": arr, "flair": arr}
          returns (H, W) uint8 mask with class indices 0-3
        """
        if self.mode == "binary":
            if isinstance(modality_arrays, dict):
                arr = next(iter(modality_arrays.values()))
            else:
                arr = modality_arrays
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32) / 255.0
            resized = cv2.resize(arr, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            tensor  = (torch.from_numpy(resized)
                       .unsqueeze(0).unsqueeze(0).to(self.device))
            with torch.no_grad():
                out = self.model(tensor)
            mask = (out[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
        else:
            tensor = self._preprocess_multiclass_from_arrays(modality_arrays)
            with torch.no_grad():
                out = self.model(tensor)
            mask = F.softmax(out, dim=1)[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

        if original_shape is not None:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask

    def predict_proba(self, image_path):
        assert self.mode == "multiclass", "predict_proba is for multiclass only"
        tensor, _ = self._preprocess_multiclass_from_file(image_path)
        with torch.no_grad():
            return F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

    def get_attention_map(self, modality_arrays_or_path):
        from src.models.attention_unet import AttentionGate

        att_gates = {name: m for name, m in self.model.named_modules()
                     if isinstance(m, AttentionGate)}
        if not att_gates:
            raise RuntimeError("No AttentionGate modules found.")

        captured = {}

        def make_hook(name):
            def hook(_, __, output):
                captured[name] = output.detach()
            return hook

        hooks = [gate.sigmoid.register_forward_hook(make_hook(n))
                 for n, gate in att_gates.items()]

        if isinstance(modality_arrays_or_path, str):
            if self.mode == "binary":
                tensor, _ = self._preprocess_binary(modality_arrays_or_path)
            else:
                tensor, _ = self._preprocess_multiclass_from_file(
                    modality_arrays_or_path)
        else:
            tensor = self._preprocess_multiclass_from_arrays(
                modality_arrays_or_path)

        try:
            with torch.no_grad():
                self.model(tensor)
        finally:
            for h in hooks:
                h.remove()

        H, W  = Config.IMG_HEIGHT, Config.IMG_WIDTH
        maps  = [F.interpolate(attn, size=(H, W), mode="bilinear",
                               align_corners=True)[0, 0].cpu().numpy()
                 for attn in captured.values()]
        heatmap = np.mean(maps, axis=0)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=8)
        mn, mx  = heatmap.min(), heatmap.max()
        if mx - mn > 1e-8:
            heatmap = (heatmap - mn) / (mx - mn)
        return heatmap.astype(np.float32)


class ClassifierPredictor:

    def __init__(self, model, model_path=Config.CLASSIFIER_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.eval()

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img    = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img    = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        tensor = (torch.from_numpy(img.astype(np.float32) / 255.0)
                  .unsqueeze(0).unsqueeze(0).to(self.device))
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        class_idx = int(probs.argmax())
        return Config.TUMOR_CLASS_NAMES[class_idx], float(probs[class_idx]) * 100, probs

    def predict_from_array(self, img_array):
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32) / 255.0
        resized = cv2.resize(img_array, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        tensor  = (torch.from_numpy(resized)
                   .unsqueeze(0).unsqueeze(0).to(self.device))
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        class_idx = int(probs.argmax())
        return Config.TUMOR_CLASS_NAMES[class_idx], float(probs[class_idx]) * 100, probs