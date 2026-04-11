# 🧠 Automatic Brain Tumor Segmentation from MRI Images

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![BraTS2021](https://img.shields.io/badge/Dataset-BraTS%202021-blue)](https://www.synapse.org/brats2021)
[![Mean Dice](https://img.shields.io/badge/Mean%20Dice-0.828-brightgreen)]()
[![ET Dice](https://img.shields.io/badge/ET%20Dice-0.860-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An End-to-End Deep Learning Pipeline for Multiclass Brain Tumor Segmentation and Grade Classification from Multi-Modal MRI**

*Rajani Lamichhane · Machine Learning Engineer · Computer Vision · Biomedical AI*

</div>

---

## 📋 Overview

This project presents a fully automated framework for **multiclass brain tumor segmentation** and **LGG/HGG grade prediction** from multi-modal MRI scans using the BraTS 2021 dataset.

```
4-Modality MRI Input (T1, T1CE, T2, FLAIR)
            ↓
  Attention U-Net Multiclass Segmentation
            ↓
 Region Analysis (NCR/NET · Edema · Enhancing Tumor)
            ↓
     Tumor Profile Classification + LGG/HGG Grade Prediction
            ↓
       Streamlit Clinical Web Interface
```

| Metric | Value |
|--------|-------|
| Mean Dice | **0.828** |
| Enhancing Tumor Dice | **0.860** |
| NCR/NET Dice | **0.827** |
| Edema Dice | **0.798** |

> 🏆 ET Dice **0.860** surpasses 3D ResU-Net (0.800) — at significantly lower computational cost.

---

## 🔬 Research Highlights

**1. Critical Data Pipeline Bug → +0.654 Mean Dice with zero architectural changes** (see [§ Bug Discovery](#-critical-bug-discovery))

**2. Attention gates provide no measurable benefit** (Δ Mean Dice: 0.001) when multi-modal input already provides strong spatial discriminative signal via T1CE.

**3. Correct 2D multi-modal assembly matches 3D architecture performance** — data quality matters more than model complexity.

---

## 📊 Results

### Model Comparison

| Model | Mean Dice | Mean IoU | Accuracy |
|-------|-----------|----------|----------|
| U-Net Binary | 0.146 | 0.096 | 0.941 |
| Attn U-Net Binary | 0.141 | 0.092 | 0.944 |
| U-Net Multiclass | **0.828** | **0.763** | 0.993 |
| Attn U-Net Multiclass | **0.828** | **0.763** | **0.993** |

### vs Published Baselines (BraTS)

| Method | NCR/NET | Edema | ET | Mean Dice |
|--------|---------|-------|----|-----------|
| Standard 2D U-Net (lit.) | 0.550 | 0.720 | 0.670 | 0.650 |
| 3D ResU-Net (Myronenko 2018) | 0.810 | 0.840 | 0.800 | 0.820 |
| **This work (2D U-Net)** | **0.827** | 0.798 | **0.860** | **0.828** |
| **This work (Attn U-Net)** | 0.820 | **0.803** | 0.859 | 0.828 |

---

## 🐛 Critical Bug Discovery

> **A modality grouping bug caused ET Dice = 0.000 for all training epochs. Fixed with no model changes.**

### The Bug

The original loader naively sliced files by alphabetical index:

```python
# WRONG: groups 4 consecutive FLAIR slices, not 4 modalities
groups = all_images[i*4 : i*4+4]
```

Alphabetical sorting orders files **by modality name first**, so `all_images[0:4]` yielded `[flair_000, flair_001, flair_002, flair_003]` — **T1CE was never included in any training batch.**

Since Enhancing Tumor is only visible on T1CE, the model had no signal to learn ET boundaries → ET Dice = 0.000.

A secondary bug: `"t1"` is a substring of `"t1ce"`, so naive string matching misclassifies T1CE files as T1. Fix: check `"t1ce"` before `"t1"`.

### The Fix: Stem-Based Modality Matching

```python
# For each mask "BraTS2021_00000_000.png", find the matching slice per modality:
#   BraTS2021_00000_t1_000.png    → Ch0 (T1)
#   BraTS2021_00000_t1ce_000.png  → Ch1 (T1CE) ← ET signal restored
#   BraTS2021_00000_t2_000.png    → Ch2 (T2)
#   BraTS2021_00000_flair_000.png → Ch3 (FLAIR)

slice_key = stem.replace(f"_{mod}_", "_")
slice_to_mods[slice_key][mod] = img_file
```

### Impact

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| ET Dice | 0.000 🔴 | **0.860** 🟢 | +0.860 |
| NCR/NET Dice | 0.170 🔴 | **0.827** 🟢 | +0.657 |
| Edema Dice | 0.004 🔴 | **0.798** 🟢 | +0.794 |
| Mean Dice | 0.174 🔴 | **0.828** 🟢 | +0.654 |

---

## 🏗️ Architecture

**Attention U-Net** — Input: `(B, 4, 240, 240)` → Output: `(B, 4, 240, 240)` 4-class probability map

- Encoder: 4→64→128→256→512 with MaxPool 2×2
- Decoder: Upsample + Attention Gate + skip concatenation at each level
- Output: 1×1 Conv → 4 classes (Background, NCR/NET, Edema, ET)

**Tumor Profile Classifier** — CNN predicting severity (No Tumor / Edema Only / Core Present / Full Tumor) from T1CE slice.

**LGG/HGG Grade Classifier** — ResNet-18 fine-tuned for binary grade prediction; first conv adapted from 3→1 channel for grayscale T1CE input.

---

## 📁 Dataset

**BraTS 2021** — 4 MRI modalities per patient, PNG format, 240×240, 4 mask classes.

| Modality | Role |
|----------|------|
| T1 | Anatomical reference |
| **T1CE** | **Enhancing Tumor** — bright contrast enhancement |
| T2 | Edema, fluid content |
| FLAIR | Peritumoral edema, suppresses CSF |

**BraTS 2021** (grade classification) — 75,487 slices total (43% LGG / 57% HGG).

> Raw BraTS ET label `4` is remapped to `3` during preprocessing.

---

## ⚙️ Training

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Epochs | 75 (segmentation) · 30 (classifiers) |
| Optimizer | Adam, lr=1×10⁻⁴ |
| LR Scheduler | StepLR (step=10, γ=0.5) |
| Loss | Dice Loss + Cross-Entropy |
| Augmentation | Flip, Rotate, Brightness/Contrast, Affine, Gaussian Noise |

Preprocessing: Z-score normalization on brain-masked pixels per channel; BraTS label 4→3 remapping.

---

## 🚀 Getting Started

```bash
git clone https://github.com/rajanilamichhane/Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net.git
cd Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net
pip install -r requirements.txt

# Train
python main.py --mode train --model attn_unet_multiclass

# Evaluate
python main.py --mode eval --model attn_unet_multiclass --checkpoint path/to/checkpoint.pth

# Web app
streamlit run app.py
```

Upload files using the naming convention: `BraTS2021_XXXXX_t1_YYY.png`, `_t1ce_`, `_t2_`, `_flair_`. Use middle slices (035–065) for best visibility.

---

## 📂 Project Structure

```
├── configs/config.py
├── data/train · val · test
├── src/
│   ├── datasets/      # Stem-based modality matching loader
│   ├── models/        # U-Net, Attention U-Net, ResNet-18
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── visualization/
├── notebook/data_processing.ipynb
├── app.py             # Streamlit interface
└── main.py
```

---

## ⚠️ Limitations & Future Work

**Limitations:** 2D slice processing loses volumetric context · mild overfitting in late epochs · PNG vs NIfTI distribution shift for clinical deployment.

**Future:** 3D volumetric segmentation · TransUNet / Swin-UNet · GradCAM explainability · multi-scanner domain adaptation · Docker deployment.

---

## 📄 References

1. Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
2. Oktay et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL.
3. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Myronenko (2018). *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization.* BraTS @ MICCAI.
5. Baid et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark.* arXiv:2107.02314.

---

## 👤 Author

**Rajani Lamichhane** · Machine Learning Engineer · Computer Vision · Biomedical AI

<div align="center"><br>
<i>If this project helped your research, consider giving it a ⭐</i>
</div>