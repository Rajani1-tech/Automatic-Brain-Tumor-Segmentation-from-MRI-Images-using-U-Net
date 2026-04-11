# Automatic Brain Tumor Segmentation from MRI Images

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%202021-blue)](https://www.synapse.org/brats2021)
[![Mean Dice](https://img.shields.io/badge/Mean%20Dice-0.828-brightgreen)]()
[![ET Dice](https://img.shields.io/badge/ET%20Dice-0.860-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Report](https://img.shields.io/badge/Report-PDF-orange?logo=adobeacrobatreader&logoColor=white)](docs/brain_tumour_seg_documents.pdf)

<br>

**An end-to-end deep learning pipeline for multiclass brain tumor segmentation and grade classification from multi-modal MRI — achieving Mean Dice 0.828 and ET Dice 0.860, competitive with published 3D architectures.**

<br>

*Rajani Lamichhane · Machine Learning Engineer · Computer Vision · Biomedical AI*

<br>

📄 [Read the full technical report (PDF)](docs/brain_tumour_seg_documents.pdf) — architecture details, training curves, bug analysis, and extended results.

</div>

---

## Abstract

Brain tumor segmentation from MRI is a critical yet time-consuming clinical task, prone to significant inter-observer variability. This project presents a fully automated pipeline that performs **pixel-level multiclass segmentation** of three clinically meaningful tumor subregions — Necrotic Core/Non-Enhancing Tumor (NCR/NET), Peritumoral Edema, and Enhancing Tumor (ET) — followed by tumor severity classification and LGG/HGG grade prediction, all from four-modality MRI input.

A 2D Attention U-Net trained on BraTS 2021 achieves **Mean Dice 0.828** and **ET Dice 0.860**, surpassing published 3D architectures at significantly lower computational cost. A critical data pipeline bug discovered during development — where naive alphabetical file slicing excluded T1CE from all training batches — is thoroughly documented. Fixing it improved ET Dice from **0.000 to 0.860** without any architectural changes, demonstrating that rigorous data engineering can matter more than model sophistication.

---

## 1. Introduction

Brain tumors present one of the most challenging diagnostic scenarios in clinical medicine. MRI is the gold standard imaging modality for tumor detection, providing high-resolution anatomical and functional information across multiple sequences. However, manual delineation of tumor subregions by radiologists is time-intensive and subject to inter-observer variability — making automated, reproducible segmentation a high-value clinical problem.

Recent advances in deep learning, particularly encoder–decoder architectures like U-Net, have demonstrated strong performance in medical image segmentation. This project builds on that foundation with several key contributions:

- A complete multiclass segmentation pipeline distinguishing all three BraTS tumor subregions
- A stem-based modality-matching data loader that guarantees correct 4-channel assembly
- A documented critical bug fix (ET Dice: 0.000 → 0.860) with analysis of root cause and impact
- An LGG/HGG grade classifier using fine-tuned ResNet-18
- A clinical-grade Streamlit interface for end-to-end MRI-to-prediction inference

---

## 2. Problem Statement

The objective is to develop a deep learning system that can:

- Perform **pixel-level multiclass segmentation** of three tumor subregions from four-modality MRI
- Classify **tumor severity** (No Tumor / Edema Only / Core Present / Full Tumor)
- Predict **tumor grade** (LGG vs. HGG) to assist in treatment planning
- Provide interpretable visualizations suitable for clinical or research review

---

## 3. Dataset

All models are trained and evaluated on **BraTS 2021**, the benchmark dataset for brain tumor segmentation.

| Property | Value |
|----------|-------|
| Modalities | T1, T1CE, T2, FLAIR |
| Format | PNG (converted from NIfTI) |
| Resolution | 240 × 240 |
| Segmentation classes | Background · NCR/NET · Edema · ET |
| Grade split | 43% LGG · 57% HGG |
| Total slices | 75,487 |

Each MRI modality carries distinct clinical information:

| Modality | Clinical Role |
|----------|--------------|
| T1 | Anatomical reference |
| **T1CE** | **Enhancing Tumor** — gadolinium contrast, critical for ET detection |
| T2 | Edema and fluid content |
| FLAIR | Peritumoral edema with CSF suppression |

> Raw BraTS label `4` (Enhancing Tumor) is remapped to `3` during preprocessing to produce contiguous class indices.

---

## 4. Methodology

### 4.1 Pipeline Overview

The system follows a sequential analysis pipeline:

```
Four-Modality MRI Input (T1, T1CE, T2, FLAIR)
        │
        ▼
  Preprocessing & Augmentation
        │
        ▼
  Multiclass Segmentation (Attention U-Net)
    ├── NCR/NET · Edema · ET masks
        │
        ▼
  Tumor Severity Classification (CNN)
    ├── No Tumor / Edema Only / Core Present / Full Tumor
        │
        ▼
  Grade Prediction (ResNet-18 fine-tuned)
    └── LGG vs. HGG
```

### 4.2 Preprocessing

- **Normalization:** Z-score normalization per channel over brain-masked pixels (threshold > 0.1 after /255 scaling)
- **Resizing:** 240 × 240 (native BraTS resolution preserved)
- **Label remapping:** BraTS class 4 → 3 for contiguous indexing

### 4.3 Data Augmentation

All spatial augmentations are applied identically across all 4 input channels and the segmentation mask to preserve modality-to-mask alignment:

- Horizontal and vertical flipping
- Random 90° rotation
- Brightness and contrast jitter
- Gaussian noise injection
- Affine transforms

### 4.4 Model Architectures

**Segmentation — Attention U-Net**

The primary model is a 2D Attention U-Net with the following structure:

- **Encoder (Contracting Path):** Hierarchical feature extraction via convolution and max-pooling blocks
- **Bottleneck:** Deep contextual representation at compressed spatial resolution
- **Decoder (Expanding Path):** Progressive upsampling with transposed convolutions
- **Skip Connections:** Preserve fine-grained spatial detail between encoder and decoder
- **Attention Gates:** Suppress irrelevant activations at each skip connection, guiding the decoder to focus on tumor-discriminative regions

**Tumor Profile Classifier — CNN**

A lightweight CNN that accepts a single T1CE slice and predicts tumor severity across four categories: No Tumor, Edema Only, Core Present, Full Tumor. Trained with weighted cross-entropy to handle class imbalance.

**Grade Classifier — ResNet-18**

ResNet-18 fine-tuned for binary LGG/HGG classification. The first convolutional layer is adapted from 3-channel RGB input to 1-channel grayscale T1CE input, with all other pretrained weights retained.

### 4.5 Training Configuration

| Parameter | Segmentation | Classifiers |
|-----------|:------------:|:-----------:|
| Epochs | 75 | 30 |
| Batch size | 16 | 16 |
| Optimizer | Adam | Adam |
| Learning rate | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| LR scheduler | StepLR (step=10, γ=0.5) | StepLR |
| Loss function | Dice + Cross-Entropy | Weighted Cross-Entropy |

---

## 5. Critical Finding — Data Pipeline Bug

> **ET Dice was exactly 0.000 across all training epochs. The complete improvement to 0.860 came from a single-line fix in the data loader — no architectural changes, no hyperparameter tuning.**

This finding is the most instructive outcome of the project, and is documented in full.

### Root Cause

The original data loader assembled 4-channel samples by naive alphabetical index slicing:

```python
# WRONG — alphabetical order groups by modality name, not by slice
groups = all_images[i*4 : i*4+4]
# Result: [flair_000, flair_001, flair_002, flair_003]
# T1CE is never included in any training batch
```

Alphabetical sorting places all FLAIR slices before T1 and T1CE. Every training sample therefore received four consecutive FLAIR slices — never the four required modalities. Since Enhancing Tumor is only visible on T1CE (gadolinium-enhanced), the model received zero signal to learn ET boundaries.

A secondary bug compounded the problem: `"t1"` is a substring of `"t1ce"`, causing naive string matching to misclassify T1CE files as T1 when checking modality names.

### Fix — Stem-Based Modality Matching

```python
# For each mask file, e.g. "BraTS2021_00000_000.png",
# locate the correct modality image for that exact slice:

slice_key = stem.replace(f"_{mod}_", "_")
slice_to_mods[slice_key][mod] = img_file

# Guarantees correct 4-channel assembly per slice:
#   BraTS2021_00000_t1_000.png    → Ch0  T1
#   BraTS2021_00000_t1ce_000.png  → Ch1  T1CE  ← ET signal restored
#   BraTS2021_00000_t2_000.png    → Ch2  T2
#   BraTS2021_00000_flair_000.png → Ch3  FLAIR
```

### Impact

| Metric | Before Fix | After Fix | Δ |
|--------|:----------:|:---------:|:-:|
| Enhancing Tumor (ET) Dice | 0.000 | **0.860** | +0.860 |
| NCR/NET Dice | 0.170 | **0.827** | +0.657 |
| Edema Dice | 0.004 | **0.798** | +0.794 |
| **Mean Dice** | 0.174 | **0.828** | **+0.654** |

---

## 6. Results

### 6.1 Model Comparison

| Model | Mean Dice | Mean IoU | Accuracy |
|-------|:---------:|:--------:|:--------:|
| U-Net Binary | 0.146 | 0.096 | 0.941 |
| Attention U-Net Binary | 0.141 | 0.092 | 0.944 |
| U-Net Multiclass | **0.828** | **0.763** | 0.993 |
| Attention U-Net Multiclass | **0.828** | **0.763** | **0.993** |

### 6.2 Comparison with Published Baselines

| Method | NCR/NET | Edema | ET | Mean Dice |
|--------|:-------:|:-----:|:--:|:---------:|
| Standard 2D U-Net (literature) | 0.550 | 0.720 | 0.670 | 0.650 |
| 3D ResU-Net — Myronenko (2018) | 0.810 | 0.840 | 0.800 | 0.820 |
| **This work — 2D U-Net** | **0.827** | 0.798 | **0.860** | **0.828** |
| **This work — Attention U-Net** | 0.820 | **0.803** | 0.859 | **0.828** |

**Key observations:**
- A correctly assembled 2D model surpasses 3D ResU-Net on ET Dice (0.860 vs. 0.800)
- Attention gates provided no measurable benefit over plain U-Net (Δ Mean Dice = 0.001), suggesting T1CE already provides sufficient spatial discriminative signal for ET localization without additional gating
- The dominant factor in performance was data pipeline correctness, not architectural choice

---

## 7. Web Application

A Streamlit interface provides real-time end-to-end analysis from raw MRI upload to grade prediction.

**Features:**
- Color-coded segmentation overlay with per-region pixel statistics
- Tumor severity profile classification
- LGG/HGG grade prediction with clinical reasoning
- Attention heatmap visualization

```bash
streamlit run app.py
```

Upload files following BraTS naming convention — `BraTS2021_XXXXX_t1_YYY.png`, `_t1ce_`, `_t2_`, `_flair_`. Middle slices (035–065) show the most complete tumor core.

---

## 8. Limitations

- **2D processing:** Slice-level inference loses volumetric context across adjacent slices; a 3D model would better capture tumor geometry
- **Overfitting:** Training and validation Dice diverge in later epochs; additional regularization (dropout, weight decay) is warranted
- **Distribution shift:** PNG-converted slices may differ from native NIfTI used in clinical workflows, potentially limiting real-world generalization

---

## 9. Future Work

- 3D volumetric segmentation on full NIfTI volumes
- Transformer-based architectures — TransUNet, Swin-UNet
- GradCAM / SHAP explainability per input modality
- Multi-scanner domain adaptation
- Docker containerization for portable clinical deployment

---

## Project Structure

```
├── configs/
│   └── config.py
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   └── report.pdf                   # Full technical report (LaTeX)
├── src/
│   ├── datasets/                    # Stem-based modality-matching loader
│   ├── models/                      # U-Net, Attention U-Net, ResNet-18
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── visualization/
├── notebook/
│   └── data_processing.ipynb
├── app.py                           # Streamlit clinical interface
├── main.py
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/rajanilamichhane/Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net.git
cd Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net
pip install -r requirements.txt
```

```bash
# Train segmentation model
python main.py --mode train --model attn_unet_multiclass

# Evaluate on test set
python main.py --mode eval --model attn_unet_multiclass --checkpoint path/to/checkpoint.pth

# Launch clinical web interface
streamlit run app.py
```

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
2. Oktay, O., et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL.
3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Myronenko, A. (2018). *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization.* BraTS @ MICCAI.
5. Baid, U., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark.* arXiv:2107.02314.
6. Menze, B. H., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark.* IEEE TMI, 34(10).

---

## Author

**Rajani Lamichhane**  
Machine Learning Engineer · Computer Vision · Biomedical AI

---

<div align="center">
<sub>If this work helped your research, please consider giving it a ⭐</sub>
</div>