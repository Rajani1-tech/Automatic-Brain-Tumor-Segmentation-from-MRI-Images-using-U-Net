# Automatic Brain Tumor Segmentation from MRI Images using U-Net


## Abstract

Brain tumor detection and localization from magnetic resonance imaging (MRI) scans is a critical task in clinical diagnosis and treatment planning. Manual segmentation by radiologists is time-consuming and prone to inter-observer variability. This project proposes an automated deep learning-based framework for **brain tumor segmentation using convolutional neural networks**. A U-Net architecture is implemented to perform pixel-level segmentation of tumor regions in MRI scans. The model is trained and evaluated using annotated medical imaging datasets, and performance is assessed using standard medical segmentation metrics such as Dice coefficient and Intersection over Union (IoU).

---

## 1. Introduction

Brain tumors are abnormal growths of cells in the brain that can significantly impact neurological function and patient survival. Early detection and accurate localization of tumor regions are essential for effective treatment planning.

Magnetic Resonance Imaging (MRI) is widely used for diagnosing brain tumors because it provides high-resolution anatomical information. However, manual interpretation and segmentation of tumor regions from MRI scans require significant expertise and time.

Recent advances in deep learning have enabled automated medical image analysis systems capable of assisting radiologists in clinical workflows. Convolutional neural networks, particularly encoder–decoder architectures such as U-Net, have demonstrated strong performance in medical image segmentation tasks.

This project explores the application of deep learning techniques for **automated segmentation of brain tumors from MRI images**.

---

## 2. Problem Statement

The goal of this project is to develop a deep learning model that can:

* Accurately identify tumor regions in MRI scans
* Perform **pixel-level segmentation** of brain tumors
* Provide visual outputs that highlight tumor boundaries
* Assist in improving efficiency in medical imaging workflows

---

## 3. Dataset

This study utilizes the **LGG MRI Segmentation Dataset**, which contains annotated MRI images with tumor masks.

Dataset characteristics:

* MRI brain scans
* Pixel-level segmentation masks
* Image format: PNG
* Corresponding mask images for each MRI scan

Dataset format:

```id="kk2ykr"
dataset/
   images/
      image_001.png
      image_002.png

   masks/
      image_001_mask.png
      image_002_mask.png
```

Mask labels:

| Pixel Value | Meaning      |
| ----------- | ------------ |
| 0           | Background   |
| 255         | Tumor Region |

---

## 4. Methodology

### 4.1 Data Preprocessing

The following preprocessing steps are applied:

* Image resizing to **256 × 256 resolution**
* Pixel intensity normalization
* Binary conversion of segmentation masks
* Dataset splitting into training, validation, and test sets

### 4.2 Data Augmentation

To improve model robustness and reduce overfitting, several augmentation techniques are applied:

* Horizontal flipping
* Random rotation
* Contrast adjustments
* Random brightness transformations

### 4.3 Model Architecture

The segmentation model is based on the **U-Net architecture**, a widely used convolutional neural network designed for biomedical image segmentation.

The architecture consists of:

**Encoder (Contracting Path)**
Extracts hierarchical spatial features using convolution and pooling layers.

**Bottleneck Layer**
Captures deep contextual representations of the input image.

**Decoder (Expanding Path)**
Upsamples feature maps to reconstruct the segmentation mask.

Skip connections between encoder and decoder layers help preserve spatial information and improve segmentation accuracy.

---

## 5. Training Configuration

| Parameter        | Value     |
| ---------------- | --------- |
| Model            | U-Net     |
| Input Resolution | 256 × 256 |
| Optimizer        | Adam      |
| Learning Rate    | 0.0001    |
| Batch Size       | 8         |
| Epochs           | 40        |

Loss Functions:

* Dice Loss
* Binary Cross Entropy Loss

These losses are commonly used in medical image segmentation to handle class imbalance and improve overlap accuracy.

---

## 6. Evaluation Metrics

The segmentation model is evaluated using the following metrics:

**Dice Coefficient**

Measures the overlap between predicted segmentation and ground truth.

**Intersection over Union (IoU)**

Evaluates the intersection area between prediction and ground truth masks.

**Precision**

Measures the proportion of correctly predicted tumor pixels.

**Recall**

Measures the ability of the model to detect tumor regions.

---

## 7. Experimental Results

The model outputs include:

* Original MRI image
* Ground truth tumor segmentation mask
* Predicted segmentation mask
* Overlay visualization highlighting tumor regions

These visual outputs enable qualitative evaluation of model performance.

Example output pipeline:

MRI Image → Segmentation Model → Predicted Tumor Mask → Visualization Overlay

---

## 8. Applications

The proposed system can assist in:

* Computer-aided diagnosis
* Radiology workflow automation
* Medical image analysis research
* Clinical decision support systems

---

## 9. Limitations

Some limitations of the current approach include:

* Dependence on dataset size
* Variability in MRI acquisition parameters
* Limited generalization across different hospitals or scanners

Future work can address these limitations through larger datasets and advanced architectures.

---

## 10. Future Work

Potential improvements include:

* Implementing **Attention U-Net**
* Exploring **3D CNN-based segmentation models**
* Multi-class segmentation of tumor subregions
* Integration of **Explainable AI techniques** for interpretability

---

## 11. Project Structure

```id="kjf94a"
Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net
.
├── brain_seg_env
├── configs
│   ├── config.py
│   └── __pycache__
├── data
│   ├── test
│   ├── train
│   └── val
├── LICENSE
├── main.py
├── notebook
│   └── data_processing.ipynb
├
├── README.md
├── requirements.txt
└── src
    ├── datasets
    ├── evaluation
    ├── inference
    ├── models
    ├── __pycache__
    ├── training
    └── visualization


```

---

## 12. Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Albumentations
* segmentation-models-pytorch

---

## Author

Rajani Lamichhane
Machine Learning Engineer | Computer Vision | Biomedical AI

