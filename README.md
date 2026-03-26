
# Automatic Brain Tumor Segmentation from MRI Images using U-Net

## Abstract

Brain tumor detection and localization from magnetic resonance imaging (MRI) scans is a critical task in clinical diagnosis and treatment planning. Manual segmentation by radiologists is time-consuming and prone to inter-observer variability. This project proposes an automated deep learning-based framework for brain tumor segmentation using convolutional neural networks. A U-Net architecture, along with an enhanced Attention U-Net variant, is implemented to perform pixel-level segmentation of tumor regions in MRI scans. The model is trained and evaluated using annotated medical imaging datasets, and performance is assessed using standard metrics such as Dice coefficient and Intersection over Union (IoU).

---

## 1. Introduction

Brain tumors are abnormal growths of cells in the brain that can significantly impact neurological function and patient survival. Early detection and accurate localization of tumor regions are essential for effective treatment planning.

Magnetic Resonance Imaging (MRI) provides high-resolution anatomical information and is widely used for diagnosis. However, manual segmentation is labor-intensive and requires expert knowledge.

Deep learning-based approaches, particularly encoder–decoder architectures such as U-Net, have shown strong performance in medical image segmentation tasks.

---

## 2. Problem Statement

The objective of this project is to develop a deep learning model that can:

* Accurately identify tumor regions in MRI scans
* Perform pixel-level segmentation
* Generate visual outputs highlighting tumor boundaries
* Assist clinical workflows through automation

---

## 3. Dataset

This project uses the BraTS 2020 (Brain Tumor Segmentation) dataset obtained from Kaggle:

🔗 [BraTS2020 Kaggle Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/code?utm_source=chatgpt.com)

### Dataset Description

The BraTS 2020 dataset consists of multi-modal MRI scans collected from multiple institutions and annotated by expert radiologists. It includes different MRI modalities such as T1, T1ce, T2, and FLAIR, along with ground truth segmentation masks for tumor regions. ([遇见数据集][1])

Each MRI volume typically has a resolution of **240 × 240 × 155**, which is commonly resized for 2D model training. 

### Dataset Split (Used in this Project)

| Split      | Number of Images |
| ---------- | ---------------- |
| Train      | 17,026           |
| Validation | 3,666            |
| Test       | 3,730            |

### Image Configuration

* Input image size: **240 × 240**
* Converted to 2D slices for training
* Binary segmentation masks used

### Dataset Structure

```bash
dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
├── masks/
│   ├── image_001_mask.png
│   ├── image_002_mask.png
```

Mask labels:

| Pixel Value | Meaning      |
| ----------- | ------------ |
| 0           | Background   |
| 255         | Tumor Region |

---

## 4. Methodology

### 4.1 Data Preprocessing

* Resizing images to 240 × 240
* Normalization of pixel values
* Binary mask conversion
* Dataset splitting into train, validation, and test sets

### 4.2 Data Augmentation

* Horizontal flipping
* Random rotation
* Brightness and contrast adjustments

### 4.3 Model Architectures

#### U-Net

U-Net is an encoder–decoder architecture designed for biomedical segmentation.

* Encoder extracts spatial features
* Bottleneck captures context
* Decoder reconstructs segmentation mask
* Skip connections preserve spatial information

#### Attention U-Net

Attention U-Net extends U-Net by incorporating attention gates.

* Filters irrelevant regions in the image
* Focuses on tumor-specific regions
* Improves localization accuracy
* Reduces false positives

This makes Attention U-Net particularly effective in complex medical segmentation tasks.

---

## 5. Training Configuration

| Parameter     | Value                   |
| ------------- | ----------------------- |
| Model         | U-Net / Attention U-Net |
| Input Size    | 240 × 240               |
| Optimizer     | Adam                    |
| Learning Rate | 0.0001                  |
| Batch Size    | 8                       |
| Epochs        | 40                      |

Loss Functions:

* Dice Loss
* Binary Cross Entropy Loss

---

## 6. Evaluation Metrics

* Dice Coefficient
* Intersection over Union (IoU)
* Precision
* Recall

---

## 7. Experimental Results

### Quantitative Results

| Model           | Accuracy | Precision | Recall | F1 Score | Dice | IoU  |
|----------------|----------|----------|--------|----------|------|------|
| U-Net           | 0.9965   | 0.9190   | 0.9500 | 0.9343   | 0.8283 | 0.7610 |
| Attention U-Net | 0.9965   | 0.9159   | 0.9525 | 0.9338   | 0.8308 | 0.7631 |

### Observation

- Both models successfully capture tumor regions  
- Attention U-Net produces slightly more refined segmentation  
- Improvement is observed in Dice coefficient and IoU  

---

### Qualitative Results

The model outputs include:

- Original MRI image  
- Ground truth tumor mask  
- Predicted mask (U-Net)  
- Predicted mask (Attention U-Net)  
- Overlay visualization  

---

#### Tumor Segmentation Comparison

<img width="100%" src="https://github.com/user-attachments/assets/3995f538-7060-4f83-b571-1f78ea22128b" />

<details>
<summary>View More Segmentation Results</summary>

##### Sample 1
<img width="100%" src="https://github.com/user-attachments/assets/6262d655-13a2-4a01-a121-5881f61d8b95" />

##### Sample 2
<img width="100%" src="https://github.com/user-attachments/assets/dc5656fb-2e22-4106-9bb8-cc066860ba5b" />

##### Sample 3
<img width="100%" src="https://github.com/user-attachments/assets/c65b2b0a-9c03-4fe4-917c-28164176c87a" />

</details>

---

#### Overlay Visualization

<img width="100%" src="https://github.com/user-attachments/assets/f474cd2f-38f7-4e22-b8c7-e6a73f1bf189" />

<details>
<summary>View More Overlay Results</summary>

##### Sample 1
<img width="100%" src="https://github.com/user-attachments/assets/7c466084-687f-4178-8209-6226455f308a" />

##### Sample 2
<img width="100%" src="https://github.com/user-attachments/assets/c417b557-fb92-4554-b33b-559cbde52db4" />

##### Sample 3
<img width="100%" src="https://github.com/user-attachments/assets/e78904e3-9835-4c28-8268-0ea2a88bbc5b" />

</details>

## 8. Applications

* Computer-aided diagnosis
* Radiology workflow automation
* Medical image analysis
* Clinical decision support

---

## 9. Limitations

* Dependency on dataset size
* MRI variability across scanners
* Limited generalization

---

## 10. Future Work

* Streamlit-based web application for tumor segmentation
* Real-time MRI upload and prediction
* Exploration of UNet++, DeepLabV3+, and 3D CNNs
* Integration of explainable AI

---

## 11. Project Structure

```bash
Automatic-Brain-Tumor-Segmentation/
├── configs/
├── data/
├── models/
├── notebook/
├── outputs/
├── src/
├── main.py
├── test.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 12. Installation and Setup

### Clone the repository

```bash
git clone git@github.com:Rajani1-tech/Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net.git
cd Automatic-Brain-Tumor-Segmentation-from-MRI-Images-using-U-Net
```

### Create virtual environment

```bash
python3 -m venv brain_seg_env
source brain_seg_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 13. Usage

### Training

```bash
python main.py
```

### Testing / Inference

```bash
python test.py
```

Outputs are saved in the `outputs/` directory.

---

## 14. Technologies Used

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

---
