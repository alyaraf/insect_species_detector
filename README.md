# 🦋 Insect Species Classification using ResNet50 & TTA

This project builds a deep learning model to classify images of insect species using transfer learning with **ResNet50** and enhanced **Test-Time Augmentation (TTA)**. It was developed for an image classification challenge involving real-world biological data.

---

## 📌 Project Overview

- **Dataset**: A labeled image dataset of insect species (`train.csv`, `train/`, `test/`)
- **Goal**: Accurately classify species from test images
- **Approach**:
  - Pre-trained ResNet50 as the base
  - Custom classification head
  - Two-stage training (feature extraction + fine-tuning)
  - Advanced data augmentation (training and inference)
  - Test-Time Augmentation (TTA) for robust predictions

---

## 🛠 Technologies

- Python, TensorFlow, Keras
- Pandas, NumPy
- ResNet50 (transfer learning)
- ImageDataGenerator (Keras)
- Test-Time Augmentation (custom implementation)

---

## 🧠 Model Architecture

- **Base**: `ResNet50` (imagenet weights, frozen in stage 1)
- **Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.5)`
  - `Dense(256, relu)`
  - `Dropout(0.5)`
  - `Dense(n_classes, softmax)`

**Training Strategy**:
1. **Feature Extraction** (5 epochs): Only top layers are trained.
2. **Fine-Tuning** (45 epochs): Last 10 layers of ResNet50 are unfrozen and retrained with a low learning rate.

---

## 🧪 Test-Time Augmentation (TTA)

Instead of predicting once per image, the model makes multiple predictions with different augmentations and averages the results. This improves robustness to real-world variance.

**TTA Techniques Used**:
- Random rotation
- Zooming
- Shearing
- Brightness shifts
- Horizontal flipping

## 📥 Dataset Download
Run the following command to download the dataset:
import gdown

# Google Drive shared link
url = '[https://drive.google.com/uc?id=FILE_ID](https://drive.google.com/file/d/1KqcW3DQqgSavImVs699yZ4Zf_eprRgsS/view?usp=drive_link)'
output = 'data.zip'
gdown.download(url, output, quiet=False)

# Unzip after download
import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data/')

