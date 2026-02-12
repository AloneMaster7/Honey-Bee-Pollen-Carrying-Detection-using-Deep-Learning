# ⚠️ Uploading ...

# 🐝 Honey Bee Pollen Carrying Detection using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning project for binary classification of honey bees to detect whether they are carrying pollen or not. Three different architectures (Custom CNN, VGG-inspired, and Lightweight ResNet) are implemented and compared on RGB images.

## 📊 Dataset Structure

```
./bee/
├── bee_data.csv          # Labels CSV file
└── bee_imgs/            # Bee images folder
```

**CSV Format:**
| Column | Description |
|--------|-------------|
| File | Image filename |
| pollen_carrying | Target label (1: carrying pollen, 0: not carrying) |

**Data Processing Pipeline:**
1. Read CSV using pandas
2. Construct full image paths
3. Load images with OpenCV (BGR → RGB conversion)
4. Resize to 128×128 pixels
5. Normalize pixel values to [0,1]
6. Create stratified train/test split using scikit-learn

**Input Specifications:**
- Image size: 128×128
- Channels: 3 (RGB)
- Normalization: [0,1] range

## 🏗️ Model Architectures

### 1️⃣ Custom CNN (Enhanced Architecture)
A deeper architecture designed for RGB image complexity with progressive filter reduction:

```
Input (128,128,3)
↓
Conv2D(128) → BatchNorm → MaxPooling
↓
Conv2D(64) → BatchNorm → MaxPooling
↓
Conv2D(32) → BatchNorm → MaxPooling
↓
Flatten → Dense(128) → Dropout(0.5)
↓
Output (1, Sigmoid)
```

**Configuration:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy
- Epochs: 15
- Batch Size: 32
- Input Shape: (128,128,3)

**Performance:** ✅ **99.71% Accuracy**

*Why this works: Progressive filter reduction (128→64→32) captures color and texture features effectively for bee classification.*

---

### 2️⃣ VGG-inspired Architecture
Deeper architecture with multiple consecutive convolutions per block, optimized for RGB bee images:

```
Input (128,128,3)
↓
[Conv2D(8) → Conv2D(8)] → BatchNorm → MaxPooling
↓
[Conv2D(16) → Conv2D(16)] → BatchNorm → MaxPooling
↓
[Conv2D(32) → Conv2D(32)] → BatchNorm → MaxPooling
↓
Flatten → Dense(128) → Dropout(0.5)
↓
Output (1, Sigmoid)
```

**Features:**
- 3×3 kernels with 'same' padding
- Dual convolution layers per block
- Progressive filter increase (8→16→32)
- Batch Normalization after each convolution block
- 15 epochs, batch size 32
- Input: (128,128,3)

**Performance:** ✅ **99.73% Accuracy** (Best Model)

*Why this works: Multiple consecutive convolutions extract fine-grained details and texture patterns that distinguish pollen-carrying bees.*

---

### 3️⃣ Lightweight ResNet
Residual architecture with skip connections, adapted for RGB bee classification:

```
Input (128,128,3)
↓
Conv2D(64) → BatchNorm → MaxPooling
    ↓
    ┌── Residual Block (64 filters) ──┐
    ↓                                 ↓
Conv2D(32) → BatchNorm → MaxPooling ← Skip Connection
    ↓
    ┌── Residual Block (32 filters) ──┐
    ↓                                 ↓
Conv2D(16) → BatchNorm → MaxPooling ← Skip Connection
    ↓
    ┌── Residual Block (16 filters) ──┐
    ↓                                 ↓
Global Average Pooling → Dense(128) → Dropout(0.5)
↓
Output (1, Sigmoid)
```

**Features:**
- Skip connections for better gradient flow
- Progressive filter reduction (64→32→16)
- Global Average Pooling instead of Flatten
- Dropout 0.5 for regularization
- Input: (128,128,3)
- 15 epochs, batch size 32

**Performance:** ✅ **99.66% Accuracy**

*Why this works: Residual blocks learn complex color and texture variations while skip connections prevent gradient vanishing in deeper layers.*

## 📊 Performance Comparison

| Architecture | Accuracy | Input | Key Strength |
|-------------|----------|--------|--------------|
| Custom CNN | 99.71% | RGB (128×128) | Progressive filter reduction |
| VGG-inspired | **99.73%** | RGB (128×128) | Multi-conv feature extraction |
| Lightweight ResNet | 99.66% | RGB (128×128) | Skip connections for deep features |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/AloneMaster7/Honey-Bee-Pollen-Carrying-Detection-using-Deep-Learning.git
cd Honey-Bee-Pollen-Carrying-Detection-using-Deep-Learning
```

## 📦 Requirements

```
tensorflow>=2.0.0
opencv-python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 📁 Project Structure

```
.
├── models/              # Model architectures
│   ├── cnn_model.py
│   ├── vgg_model.py
│   └── resnet_model.py
├── notebooks/
│   ├── cnn_model.py
│   ├── vgg_model.py
│   └── resnet_model.py
├── dataset/
│   ├── BeeDataset.rar
└── README.md          # This file
```

## 📈 Results Visualization

The project includes:
- Training/validation accuracy curves
- Confusion matrices
- ROC curves
- Sample predictions visualization

## 🔬 Key Findings

1. **VGG-inspired architecture** achieved the highest accuracy (99.73%) for pollen detection
2. **RGB color information** is crucial - grayscale conversion reduces accuracy significantly
3. **Stratified splitting** ensures balanced class distribution in train/test sets
4. **Dropout (0.5)** effectively prevents overfitting despite small dataset

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

---

**⭐ If you find this project useful for bee research or agricultural applications, please consider giving it a star!**

---

**📧 Contact:** For questions or collaborations, please open an issue or contact [jvd.r.403@gmail.com].

---
*This project is part of research on automated bee behavior monitoring using computer vision.* 🐝
