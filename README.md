# 🧠 Diabetic Retinopathy Classification Using Deep Learning

This project presents a deep learning-based approach to classify retinal images into different stages of Diabetic Retinopathy (DR) using state-of-the-art architectures including **ConvNeXt** and a **Hybrid ConvNeXt + CNN** model.

## 📁 Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Comparative Analysis](#comparative-analysis)
- [Innovations in ConvNeXt](#innovations-in-convnext)
- [Conclusion](#conclusion)
- [References](#references)
- [How to Run](#how-to-run)

---

## 📌 Overview

Diabetic Retinopathy is a serious eye condition caused by diabetes that can lead to vision loss. Early and accurate detection is vital. This project leverages modern deep learning architectures to classify retinal images into five categories representing different stages of DR.

---

## 📊 Dataset

We used the **APTOS 2019 Blindness Detection** dataset which includes:

- **3,662** training images
- **1,928** test images  
- **5 DR classes**:
  - No DR
  - Mild
  - Moderate
  - Severe
  - Proliferative DR

To address class imbalance, we curated a balanced subset:

- **Balanced Training Set**: 1500 images  
- **Testing Set**: 600 images (split from training leftovers)

---

## 🧠 Models

### Model 01: ConvNeXt

- Base: `ConvNeXt-Tiny` (Pretrained on ImageNet)
- Custom classification head:
  - Linear (1280 → 512) → ReLU → Dropout(0.3)
  - Linear (512 → 5)

---

### Model 02: ConvNeXt + CNN (Hybrid)

- Base: `ConvNeXt-Tiny` as feature extractor
- Custom head:
  - Linear (768 → 512)
  - CNN Block: Conv2D → BatchNorm → ReLU → Dropout(0.3)
  - Adaptive Average Pooling
  - Linear (256 → 5)

---

## 🛠 Preprocessing

Images were preprocessed as follows:

- Resized to **224x224**
- Normalized using **ImageNet** statistics
- Data Augmentation:
  - Horizontal & vertical flips
  - Random rotations
  - Elastic transformations
  - Brightness & contrast adjustments

---

## ⚙️ Training Configuration

- **Optimizer**: AdamW  
- **Learning Rate**: 1e-4  
- **Weight Decay**: 1e-4  
- **Batch Size**: 32  
- **Epochs**: 10  
- **Scheduler**: ReduceLROnPlateau  
- **Loss Function**: CrossEntropyLoss  
- **Checkpointing**: Best model saved based on validation accuracy  

---

## 📈 Results

### Validation Performance

| Model            | Train Acc | Val Acc | Train Loss | Val Loss |
|------------------|-----------|---------|------------|----------|
| ConvNeXt         | 82.20%    | 84.64%  | 0.4798     | 0.4000   |
| ConvNeXt + CNN   | 91.71%    | 81.42%  | 0.0079     | 0.0192   |

### Testing Metrics

| Model            | Precision | Recall | F1-Score | Test Accuracy |
|------------------|-----------|--------|----------|----------------|
| ConvNeXt         | 0.86      | 0.85   | 0.84     | 84.97%         |
| ConvNeXt + CNN   | 0.85      | 0.84   | 0.84     | 84.15%         |

Confusion matrices and ROC curves are provided in the report.

---

## 🔬 Comparative Analysis

While the hybrid model achieved higher training accuracy, ConvNeXt showed better generalization on the validation and test sets, indicating robustness and reliability for real-world application.

---

## 💡 Innovations in ConvNeXt

ConvNeXt improves upon traditional CNNs through:

- Modernized ResNet design
- Depthwise convolutions for efficiency
- LayerNorm replacing BatchNorm
- GELU activation for smoother gradients
- Stochastic depth regularization

---

## ✅ Conclusion

ConvNeXt proved to be a highly effective model for classifying diabetic retinopathy. Its architecture offers a robust, efficient, and scalable solution suitable for clinical use.

---

## 📚 References

1. Z. Liu et al., *A ConvNet for the 2020s*, 2022.  
2. K. O’Shea, R. Nash, *An Introduction to Convolutional Neural Networks*, 2015.  
3. APTOS 2019 Blindness Detection Dataset  
4. Diabetic Retinopathy Detection Guidelines  

---

## ▶️ How to Run

1. Clone this repository  
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
