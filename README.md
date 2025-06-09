# Diabetic Retinopathy Classification Using Deep Learning

This project leverages deep learning techniques to classify diabetic retinopathy from retinal images. Diabetic retinopathy is a diabetes complication that affects eyes and can lead to blindness if not detected early. This repository provides a comprehensive pipeline for data preprocessing, model training, evaluation, and prediction, aimed at assisting healthcare professionals in early diagnosis.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to automate the classification of diabetic retinopathy stages using fundus images and deep learning models (such as Convolutional Neural Networks). The solution includes data preprocessing, augmentation, model training, evaluation, and visualization of results.

## Features

- Data loading and preprocessing for retinal images
- Deep learning model implementation (CNN-based)
- Training and validation scripts
- Model evaluation and metrics visualization
- Prediction on new/unseen images
- Easily extensible and modular code

## Dataset

You can use publicly available datasets such as the [Kaggle Diabetic Retinopathy Detection dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). Please download and place the dataset in the appropriate directory as described below.

**Directory Structure:**
```
dataset/
  ├── train/
  ├── test/
  └── labels.csv
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dawoodshahzad07/Diabetic-Retinopathy-Classification-Using-Deep-Learning.git
   cd Diabetic-Retinopathy-Classification-Using-Deep-Learning
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your dataset:**  
   Place your training and test images in the `dataset/train` and `dataset/test` folders. Ensure that the `labels.csv` file is present and properly formatted.

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```

4. **Make predictions on new images:**
   ```bash
   python predict.py --image path/to/image.jpg
   ```

5. **View results and metrics:**  
   Check the `outputs/` directory for saved models, plots, and prediction results.

## Model Architecture

The core model is based on a Convolutional Neural Network (CNN) structure, which is effective for image classification tasks. The architecture can be customized in the `model.py` file. Transfer learning with pre-trained models such as ResNet or EfficientNet is also supported.

## Results

Results and model evaluation metrics (accuracy, confusion matrix, ROC curve, etc.) are saved in the `outputs/` directory after training and evaluation. Example results:

- **Accuracy:** 90% (example, please update with actual results)
- **Confusion Matrix:** ![Confusion Matrix](outputs/confusion_matrix.png)
- **ROC Curve:** ![ROC Curve](outputs/roc_curve.png)

## Contributing

Contributions are welcome! Please open issues and submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Contact:**  
Dawood Shahzad  
[GitHub Profile](https://github.com/dawoodshahzad07)
