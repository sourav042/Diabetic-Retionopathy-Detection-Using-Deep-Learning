# Diabetic Retinopathy Detection using Deep Learning

## 🩺 Problem Statement

Diabetic Retinopathy (DR) is one of the leading causes of blindness among the working-age population. Early detection and treatment are crucial to prevent severe vision loss. Manual grading of retinal images is time-consuming and prone to subjectivity. Automatic retinal image analysis has become a vital screening tool, helping to reduce both the workload of ophthalmologists and the cost of diagnosis.

This project leverages deep learning techniques to automatically detect Diabetic Retinopathy from retinal fundus images. Our goal is to build an effective and scalable model that can assist in early DR diagnosis and be deployed in large-scale screening programs.

---

## 📁 Dataset

**Source**: [APTOS 2019 Blindness Detection Dataset - Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

The dataset consists of high-resolution retina images taken under a variety of imaging conditions. Each image is labeled with a severity grade of Diabetic Retinopathy on a scale from 0 to 4:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

---

## 🧠 Approach

We adopt a deep learning-based approach for the multi-class classification of Diabetic Retinopathy severity levels.

### 🔧 Architecture Overview

- **Transfer Learning**: We leverage pre-trained convolutional neural networks (CNNs) such as **EfficientNet**, **ResNet**, and **InceptionV3** as the backbone of our models. These architectures are fine-tuned on the APTOS dataset to adapt to the task of retinal image classification.
- **Image Preprocessing**: Includes resizing, normalization, and image enhancement techniques (e.g., CLAHE, Gaussian blurring).
- **Data Augmentation**: Random rotations, horizontal/vertical flips, brightness and contrast adjustment to increase dataset variability and prevent overfitting.
- **Loss Function**: 
  - `Categorical Crossentropy` for balanced training.
  - `Focal Loss` to handle class imbalance more effectively.
- **Optimizer**: Adam / SGD with learning rate scheduling.
- **Evaluation Metrics**:
  - **Quadratic Weighted Kappa (QWK)** – preferred metric for DR detection competitions.
  - **Accuracy**, **Precision**, **Recall**, and **Confusion Matrix** for performance insights.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Kaggle Notebooks (for training environment)

---

## 🚀 Getting Started

### Prerequisites

Make sure the following packages are installed:

```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow
