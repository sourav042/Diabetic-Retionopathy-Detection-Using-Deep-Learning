# Diabetic Retinopathy Detection using Ensemble Deep Learning

## 🩺 Problem Statement  
Diabetic Retinopathy (DR) is a leading cause of vision loss in the global working-age population. Early detection and treatment can significantly reduce the risk of blindness. However, manual screening is time-consuming, expensive, and subject to human error.  
This project aims to automate the detection of diabetic retinopathy from retinal images using deep learning models, improving screening efficiency and diagnostic accuracy.

## 📊 Dataset  
- **Source**: [APTOS 2019 Blindness Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- The dataset consists of high-resolution retinal fundus images labeled across five DR severity classes:
  - 0: No DR  
  - 1: Mild  
  - 2: Moderate  
  - 3: Severe  
  - 4: Proliferative DR  

## 🧠 Model Architecture  
The project uses an **ensemble of three CNN architectures** to enhance prediction performance:
- EfficientNetB4  
- EfficientNetB5  
- ResNet152V2  

Each model is pretrained on ImageNet and fine-tuned on the APTOS dataset. A **max-voting ensemble** strategy is used to combine predictions from all three models.

## ⚙️ Preprocessing  
- Image resizing to 380×380 pixels  
- Pixel value normalization  
- Data augmentation (optional for training phase)  
- One-hot encoding of class labels  

## 🖥️ Deployment  
A lightweight, interactive **Gradio web interface** was developed to:
- Accept retinal image uploads
- Display the predicted DR class with associated confidence
- Run entirely client-side for demonstration purposes

## ✅ Features  
- Robust ensemble classification using pre-trained deep models  
- Interactive image prediction with Gradio  
- Clean modular code with custom layer support  
- Max-voting strategy to increase prediction robustness  

## 🔧 Requirements  
- Python 3.7+  
- TensorFlow / Keras  
- OpenCV  
- Gradio  
- NumPy, Pandas, PIL  

Install dependencies using:
```bash
pip install -r requirements.txt
