# 👁️ Diabetic Retinopathy Detection System

A production-ready deep learning system for automated **Diabetic Retinopathy (DR)** screening using state-of-the-art computer vision and explainable AI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Highlights

| Feature | Description |
|---------|-------------|
| **🏆 High Accuracy** | QWK Score of **0.87** - state-of-the-art performance |
| **🖥️ Web Interface** | Interactive Streamlit app for easy screening |
| **🔍 Explainable AI** | Grad-CAM visualizations show model reasoning |
| **⚡ Fast Inference** | ~3.5 images/second on CPU, faster on GPU |
| **📦 Production Ready** | FastAPI + Docker for deployment |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Quick Start](#-quick-start)
- [Web Application](#-web-application)
- [Technical Architecture](#-technical-architecture)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Inference](#-inference)
- [API Deployment](#-api-deployment)
- [References](#-references)

---

## 🔬 Overview

Diabetic Retinopathy is a diabetes complication affecting the eyes and a leading cause of blindness. This system provides automated screening by grading retinal fundus images into 5 severity levels:

| Grade | Name | Severity | Clinical Action |
|:-----:|------|----------|-----------------|
| **0** | No DR | Normal | Annual screening |
| **1** | Mild NPDR | Low | Annual follow-up |
| **2** | Moderate NPDR | Medium | 6-month follow-up |
| **3** | Severe NPDR | High | Refer to ophthalmologist |
| **4** | Proliferative DR | Critical | **Urgent treatment** |

---

## 📊 Model Performance

### Trained Model Statistics

| Metric | Value |
|--------|-------|
| **Validation QWK** | 0.8646 |
| **Best QWK (optimized thresholds)** | 0.8716 |
| **Backbone** | EfficientNet-B5 (Noisy Student) |
| **Parameters** | 29.4M |
| **Input Size** | 456 × 456 |
| **Inference Speed** | ~3.5 img/sec (CPU) |

### Optimized Thresholds

The model uses regression with optimized thresholds for better ordinal classification:

```
Grade 0 → 1: 0.5406
Grade 1 → 2: 1.4384  
Grade 2 → 3: 2.6956
Grade 3 → 4: 3.3914
```

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone/navigate to project
cd "Diabetic_RetinoPathy"

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web App

```powershell
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 3. Run Demo Script

```powershell
python demo_implementation.py
```

This will:
- Load the trained model
- Run predictions on test images
- Generate Grad-CAM visualizations
- Save results to `outputs/demo_predictions/`

---

## 🖥️ Web Application

The Streamlit app provides an interactive interface for DR screening.

### Features

| Page | Description |
|------|-------------|
| **🔍 Screening** | Upload images for instant DR grading |
| **🎓 Demo** | See examples from each DR grade with ground truth comparison |
| **ℹ️ About** | Model architecture and clinical information |

### Running the App

```powershell
# Start the app
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8080
```

### App Features

**Screening Page:**
- Upload fundus images (PNG/JPG)
- Get instant DR grade prediction
- View confidence scores
- See Grad-CAM explainability heatmap

**Demo Page:**
- One sample from each DR grade (0-4)
- Ground truth vs. prediction comparison
- Accuracy metrics and model evaluation
- Full Grad-CAM visualizations

---

## 🏗️ Technical Architecture

### Model Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (456×456×3)                    │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│              EFFICIENTNET-B5 (Noisy Student)                  │
│  • 30M parameters                                             │
│  • Compound scaling (width=1.6, depth=2.2)                   │
│  • Pre-trained on ImageNet with semi-supervised learning     │
│  Output: 2048-channel feature maps                           │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   GeM POOLING (p=3.0)                         │
│  • Learnable generalized mean pooling                        │
│  • Between avg pooling (p=1) and max pooling (p→∞)          │
│  Output: 2048-dim vector                                     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    REGRESSION HEAD                            │
│  • Linear(2048→512) + BatchNorm + ReLU + Dropout(0.5)       │
│  • Linear(512→1)                                             │
│  Output: Single value [0, 4]                                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│               THRESHOLD CONVERSION                            │
│  • Optimized thresholds: [0.54, 1.44, 2.70, 3.39]           │
│  Output: DR Grade [0-4]                                      │
└──────────────────────────────────────────────────────────────┘
```

### Why Regression Instead of Classification?

DR grades are **ordinal** (ordered): 0 < 1 < 2 < 3 < 4

| Approach | Pros | Cons |
|----------|------|------|
| **Classification** | Simple | Treats all errors equally |
| **Regression** ✓ | Respects order, optimizable thresholds | Requires threshold tuning |

### Ben Graham Preprocessing

Industry-standard preprocessing for fundus images (2015 Kaggle DR competition winner):

```
1. CROP TO SQUARE
   └── Find fundus region, crop with 5% padding

2. GAUSSIAN BLUR SUBTRACTION  
   └── I' = 4×I - 4×blur(I) + 128
   └── Acts as high-pass filter (enhances lesions)

3. CIRCULAR MASK
   └── Remove edge artifacts, focus on fundus
```

### Grad-CAM Explainability

Visualizes which regions the model focuses on:

- **🔴 Red/Hot**: High attention (potential lesions)
- **🔵 Blue/Cold**: Low attention (healthy tissue)

---

## 📁 Project Structure

```
Diabetic_RetinoPathy/
│
├── 🚀 MAIN APPLICATIONS
│   ├── app.py                    # Streamlit web application
│   ├── demo_implementation.py    # Educational demo script
│   ├── train.py                  # Training script
│   └── inference.py              # CLI inference tool
│
├── 📂 models/                    # Trained model checkpoints
│   ├── dr-epoch=04-val_qwk=0.8646.ckpt   # Best model
│   ├── model_info.json           # Model metadata & thresholds
│   └── last.ckpt                 # Latest checkpoint
│
├── 📂 src/                       # Source code
│   ├── models/                   # Model architecture
│   ├── datamodules/              # Data loading & transforms
│   ├── utils/                    # Preprocessing & metrics
│   ├── xai/                      # Explainability (Grad-CAM)
│   └── api/                      # FastAPI deployment
│
├── 📂 data/aptos/                # APTOS 2019 dataset
├── 📂 notebooks/                 # Jupyter notebooks
├── 📂 outputs/                   # Generated predictions
├── 📂 logs/                      # Training logs
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🎓 Training

### Dataset: APTOS 2019

| Attribute | Value |
|-----------|-------|
| **Training Images** | 3,662 |
| **Test Images** | 1,928 |
| **Source** | Aravind Eye Hospital, India |

### Training on Google Colab

Use the provided notebook for free GPU training:

1. Open `notebooks/04_colab_training.ipynb` in Google Colab
2. Mount your Google Drive
3. Upload the dataset
4. Run all cells
5. Download trained model to `models/` directory

### Local Training

```powershell
python train.py --quick --epochs 30 --batch-size 16
```

---

## 🔮 Inference

### Command Line

```powershell
python inference.py --image path/to/fundus.png
```

### Python API

```python
from demo_implementation import DRModel, predict_image

result = predict_image("image.png", model, preprocessor, thresholds, device)
print(f"Grade: {result['predicted_class']}, Confidence: {result['confidence']:.1%}")
```

---

## 🌐 API Deployment

### Streamlit (Recommended)

```powershell
streamlit run app.py
```

### FastAPI

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Docker

```powershell
docker-compose up
```

---

## 📚 References

- **EfficientNet** - Tan & Le (2019). ICML.
- **Noisy Student** - Xie et al. (2020). CVPR.
- **Grad-CAM** - Selvaraju et al. (2017). ICCV.
- **APTOS 2019** - [Kaggle Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

---

## ⚠️ Disclaimer

This tool is for **screening and educational purposes only**. It is **NOT** a certified medical device. Always consult qualified healthcare professionals.

---

## 📄 License

MIT License

---

<p align="center">
  Made with ❤️ for better diabetic eye care<br>
  <strong>Early detection saves sight!</strong>
</p>
