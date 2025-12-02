# 🔬 Diabetic Retinopathy Detection & Grading Pipeline

A comprehensive deep learning pipeline for automated **Diabetic Retinopathy (DR)** detection and severity grading using state-of-the-art computer vision techniques.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Clinical Background](#-clinical-background)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Usage Guide](#-usage-guide)
- [Results](#-expected-results)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

Diabetic Retinopathy is a diabetes complication that affects the eyes and is a leading cause of blindness in working-age adults. Early detection through regular screening is crucial but requires trained specialists, making automated screening tools invaluable.

This pipeline provides:
- **Automated grading** of fundus images into 5 severity levels
- **State-of-the-art accuracy** using EfficientNet with custom pooling
- **Clinical explainability** through Grad-CAM visualizations
- **Production-ready deployment** with FastAPI and Docker

---

## 🏥 Clinical Background

This pipeline grades Diabetic Retinopathy according to the **International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale**:

| Grade | Severity | Clinical Findings | Action |
|:-----:|----------|-------------------|--------|
| **0** | No DR | No visible abnormalities | Annual screening |
| **1** | Mild NPDR | Microaneurysms only | Annual follow-up |
| **2** | Moderate NPDR | Microaneurysms + hemorrhages, hard exudates | 6-month follow-up |
| **3** | Severe NPDR | 4-2-1 Rule: hemorrhages in 4 quadrants, venous beading in 2+ quadrants, IRMA in 1+ quadrant | Refer to ophthalmologist |
| **4** | Proliferative DR | Neovascularization, vitreous/preretinal hemorrhage | **Urgent referral** |

**NPDR** = Non-Proliferative Diabetic Retinopathy  
**IRMA** = Intraretinal Microvascular Abnormalities

---

## 📊 Dataset

### APTOS 2019 Blindness Detection Challenge

This project uses the **APTOS 2019** dataset from Kaggle, provided by the **Aravind Eye Hospital** in India.

| Attribute | Value |
|-----------|-------|
| **Training Images** | 3,662 |
| **Image Format** | PNG (various resolutions) |
| **Classes** | 5 (0-4 severity grades) |
| **Source** | Aravind Eye Hospital, India |
| **Challenge** | Kaggle APTOS 2019 |

**Class Distribution:**
| Grade | Count | Percentage |
|-------|-------|------------|
| 0 - No DR | 1,805 | 49.3% |
| 1 - Mild | 370 | 10.1% |
| 2 - Moderate | 999 | 27.3% |
| 3 - Severe | 193 | 5.3% |
| 4 - Proliferative | 295 | 8.1% |

### Download Instructions

1. **Get Kaggle API Token:**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Click "Create New API Token"
   - Note your username and API key

2. **Setup credentials in `.env`:**
   ```env
   Kaggle_Username=your_kaggle_username
   Kaggle_API_Token=your_api_token
   ```

3. **Download the dataset:**
   - Visit [APTOS 2019 Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
   - Download and extract to `data/aptos/`

### Dataset Citation

```bibtex
@misc{aptos2019-blindness-detection,
    author = {APTOS, Aravind Eye Hospital},
    title = {APTOS 2019 Blindness Detection},
    publisher = {Kaggle},
    year = {2019},
    url = {https://www.kaggle.com/competitions/aptos2019-blindness-detection}
}
```

---

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Setup

```powershell
# Clone/navigate to project directory
cd "c:\path\to\Diabetic_RetinoPathy"

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Step 1: Prepare Data

Extract the downloaded APTOS dataset to:
```
data/
└── aptos/
    ├── train.csv
    ├── test.csv
    ├── train_images/
    │   ├── 000c1434d8d7.png
    │   └── ... (3,662 images)
    └── test_images/
```

### Step 2: Preprocess Images (Recommended)

Apply Ben Graham preprocessing for better results:

```powershell
python scripts/preprocess_images.py --input data/aptos/train_images --output data/aptos/processed --size 512
```

This applies:
- Circular cropping to remove black borders
- Gaussian blur subtraction for luminosity normalization
- CLAHE for contrast enhancement

### Step 3: Train Model

**Quick training (recommended for first run):**
```powershell
python train.py --quick --epochs 30 --batch-size 16
```

**With smaller batch size (if GPU memory is limited):**
```powershell
python train.py --quick --epochs 30 --batch-size 8 --accumulate 4
```

**Full training with Hydra config:**
```powershell
python train.py
```

### Step 4: Monitor Training

In a separate terminal:
```powershell
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

### Step 5: Run Inference

```powershell
python inference.py --image path/to/fundus_image.png --checkpoint checkpoints/YYYYMMDD_HHMMSS/last.ckpt
```

---

## 📁 Project Structure

```
Diabetic_RetinoPathy/
├── 📂 conf/                          # Hydra Configuration
│   ├── config.yaml                   # Main config entry point
│   ├── dataset/
│   │   └── aptos.yaml               # APTOS dataset settings
│   ├── model/
│   │   ├── efficientnet_b5.yaml     # EfficientNet-B5 config
│   │   └── efficientnet_b6.yaml     # EfficientNet-B6 config
│   ├── loss/
│   │   ├── regression.yaml          # MSE loss config
│   │   └── classification.yaml      # Focal loss config
│   └── training/
│       └── default.yaml             # Training hyperparameters
│
├── 📂 data/
│   └── aptos/                       # ← PUT YOUR DATA HERE
│       ├── train.csv
│       ├── train_images/
│       ├── test_images/
│       └── processed/               # Preprocessed images
│
├── 📂 src/
│   ├── datamodules/
│   │   ├── aptos_datamodule.py      # PyTorch Lightning DataModule
│   │   └── transforms.py            # Albumentations augmentations
│   ├── models/
│   │   ├── backbones.py             # EfficientNet + GeM pooling
│   │   ├── heads.py                 # Regression/Classification heads
│   │   └── dr_model.py              # Main model class
│   ├── utils/
│   │   ├── ben_graham.py            # Ben Graham preprocessing
│   │   ├── metrics.py               # QWK, confusion matrix
│   │   └── threshold_optimizer.py   # Optimal threshold finding
│   ├── xai/
│   │   └── gradcam.py               # Grad-CAM explainability
│   └── api/
│       └── app.py                   # FastAPI deployment
│
├── 📂 scripts/
│   ├── download_data.py             # Kaggle dataset download
│   └── preprocess_images.py         # Ben Graham preprocessing
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_training.ipynb            # Interactive training guide
│   └── 03_inference.ipynb           # Inference & visualization
│
├── 📂 checkpoints/                   # Saved model weights
├── 📂 logs/                          # TensorBoard logs
│
├── train.py                         # 🚀 Main training script
├── inference.py                     # Run predictions
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker container
├── docker-compose.yml               # Docker orchestration
└── README.md                        # This file
```

---

## 🔧 Technical Details

### Preprocessing: Ben Graham Method

The Ben Graham preprocessing technique (2015 Kaggle DR competition winner) normalizes retinal images:

```python
# Pseudocode
1. Estimate illumination using Gaussian blur (σ = image_size/10)
2. Subtract blurred image: I' = α*I + β*blur(I) + γ
3. Apply CLAHE for local contrast enhancement
4. Crop to circular fundus region
```

### Model Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backbone** | EfficientNet-B5 | Best accuracy/efficiency trade-off |
| **Pretrained** | Noisy Student | Better generalization than ImageNet |
| **Pooling** | GeM (p=3) | Preserves lesion information |
| **Head** | Regression (1 output) | Respects ordinal nature of grades |
| **Output** | Continuous [0-4] | Thresholds optimized post-training |

### Training Strategy

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | Weight decay for regularization |
| **Learning Rate** | 1e-4 | With cosine annealing |
| **Batch Size** | 16 (effective 32) | Gradient accumulation |
| **Epochs** | 30 | Early stopping on QWK |
| **Loss** | MSE | Smooth L1 alternative |
| **Metric** | Quadratic Weighted Kappa | Penalizes large errors more |

### Augmentations

```python
# Geometric (safe for fundus)
- Horizontal/Vertical Flip
- Rotation (0-360°)
- Scale (±20%)

# Photometric (moderate)
- Brightness/Contrast (±20%)
- Saturation (0.8-1.2x)
- Hue shift (±10°)  # Small to preserve hemorrhage colors

# Regularization
- CoarseDropout (30% probability)
```

---

## 📖 Usage Guide

### Training Options

```powershell
# Quick mode (recommended)
python train.py --quick

# Custom parameters
python train.py --quick \
    --backbone efficientnet_b5 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001 \
    --image-size 456

# Hydra mode (advanced)
python train.py training.epochs=50 model=efficientnet_b6
```

### Inference

```powershell
# Single image
python inference.py --image fundus.png --checkpoint checkpoints/best.ckpt

# Batch inference
python inference.py --input-dir test_images/ --output results.csv

# With TTA (Test Time Augmentation)
python inference.py --image fundus.png --tta
```

### API Deployment

```powershell
# Start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

API Endpoints:
- `POST /predict` - Single image prediction
- `POST /batch-predict` - Multiple images
- `GET /health` - Health check

---

## 📈 Expected Results

| Model | Val QWK | Test QWK | Notes |
|-------|---------|----------|-------|
| EfficientNet-B5 | ~0.90 | ~0.88 | Single model |
| EfficientNet-B6 | ~0.91 | ~0.89 | Higher resolution |
| 5-Fold Ensemble | ~0.92 | ~0.90 | Competition-level |

**Note:** Results may vary based on preprocessing, augmentation, and random seed.

---

## 📚 References

### Papers

1. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.

2. **Ben Graham Preprocessing**: Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition, 1st Place Solution.

3. **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.

4. **ICDR Scale**: Wilkinson, C. P., et al. (2003). Proposed International Clinical Diabetic Retinopathy Disease Severity Scale. Ophthalmology.

### Competitions & Datasets

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) - Kaggle
- [Diabetic Retinopathy Detection 2015](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) - Kaggle (EyePACS)

### Acknowledgments

- **Aravind Eye Hospital** for providing the APTOS dataset
- **Kaggle** for hosting the competition
- **EyePACS** for the 2015 DR dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is **NOT** a medical device and should **NOT** be used for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

<p align="center">
  Made with ❤️ for better diabetic eye care
</p>
