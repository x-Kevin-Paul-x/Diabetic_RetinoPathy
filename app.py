"""
Diabetic Retinopathy Detection - Streamlit App
================================================
A web-based interface for DR screening using deep learning.

Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import warnings
import random

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import timm
import pandas as pd

# =====================================================================================
# PAGE CONFIGURATION
# =====================================================================================
st.set_page_config(
    page_title="DR Detection - AI Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# CUSTOM CSS
# =====================================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .grade-0 { background-color: #c8e6c9; border-left: 5px solid #4caf50; }
    .grade-1 { background-color: #fff9c4; border-left: 5px solid #ffeb3b; }
    .grade-2 { background-color: #ffe0b2; border-left: 5px solid #ff9800; }
    .grade-3 { background-color: #ffcdd2; border-left: 5px solid #f44336; }
    .grade-4 { background-color: #f8bbd9; border-left: 5px solid #880e4f; }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# MODEL COMPONENTS
# =====================================================================================

class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else torch.tensor([p])
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x.pow(self.p), 1)
        return pooled.pow(1.0 / self.p).flatten(1)


class RegressionHead(nn.Module):
    """Regression head for ordinal output."""
    def __init__(self, in_features: int, hidden_dims: List[int] = [512], dropout: float = 0.5):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)
        
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class DRModel(nn.Module):
    """Diabetic Retinopathy Detection Model."""
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b5_ns",
        pretrained: bool = True,
        pooling: str = "gem",
        head_dropout: float = 0.5,
        hidden_dims: List[int] = [512],
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.num_features = features.shape[1]
        
        if pooling == "gem":
            self.pool = GeM(p=3.0, trainable=True)
        else:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        
        self.head = RegressionHead(
            in_features=self.num_features,
            hidden_dims=hidden_dims,
            dropout=head_dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.head(pooled)


class BenGrahamPreprocessor:
    """Ben Graham preprocessing for fundus images."""
    def __init__(
        self,
        output_size: int = 512,
        alpha: float = 4.0,
        beta: float = -4.0,
        gamma: float = 128.0,
        sigma_ratio: float = 0.1,
        apply_circular_mask: bool = True,
    ):
        self.output_size = output_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_ratio = sigma_ratio
        self.apply_circular_mask = apply_circular_mask
        
        sigma = int(output_size * sigma_ratio)
        self.kernel_size = sigma * 2 + 1
        self.sigma = sigma
    
    def _find_eye_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, image.shape[1], image.shape[0]
        
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    
    def _crop_to_square(self, image: np.ndarray) -> np.ndarray:
        x, y, w, h = self._find_eye_region(image)
        center_x = x + w // 2
        center_y = y + h // 2
        
        size = max(w, h)
        size = int(size * 1.05)
        half_size = size // 2
        
        img_h, img_w = image.shape[:2]
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(img_w, center_x + half_size)
        y2 = min(img_h, center_y + half_size)
        
        cropped = image[y1:y2, x1:x2]
        
        crop_h, crop_w = cropped.shape[:2]
        if crop_h != crop_w:
            max_side = max(crop_h, crop_w)
            square = np.zeros((max_side, max_side, 3), dtype=np.uint8)
            y_offset = (max_side - crop_h) // 2
            x_offset = (max_side - crop_w) // 2
            square[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = cropped
            cropped = square
        
        return cropped
    
    def _apply_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        processed = cv2.addWeighted(image, self.alpha, blurred, self.beta, self.gamma)
        return np.clip(processed, 0, 255).astype(np.uint8)
    
    def _apply_circular_mask_fn(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center) - 2
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        result = image.copy()
        result[mask == 0] = 0
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        cropped = self._crop_to_square(image)
        resized = cv2.resize(cropped, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        processed = self._apply_gaussian_filter(resized)
        
        if self.apply_circular_mask:
            processed = self._apply_circular_mask_fn(processed)
        
        return processed


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
        self.target_layer = None
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layer = module
        
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        if self.target_layer:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output.squeeze()
        score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# =====================================================================================
# CLASS INFORMATION
# =====================================================================================

CLASS_INFO = {
    0: {
        "name": "No DR",
        "description": "No diabetic retinopathy detected. The retina appears healthy.",
        "severity": "Normal",
        "color": "#4caf50",
        "recommendation": "Continue annual screening. Maintain good blood sugar control.",
        "css_class": "grade-0"
    },
    1: {
        "name": "Mild NPDR",
        "description": "Mild Non-Proliferative DR. Small areas of swelling (microaneurysms) detected.",
        "severity": "Low",
        "color": "#ffeb3b",
        "recommendation": "Annual screening recommended. Focus on blood sugar and blood pressure control.",
        "css_class": "grade-1"
    },
    2: {
        "name": "Moderate NPDR", 
        "description": "Moderate Non-Proliferative DR. Blood vessels are showing signs of blockage or damage.",
        "severity": "Medium",
        "color": "#ff9800",
        "recommendation": "6-month follow-up recommended. Consider referral to ophthalmologist.",
        "css_class": "grade-2"
    },
    3: {
        "name": "Severe NPDR",
        "description": "Severe Non-Proliferative DR. Many blood vessels are blocked, high risk of progression.",
        "severity": "High",
        "color": "#f44336",
        "recommendation": "Immediate referral to ophthalmologist. Close monitoring required.",
        "css_class": "grade-3"
    },
    4: {
        "name": "Proliferative DR",
        "description": "Proliferative DR. New abnormal blood vessels are growing. Risk of vision loss.",
        "severity": "Critical",
        "color": "#880e4f",
        "recommendation": "URGENT: Immediate treatment required (laser therapy or injections).",
        "css_class": "grade-4"
    }
}

# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find best checkpoint
    models_dir = PROJECT_ROOT / "models"
    checkpoints = list(models_dir.glob("dr-epoch=*.ckpt"))
    
    if not checkpoints:
        st.error("No model checkpoints found in 'models' directory!")
        return None, None, None
    
    best_ckpt = None
    best_qwk = 0
    
    for ckpt in checkpoints:
        name = ckpt.stem
        if "val_qwk=" in name:
            qwk = float(name.split("val_qwk=")[1])
            if qwk > best_qwk:
                best_qwk = qwk
                best_ckpt = ckpt
    
    # Load model info
    info_path = PROJECT_ROOT / "models" / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            model_info = json.load(f)
        thresholds = model_info.get('optimized_thresholds', [0.5, 1.5, 2.5, 3.5])
    else:
        model_info = {}
        thresholds = [0.5, 1.5, 2.5, 3.5]
    
    # Load checkpoint
    checkpoint = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Create model
    model = DRModel(
        backbone_name="tf_efficientnet_b5_ns",
        pretrained=False,
        pooling="gem",
        head_dropout=0.5,
        hidden_dims=[512],
    )
    
    # Fix state dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, thresholds, device


def regression_to_class(value: float, thresholds: List[float]) -> int:
    """Convert regression value to class using thresholds."""
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)


def compute_confidence(raw_value: float, thresholds: List[float]) -> float:
    """Compute prediction confidence."""
    min_dist = float('inf')
    for threshold in thresholds:
        dist = abs(raw_value - threshold)
        min_dist = min(min_dist, dist)
    
    confidence = 1.0 - np.exp(-min_dist * 2)
    return float(confidence)


@torch.no_grad()
def predict_image(image: np.ndarray, model, preprocessor, thresholds, device, image_size=456):
    """Run prediction on an image."""
    # Preprocess
    processed = preprocessor(image)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Transform
    img = cv2.resize(processed_rgb, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Inference
    output = model(tensor)
    raw_value = float(output.squeeze().cpu().numpy())
    
    predicted_class = regression_to_class(raw_value, thresholds)
    confidence = compute_confidence(raw_value, thresholds)
    
    return {
        "raw_value": raw_value,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "processed_image": processed_rgb,
        "tensor": tensor,
    }


def generate_gradcam(model, tensor, processed_image):
    """Generate Grad-CAM heatmap."""
    gradcam = GradCAM(model)
    
    with torch.enable_grad():
        tensor_grad = tensor.clone().requires_grad_(True)
        heatmap = gradcam(tensor_grad)
    
    heatmap_resized = cv2.resize(heatmap, (processed_image.shape[1], processed_image.shape[0]))
    
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(processed_image, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap_resized, overlay


# =====================================================================================
# MAIN APP
# =====================================================================================

def get_sample_images_by_grade():
    """Get sample images for each DR grade from training set."""
    train_csv = PROJECT_ROOT / "data" / "aptos" / "train.csv"
    train_dir = PROJECT_ROOT / "data" / "aptos" / "train_images"
    
    if not train_csv.exists() or not train_dir.exists():
        return None
    
    df = pd.read_csv(train_csv)
    
    samples = {}
    for grade in range(5):
        grade_images = df[df['diagnosis'] == grade]['id_code'].tolist()
        if grade_images:
            # Get up to 5 random samples per grade
            selected = random.sample(grade_images, min(5, len(grade_images)))
            samples[grade] = [train_dir / f"{img}.png" for img in selected if (train_dir / f"{img}.png").exists()]
    
    return samples


def render_prediction_card(result, info, predicted_class):
    """Render a styled prediction result card."""
    st.markdown(f"""
    <div class="prediction-box {info['css_class']}">
        <h2 style="margin: 0; color: {info['color']};">Grade {predicted_class}: {info['name']}</h2>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">{info['description']}</p>
        <p style="font-weight: bold;">📋 Recommendation: {info['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_score_visualization(raw_value, thresholds, predicted_class):
    """Render the score visualization chart."""
    score_normalized = min(max(raw_value / 4.0, 0), 1)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    colors = ['#4caf50', '#ffeb3b', '#ff9800', '#f44336', '#880e4f']
    regions = [0] + [t/4 for t in thresholds] + [1]
    
    for i in range(5):
        ax.axvspan(regions[i], regions[i+1], alpha=0.3, color=colors[i])
    
    for t in thresholds:
        ax.axvline(t/4, color='gray', linestyle='--', linewidth=1)
    
    ax.axvline(score_normalized, color='blue', linewidth=3)
    ax.scatter([score_normalized], [0.5], color='blue', s=200, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0\nNo DR', '1\nMild', '2\nModerate', '3\nSevere', '4\nPDR'])
    ax.set_yticks([])
    ax.set_title(f"Prediction: {raw_value:.3f} → Grade {predicted_class}")
    
    st.pyplot(fig)
    plt.close()


def page_screening():
    """Main screening page."""
    st.header("📤 Upload Retinal Image")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, thresholds, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the models directory.")
        return
    
    preprocessor = BenGrahamPreprocessor(output_size=512)
    
    uploaded_file = st.file_uploader(
        "Choose a fundus image...",
        type=["png", "jpg", "jpeg"],
        help="Upload a retinal fundus image for DR screening"
    )
    
    use_demo = st.checkbox("🎮 Use demo image from test set")
    
    if use_demo:
        test_dir = PROJECT_ROOT / "data" / "aptos" / "test_images"
        if test_dir.exists():
            test_images = sorted(list(test_dir.glob("*.png")))[:10]
            if test_images:
                selected_image = st.selectbox(
                    "Select a test image:",
                    test_images,
                    format_func=lambda x: x.name
                )
                image = cv2.imread(str(selected_image))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                st.warning("No test images found")
                return
        else:
            st.warning("Test images directory not found")
            return
    elif uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        st.info("👆 Please upload an image or use a demo image to get started.")
        
        st.markdown("---")
        st.subheader("📖 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Upload
            Upload a retinal fundus photograph (color image of the back of the eye).
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Process
            The AI applies Ben Graham preprocessing and analyzes the image.
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Results
            Get DR grade prediction with confidence score and Grad-CAM visualization.
            """)
        
        return
    
    # Process the image
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image_rgb, use_container_width=True)
    
    with st.spinner("🔍 Analyzing image..."):
        result = predict_image(image, model, preprocessor, thresholds, device)
    
    with col2:
        st.subheader("🔬 Preprocessed Image")
        st.image(result["processed_image"], use_container_width=True)
    
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    predicted_class = result["predicted_class"]
    info = CLASS_INFO[predicted_class]
    
    render_prediction_card(result, info, predicted_class)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Raw Score", f"{result['raw_value']:.3f}", help="Model's raw regression output (0-4)")
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}", help="Distance from decision boundary")
    
    with col3:
        st.metric("Severity", info["severity"], help="Clinical severity level")
    
    st.subheader("📊 Score Visualization")
    render_score_visualization(result['raw_value'], thresholds, predicted_class)
    
    st.markdown("---")
    st.header("🔍 Explainability (Grad-CAM)")
    
    st.markdown("""
    **Grad-CAM** shows which regions of the image the model focused on when making its prediction.
    - 🔴 **Red/Hot regions**: High focus areas
    - 🔵 **Blue/Cold regions**: Low focus areas
    """)
    
    with st.spinner("Generating Grad-CAM visualization..."):
        heatmap, overlay = generate_gradcam(model, result["tensor"], result["processed_image"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heatmap")
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)
    
    st.markdown("---")
    st.header("🏥 Clinical Interpretation")
    
    if predicted_class == 0:
        st.success("""
        ✅ **No signs of Diabetic Retinopathy detected.**
        
        The retina appears healthy. Continue with regular annual screenings and maintain good glycemic control.
        """)
    elif predicted_class == 1:
        st.info("""
        ⚠️ **Mild Non-Proliferative Diabetic Retinopathy detected.**
        
        Small microaneurysms may be present. Annual follow-up recommended. 
        Focus on blood sugar and blood pressure control.
        """)
    elif predicted_class == 2:
        st.warning("""
        ⚠️ **Moderate Non-Proliferative Diabetic Retinopathy detected.**
        
        Blood vessels show signs of damage. 6-month follow-up recommended.
        Consider referral to an ophthalmologist.
        """)
    elif predicted_class == 3:
        st.error("""
        🚨 **Severe Non-Proliferative Diabetic Retinopathy detected.**
        
        Many blood vessels are blocked. High risk of progression to proliferative DR.
        **Immediate referral to ophthalmologist required.**
        """)
    else:
        st.error("""
        🚨 **Proliferative Diabetic Retinopathy detected.**
        
        New abnormal blood vessels are growing. Risk of severe vision loss.
        **URGENT: Immediate treatment required (laser therapy or anti-VEGF injections).**
        """)


def page_demo():
    """Demo page showing examples from each DR grade."""
    st.header("🎮 Demo: All DR Grades")
    
    st.markdown("""
    This page demonstrates the model's predictions on **real examples from each DR grade**.
    You can see how the model processes different severity levels and what regions it focuses on.
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, thresholds, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the models directory.")
        return
    
    preprocessor = BenGrahamPreprocessor(output_size=512)
    
    # Get sample images
    samples = get_sample_images_by_grade()
    
    if samples is None:
        st.error("Training data not found. Please ensure data/aptos/train.csv and train_images exist.")
        return
    
    st.markdown("---")
    
    # Option to randomize
    if st.button("🔄 Randomize Samples"):
        st.rerun()
    
    # Track predictions for summary
    predictions_summary = []
    
    # Show one example from each grade
    for grade in range(5):
        info = CLASS_INFO[grade]
        
        st.markdown(f"""
        <h2 style="color: {info['color']}; border-bottom: 3px solid {info['color']}; padding-bottom: 0.5rem;">
            Grade {grade}: {info['name']} ({info['severity']} Severity)
        </h2>
        """, unsafe_allow_html=True)
        
        st.markdown(f"*{info['description']}*")
        
        if grade not in samples or not samples[grade]:
            st.warning(f"No sample images available for Grade {grade}")
            continue
        
        # Pick a random sample
        sample_path = random.choice(samples[grade])
        
        if not sample_path.exists():
            st.warning(f"Sample image not found: {sample_path}")
            continue
        
        # Load and process
        image = cv2.imread(str(sample_path))
        if image is None:
            st.warning(f"Could not load image: {sample_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with st.spinner(f"Processing Grade {grade} example..."):
            result = predict_image(image, model, preprocessor, thresholds, device)
            heatmap, overlay = generate_gradcam(model, result["tensor"], result["processed_image"])
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.caption("Original")
            st.image(image_rgb, use_container_width=True)
        
        with col2:
            st.caption("Ben Graham Preprocessed")
            st.image(result["processed_image"], use_container_width=True)
        
        with col3:
            st.caption("Grad-CAM Heatmap")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(heatmap, cmap='jet')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col4:
            st.caption("Model Focus Overlay")
            st.image(overlay, use_container_width=True)
        
        # Prediction info with Ground Truth comparison
        predicted_class = result["predicted_class"]
        pred_info = CLASS_INFO[predicted_class]
        
        # Ground Truth vs Prediction comparison box
        is_correct = predicted_class == grade
        comparison_color = "#c8e6c9" if is_correct else "#ffcdd2"
        comparison_icon = "✅" if is_correct else "❌"
        
        st.markdown(f"""
        <div style="background-color: {comparison_color}; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                <div style="text-align: center; padding: 0.5rem;">
                    <p style="margin: 0; font-size: 0.9rem; color: #666;">🎯 GROUND TRUTH</p>
                    <p style="margin: 0; font-size: 2rem; font-weight: bold; color: {info['color']};">Grade {grade}</p>
                    <p style="margin: 0; font-size: 1.1rem;">{info['name']}</p>
                </div>
                <div style="font-size: 2rem;">{comparison_icon}</div>
                <div style="text-align: center; padding: 0.5rem;">
                    <p style="margin: 0; font-size: 0.9rem; color: #666;">🤖 PREDICTION</p>
                    <p style="margin: 0; font-size: 2rem; font-weight: bold; color: {pred_info['color']};">Grade {predicted_class}</p>
                    <p style="margin: 0; font-size: 1.1rem;">{pred_info['name']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Raw Score", f"{result['raw_value']:.3f}")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col3:
            error = abs(predicted_class - grade)
            if error == 0:
                st.metric("Error", "0 (Correct!)", delta="Perfect", delta_color="normal")
            else:
                st.metric("Error", f"{error} grade(s)", delta=f"-{error}", delta_color="inverse")
        
        # Show if prediction matches
        if predicted_class == grade:
            st.success(f"✅ **Correct prediction!** Model correctly identified this as **{pred_info['name']}** (Grade {grade})")
        elif abs(predicted_class - grade) == 1:
            st.warning(f"⚠️ **Close!** Model predicted **{pred_info['name']}** (Grade {predicted_class}) — off by 1 grade from true label **{info['name']}** (Grade {grade})")
        else:
            st.error(f"❌ **Incorrect!** Model predicted **{pred_info['name']}** (Grade {predicted_class}) — true label is **{info['name']}** (Grade {grade})")
        
        # Track for summary
        predictions_summary.append({
            "true_grade": grade,
            "pred_grade": predicted_class,
            "raw_value": result['raw_value'],
            "correct": predicted_class == grade
        })
        
        st.markdown("---")
    
    # Summary statistics
    st.header("📊 Demo Summary")
    
    if predictions_summary:
        correct_count = sum(1 for p in predictions_summary if p['correct'])
        total_count = len(predictions_summary)
        accuracy = correct_count / total_count * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correct Predictions", f"{correct_count}/{total_count}")
        
        with col2:
            st.metric("Accuracy", f"{accuracy:.0f}%")
        
        with col3:
            avg_error = sum(abs(p['pred_grade'] - p['true_grade']) for p in predictions_summary) / total_count
            st.metric("Average Error", f"{avg_error:.2f} grades")
        
        # Confusion-like display
        st.subheader("Prediction Results Table")
        
        results_df = pd.DataFrame([
            {
                "True Grade": f"Grade {p['true_grade']} ({CLASS_INFO[p['true_grade']]['name']})",
                "Predicted": f"Grade {p['pred_grade']} ({CLASS_INFO[p['pred_grade']]['name']})",
                "Raw Score": f"{p['raw_value']:.3f}",
                "Result": "✅ Correct" if p['correct'] else f"❌ Off by {abs(p['pred_grade'] - p['true_grade'])}"
            }
            for p in predictions_summary
        ])
        st.table(results_df)
    
    st.markdown("---")
    
    # Model info
    st.header("📋 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimized Thresholds")
        threshold_df = pd.DataFrame({
            "Transition": ["Grade 0 → 1", "Grade 1 → 2", "Grade 2 → 3", "Grade 3 → 4"],
            "Threshold": [f"{t:.4f}" for t in thresholds]
        })
        st.table(threshold_df)
    
    with col2:
        st.subheader("Model Performance")
        st.markdown("""
        - **Architecture**: EfficientNet-B5 Noisy Student
        - **Pooling**: Generalized Mean (GeM)
        - **Head**: Regression with threshold optimization
        - **Best QWK**: ~0.87
        - **Image Size**: 456×456
        """)


def page_learn():
    """Educational page about the model and DR."""
    st.header("📚 Learn: How It Works")
    
    # Model Architecture
    st.subheader("🏗️ Model Architecture")
    
    st.markdown("""
    Our model uses a state-of-the-art deep learning architecture optimized for 
    Diabetic Retinopathy detection:
    """)
    
    st.code("""
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                         MODEL ARCHITECTURE                                │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  INPUT IMAGE (456×456×3)                                                 │
    │         │                                                                │
    │         ▼                                                                │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  BACKBONE: EfficientNet-B5 (Noisy Student)                      │    │
    │  │  - 30M parameters, 456×456 input                                │    │
    │  │  - Compound scaling (width=1.6, depth=2.2, resolution=456)      │    │
    │  │  - Pre-trained with semi-supervised learning on ImageNet        │    │
    │  │  Output: Feature Maps (2048 channels)                           │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │         │                                                                │
    │         ▼                                                                │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  GeM POOLING (Generalized Mean Pooling)                         │    │
    │  │  - Formula: GeM(x) = (1/n × Σ x_i^p)^(1/p)                      │    │
    │  │  - Learnable p parameter (initialized at 3.0)                   │    │
    │  │  Output: 2048-dim vector                                        │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │         │                                                                │
    │         ▼                                                                │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  REGRESSION HEAD                                                 │    │
    │  │  - Linear(2048 → 512) + BatchNorm + ReLU + Dropout(0.5)         │    │
    │  │  - Linear(512 → 1)                                               │    │
    │  │  Output: Single regression value [0, 4]                         │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    """, language=None)
    
    st.markdown("---")
    
    # Ben Graham Preprocessing
    st.subheader("🔬 Ben Graham Preprocessing")
    
    st.markdown("""
    Named after **Ben Graham** who developed this technique for the 2015 Kaggle 
    Diabetic Retinopathy Detection competition (he won 1st place!).
    
    The technique normalizes lighting variations and enhances pathological features:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Steps:**
        1. **Crop to Square**: Find fundus region and crop with 5% padding
        2. **Gaussian Blur Subtraction**: 
           - Formula: `I_out = α × I_original + β × Gaussian(I_original) + γ`
           - Default: α=4, β=-4, γ=128
        3. **Circular Mask**: Remove edge artifacts
        """)
    
    with col2:
        st.markdown("""
        **Why it works:**
        - Retinal images have inconsistent lighting
        - Lesions are local features (high frequency)
        - Background is global (low frequency)
        - Subtracting blur removes illumination variations
        - Enhances microaneurysms, hemorrhages, exudates
        """)
    
    st.markdown("---")
    
    # Why Regression
    st.subheader("📈 Why Regression Instead of Classification?")
    
    st.markdown("""
    DR grades are **ORDINAL** (ordered): 0 < 1 < 2 < 3 < 4
    
    | Approach | Pros | Cons |
    |----------|------|------|
    | **Classification** | Simple, direct class probabilities | Loses ordinal information, all errors equal |
    | **Regression** | Preserves ordinal nature, threshold optimization | Needs post-processing |
    
    With regression:
    - Model learns Grade 2 is between Grade 1 and 3
    - Prediction of 1.9 is closer to Grade 2 than Grade 0
    - We can optimize thresholds post-training using QWK metric
    """)
    
    st.markdown("---")
    
    # QWK Metric
    st.subheader("📊 Quadratic Weighted Kappa (QWK)")
    
    st.markdown("""
    The main evaluation metric for DR detection:
    
    - Takes into account the **ordinal nature** of grades
    - **Penalizes large errors more** than small errors
    - Score of 1.0 = perfect agreement
    - Score of 0.0 = random agreement
    
    **Our model achieves QWK ≈ 0.87** (state-of-the-art range!)
    """)
    
    st.latex(r"\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}")
    
    st.markdown("Where $w_{ij} = (i-j)^2$ (quadratic weights)")
    
    st.markdown("---")
    
    # Grad-CAM
    st.subheader("🔍 Grad-CAM Explainability")
    
    st.markdown("""
    **Gradient-weighted Class Activation Mapping** helps us understand what the model sees:
    
    1. Forward pass: Get prediction and feature maps
    2. Backward pass: Compute gradients of prediction w.r.t. feature maps
    3. Weight feature maps by gradient importance
    4. Sum and apply ReLU to get heatmap
    
    **Interpretation:**
    - 🔴 **Red/Hot**: Model focuses heavily here
    - 🔵 **Blue/Cold**: Model ignores these areas
    
    **For DR, model should focus on:**
    - Microaneurysms (tiny red dots)
    - Hemorrhages (larger red spots)
    - Hard exudates (yellow spots)
    - Cotton wool spots (white fluffy patches)
    - Neovascularization (new blood vessels)
    """)
    
    st.markdown("---")
    
    # DR Grades
    st.subheader("👁️ Understanding DR Grades")
    
    for grade, info in CLASS_INFO.items():
        with st.expander(f"Grade {grade}: {info['name']} ({info['severity']} Severity)"):
            st.markdown(f"""
            **Description:** {info['description']}
            
            **Clinical Recommendation:** {info['recommendation']}
            """)


def main():
    # Header
    st.markdown('<p class="main-header">👁️ Diabetic Retinopathy Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Retinal Screening using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🏥 Screening", "🎮 Demo", "📚 Learn"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model to screen retinal fundus images 
        for signs of **Diabetic Retinopathy (DR)**.
        
        **Model Architecture:**
        - EfficientNet-B5 backbone
        - GeM Pooling
        - Regression head with optimized thresholds
        
        **Performance:**
        - QWK Score: ~0.87
        - State-of-the-art range
        """)
        
        st.divider()
        
        st.header("🎯 DR Grades")
        for grade, info in CLASS_INFO.items():
            color = info["color"]
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 4px solid {color}; background-color: {color}22;">
                <strong>Grade {grade}: {info['name']}</strong><br>
                <small>{info['severity']} severity</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for **screening purposes only** and should not replace 
        professional medical diagnosis. Always consult an ophthalmologist 
        for proper evaluation.
        """)
    
    # Route to selected page
    if page == "🏥 Screening":
        page_screening()
    elif page == "🎮 Demo":
        page_demo()
    elif page == "📚 Learn":
        page_learn()


if __name__ == "__main__":
    main()
