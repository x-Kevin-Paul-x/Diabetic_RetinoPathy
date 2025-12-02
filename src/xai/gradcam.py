"""
Grad-CAM and Explainability for Diabetic Retinopathy Detection

This module provides visualization tools to explain model predictions:
1. Grad-CAM: Gradient-weighted Class Activation Mapping
2. Integrated Gradients: Pixel-level attribution
3. Visualization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates heatmaps highlighting regions important for the model's prediction.
    
    Reference: Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The model to explain
            target_layer: Target layer for CAM (uses last conv layer if None)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer if not specified
        if target_layer is None:
            # For EfficientNet, use the last conv block
            if hasattr(model, 'backbone'):
                # Navigate to the backbone's feature extractor
                backbone = model.backbone.backbone[0]  # The timm model
                # Get the last convolutional module
                for name, module in backbone.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
            else:
                # Generic: find last conv layer
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
        
        self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image [1, 3, H, W]
            target_class: Target class index (uses predicted class if None)
            
        Returns:
            Heatmap array [H, W] in range [0, 1]
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output types
        if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
            # Regression output
            score = output.squeeze()
        else:
            # Classification output
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            score = output[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image [H, W, 3] in range [0, 255]
            heatmap: Grad-CAM heatmap [h, w] in range [0, 1]
            alpha: Blending factor
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image [H, W, 3] in range [0, 255]
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Convert image to uint8 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


class IntegratedGradients:
    """
    Integrated Gradients for pixel-level attribution.
    
    Reference: Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
    ):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: The model to explain
            n_steps: Number of interpolation steps
        """
        self.model = model
        self.model.eval()
        self.n_steps = n_steps
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute integrated gradients.
        
        Args:
            input_tensor: Input image [1, 3, H, W]
            baseline: Baseline image (black image if None)
            target_class: Target class (uses predicted class if None)
            
        Returns:
            Attribution map [H, W, 3]
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps, device=input_tensor.device)
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        
        # Forward pass for all interpolated inputs
        outputs = []
        for i in range(self.n_steps):
            output = self.model(interpolated[i:i+1])
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=0)
        
        # Get target scores
        if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
            scores = outputs.squeeze()
        else:
            if target_class is None:
                with torch.no_grad():
                    target_class = self.model(input_tensor).argmax(dim=1).item()
            scores = outputs[:, target_class]
        
        # Compute gradients
        gradients = []
        for i in range(self.n_steps):
            self.model.zero_grad()
            interpolated.grad = None
            scores[i].backward(retain_graph=True)
            gradients.append(interpolated.grad[i:i+1].clone())
        
        gradients = torch.cat(gradients, dim=0)
        
        # Integrate gradients
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Convert to numpy
        ig = integrated_gradients.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Take absolute value and sum across channels
        ig = np.abs(ig).sum(axis=-1)
        
        # Normalize
        ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
        
        return ig


def generate_explanation_report(
    model: nn.Module,
    image: np.ndarray,
    image_tensor: torch.Tensor,
    prediction: Union[int, float],
    true_label: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate a comprehensive explanation report for a single prediction.
    
    Args:
        model: Trained model
        image: Original image [H, W, 3]
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        prediction: Model prediction
        true_label: Ground truth label (optional)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam(image_tensor)
    overlay = gradcam.visualize(image, heatmap)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    title = f"Overlay (Pred: {prediction:.2f})" if isinstance(prediction, float) else f"Overlay (Pred: Grade {prediction})"
    if true_label is not None:
        title += f" [True: Grade {true_label}]"
    axes[2].set_title(title)
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def batch_generate_gradcam(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    num_samples: int = 50,
    device: str = "cuda",
):
    """
    Generate Grad-CAM visualizations for multiple samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader providing images and labels
        output_dir: Directory to save visualizations
        num_samples: Maximum number of samples to process
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    gradcam = GradCAM(model)
    
    sample_count = 0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"]
        
        for i in range(images.size(0)):
            if sample_count >= num_samples:
                return
            
            # Get single image
            img_tensor = images[i:i+1]
            target = targets[i].item()
            
            # Generate heatmap
            with torch.enable_grad():
                heatmap = gradcam(img_tensor)
            
            # Get prediction
            with torch.no_grad():
                pred = model(img_tensor).squeeze().item()
            
            # Denormalize image for visualization
            img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + 
                   np.array([0.485, 0.456, 0.406]))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Create overlay
            overlay = gradcam.visualize(img, heatmap)
            
            # Save
            save_path = output_dir / f"sample_{sample_count:04d}_pred{pred:.1f}_true{target}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            sample_count += 1
    
    print(f"Generated {sample_count} Grad-CAM visualizations in {output_dir}")


if __name__ == "__main__":
    # Test Grad-CAM
    import torch
    from src.models import DRModel
    
    # Create dummy model
    model = DRModel(
        backbone_name="efficientnet_b0",
        pretrained=True,
    )
    
    # Create dummy input
    x = torch.randn(1, 3, 456, 456)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam(x)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")
