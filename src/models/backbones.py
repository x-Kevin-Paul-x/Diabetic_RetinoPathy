"""
Model Backbones for Diabetic Retinopathy Detection

This module provides backbone networks using the timm library:
- EfficientNet family (B0-B7, with Noisy Student weights)
- ResNet family (as baselines)
- Vision Transformers (ViT, DeiT)
- Hybrid models

Also includes custom pooling layers (GeM, Concat).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    
    GeM is a learnable pooling layer that generalizes between average and max pooling.
    As p → 1, it behaves like average pooling.
    As p → ∞, it behaves like max pooling.
    
    Formula: f = (1/|Ω| * Σ x^p)^(1/p)
    
    Reference: Fine-tuning CNN Image Retrieval with No Human Annotation (Radenović et al., 2018)
    """
    
    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        trainable: bool = True,
    ):
        """
        Initialize GeM pooling.
        
        Args:
            p: Initial power parameter (3.0 is a good starting point)
            eps: Small constant for numerical stability
            trainable: Whether p should be learned
        """
        super().__init__()
        
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Pooled tensor [B, C]
        """
        # Clamp values for numerical stability
        x = x.clamp(min=self.eps)
        
        # GeM: (mean(x^p))^(1/p)
        pooled = F.adaptive_avg_pool2d(x.pow(self.p), 1)
        pooled = pooled.pow(1.0 / self.p)
        
        return pooled.flatten(1)
    
    def __repr__(self):
        return f"GeM(p={self.p.item():.2f}, trainable={isinstance(self.p, nn.Parameter)})"


class ConcatPool(nn.Module):
    """
    Concatenate Average and Max pooling.
    
    Doubles the feature dimension but captures both global statistics
    and peak activations.
    """
    
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply concat pooling.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Pooled tensor [B, 2*C]
        """
        avg = self.avg_pool(x).flatten(1)
        max_p = self.max_pool(x).flatten(1)
        return torch.cat([avg, max_p], dim=1)


class BackboneFactory:
    """
    Factory for creating backbone networks with custom pooling.
    """
    
    EFFICIENTNET_FEATURES = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
        "tf_efficientnet_b5_ns": 2048,  # Noisy Student
        "tf_efficientnet_b6_ns": 2304,
        "tf_efficientnet_b7_ns": 2560,
    }
    
    RESNET_FEATURES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }
    
    @classmethod
    def get_backbone(
        cls,
        name: str,
        pretrained: bool = True,
        pooling_type: str = "gem",
        gem_p: float = 3.0,
        gem_trainable: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> Tuple[nn.Module, int]:
        """
        Create a backbone network with custom pooling.
        
        Args:
            name: Model name (e.g., "tf_efficientnet_b5_ns")
            pretrained: Whether to use pretrained weights
            pooling_type: "gem", "avg", "max", or "concat"
            gem_p: Initial p value for GeM pooling
            gem_trainable: Whether GeM p is learnable
            drop_rate: Dropout rate
            drop_path_rate: Drop path (stochastic depth) rate
            
        Returns:
            Tuple of (backbone_module, output_features)
        """
        # Create model without classifier
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove pooling (we'll add our own)
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Get feature dimension
        if name in cls.EFFICIENTNET_FEATURES:
            num_features = cls.EFFICIENTNET_FEATURES[name]
        elif name in cls.RESNET_FEATURES:
            num_features = cls.RESNET_FEATURES[name]
        else:
            # Try to infer from model
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = model(dummy)
                num_features = features.shape[1]
            logger.info(f"Inferred feature dimension: {num_features}")
        
        # Create pooling layer
        if pooling_type == "gem":
            pool = GeM(p=gem_p, trainable=gem_trainable)
            out_features = num_features
        elif pooling_type == "avg":
            pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            out_features = num_features
        elif pooling_type == "max":
            pool = nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten()
            )
            out_features = num_features
        elif pooling_type == "concat":
            pool = ConcatPool()
            out_features = num_features * 2
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Combine backbone and pooling
        backbone = nn.Sequential(model, pool)
        
        logger.info(f"Created backbone: {name} with {pooling_type} pooling, "
                   f"output features: {out_features}")
        
        return backbone, out_features
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List available model names."""
        return list(cls.EFFICIENTNET_FEATURES.keys()) + list(cls.RESNET_FEATURES.keys())


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone wrapper with GeM pooling.
    
    This is the recommended backbone for DR detection.
    """
    
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b5_ns",
        pretrained: bool = True,
        pooling: str = "gem",
        gem_p: float = 3.0,
        gem_trainable: bool = True,
        drop_rate: float = 0.4,
        drop_path_rate: float = 0.2,
    ):
        """
        Initialize EfficientNet backbone.
        
        Args:
            model_name: EfficientNet variant
            pretrained: Use ImageNet pretrained weights
            pooling: Pooling type ("gem", "avg", "max", "concat")
            gem_p: GeM power parameter
            gem_trainable: Whether GeM p is learnable
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.backbone, self.num_features = BackboneFactory.get_backbone(
            name=model_name,
            pretrained=pretrained,
            pooling_type=pooling,
            gem_p=gem_p,
            gem_trainable=gem_trainable,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Feature vector [B, num_features]
        """
        return self.backbone(x)
    
    def get_features_for_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get features before and after pooling (for Grad-CAM).
        
        Args:
            x: Input images
            
        Returns:
            Tuple of (feature_maps [B, C, H, W], pooled_features [B, C])
        """
        # Get feature maps before pooling
        feature_maps = self.backbone[0](x)
        
        # Apply pooling
        pooled = self.backbone[1](feature_maps)
        
        return feature_maps, pooled


def create_backbone(config: Dict) -> Tuple[nn.Module, int]:
    """
    Create backbone from configuration dictionary.
    
    Args:
        config: Model configuration with 'backbone', 'pooling', etc.
        
    Returns:
        Tuple of (backbone_module, output_features)
    """
    backbone_name = config.get("backbone", "tf_efficientnet_b5_ns")
    pooling_config = config.get("pooling", {})
    arch_config = config.get("architecture", {})
    
    return BackboneFactory.get_backbone(
        name=backbone_name,
        pretrained=arch_config.get("pretrained", True),
        pooling_type=pooling_config.get("type", "gem"),
        gem_p=pooling_config.get("gem_p", 3.0),
        gem_trainable=pooling_config.get("gem_trainable", True),
        drop_rate=arch_config.get("drop_rate", 0.0),
        drop_path_rate=arch_config.get("drop_path_rate", 0.0),
    )


if __name__ == "__main__":
    # Test backbones
    print("Available models:", BackboneFactory.list_available_models())
    
    # Test EfficientNet-B5 with GeM
    backbone = EfficientNetBackbone(
        model_name="tf_efficientnet_b5_ns",
        pretrained=True,
        pooling="gem",
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 456, 456)
    features = backbone(x)
    print(f"EfficientNet-B5 output shape: {features.shape}")
    
    # Test CAM features
    feature_maps, pooled = backbone.get_features_for_cam(x)
    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"Pooled features shape: {pooled.shape}")
