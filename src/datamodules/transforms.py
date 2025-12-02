"""
Data Augmentation Transforms for Diabetic Retinopathy Detection

This module provides augmentation pipelines using Albumentations.
Augmentations are designed to be anatomically valid for retinal images:
- Geometric: rotation, flip (fully invariant for fundus images)
- Photometric: moderate color/brightness changes to preserve pathology colors
- Dropout: CoarseDropout to improve robustness
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Optional, Tuple
import cv2


def get_train_transforms(
    image_size: int = 456,
    # Geometric augmentations
    horizontal_flip: float = 0.5,
    vertical_flip: float = 0.5,
    rotate_limit: int = 360,
    rotate_prob: float = 0.5,
    scale_limit: float = 0.2,
    # Photometric augmentations
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    saturation_range: Tuple[float, float] = (0.8, 1.2),
    hue_shift_limit: int = 10,
    # Dropout
    coarse_dropout_prob: float = 0.3,
    coarse_dropout_max_holes: int = 8,
    coarse_dropout_max_size: float = 0.1,
    # Normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size
        horizontal_flip: Probability of horizontal flip
        vertical_flip: Probability of vertical flip
        rotate_limit: Maximum rotation angle (360 for full rotation)
        rotate_prob: Probability of applying rotation
        scale_limit: Scale range (0.2 means 0.8x to 1.2x)
        brightness_limit: Brightness change range
        contrast_limit: Contrast change range
        saturation_range: Saturation multiplier range
        hue_shift_limit: Maximum hue shift (keep small for pathology preservation)
        coarse_dropout_prob: Probability of applying CoarseDropout
        coarse_dropout_max_holes: Maximum number of dropout holes
        coarse_dropout_max_size: Maximum hole size as fraction of image
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        
    Returns:
        Albumentations Compose transform
    """
    # Calculate dropout hole size in pixels
    max_hole_size = int(image_size * coarse_dropout_max_size)
    min_hole_size = max(1, max_hole_size // 4)
    
    transforms = A.Compose([
        # Resize first
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Geometric augmentations (safe for retinal images)
        A.HorizontalFlip(p=horizontal_flip),
        A.VerticalFlip(p=vertical_flip),
        A.Rotate(
            limit=rotate_limit,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=rotate_prob
        ),
        A.RandomScale(scale_limit=scale_limit, p=0.5),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1.0
        ),
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        
        # Photometric augmentations (moderate to preserve pathology)
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=int((saturation_range[1] - 1.0) * 100),
            val_shift_limit=int(brightness_limit * 100),
            p=0.5
        ),
        
        # Optional: Gaussian blur for robustness
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Coarse Dropout (cutout-style regularization)
        A.CoarseDropout(
            num_holes_range=(1, coarse_dropout_max_holes),
            hole_height_range=(min_hole_size, max_hole_size),
            hole_width_range=(min_hole_size, max_hole_size),
            fill=0,
            p=coarse_dropout_prob
        ),
        
        # Normalize and convert to tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return transforms


def get_val_transforms(
    image_size: int = 456,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get validation/test augmentation pipeline (minimal transforms).
    
    Args:
        image_size: Target image size
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_tta_transforms(
    image_size: int = 456,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Dict[str, A.Compose]:
    """
    Get Test-Time Augmentation (TTA) transforms.
    
    Returns multiple transforms to be applied at inference,
    predictions are then averaged.
    
    Args:
        image_size: Target image size
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        
    Returns:
        Dictionary of TTA transforms
    """
    base = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    
    tta_transforms = {
        "none": A.Compose(base),
        "hflip": A.Compose([
            A.HorizontalFlip(p=1.0),
            *base
        ]),
        "vflip": A.Compose([
            A.VerticalFlip(p=1.0),
            *base
        ]),
        "hflip_vflip": A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            *base
        ]),
        "rotate90": A.Compose([
            A.Rotate(limit=(90, 90), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0),
            *base
        ]),
        "rotate180": A.Compose([
            A.Rotate(limit=(180, 180), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0),
            *base
        ]),
        "rotate270": A.Compose([
            A.Rotate(limit=(270, 270), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0),
            *base
        ]),
    }
    
    return tta_transforms


def get_light_transforms(
    image_size: int = 456,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get light augmentation pipeline (for EyePACS pretraining or quick experiments).
    
    Args:
        image_size: Target image size
        mean: ImageNet mean
        std: ImageNet std
        
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_transforms_from_config(config: Dict, mode: str = "train") -> A.Compose:
    """
    Create transforms from Hydra config dictionary.
    
    Args:
        config: Dataset configuration containing augmentation parameters
        mode: "train", "val", or "test"
        
    Returns:
        Albumentations Compose transform
    """
    image_size = config.get("input_size", 456)
    aug_config = config.get("augmentation", {})
    
    if mode == "train":
        return get_train_transforms(
            image_size=image_size,
            horizontal_flip=aug_config.get("horizontal_flip", 0.5),
            vertical_flip=aug_config.get("vertical_flip", 0.5),
            rotate_limit=aug_config.get("rotate_limit", 360),
            rotate_prob=aug_config.get("rotate_prob", 0.5),
            scale_limit=aug_config.get("scale_limit", 0.2),
            brightness_limit=aug_config.get("brightness_limit", 0.2),
            contrast_limit=aug_config.get("contrast_limit", 0.2),
            saturation_range=tuple(aug_config.get("saturation_range", [0.8, 1.2])),
            hue_shift_limit=aug_config.get("hue_shift_limit", 10),
            coarse_dropout_prob=aug_config.get("coarse_dropout_prob", 0.3),
            coarse_dropout_max_holes=aug_config.get("coarse_dropout_max_holes", 8),
            coarse_dropout_max_size=aug_config.get("coarse_dropout_max_size", 0.1),
        )
    else:
        return get_val_transforms(image_size=image_size)


class MixUp:
    """
    MixUp augmentation for batch-level mixing.
    
    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 0.4, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying MixUp
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(self, images, labels):
        """
        Apply MixUp to a batch.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B] for regression or [B, C] for classification
            
        Returns:
            Mixed images and labels
        """
        if np.random.rand() > self.p:
            return images, labels
        
        batch_size = images.size(0)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels (for regression, linear interpolation works)
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels


class CutMix:
    """
    CutMix augmentation for batch-level mixing.
    
    Reference: CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying CutMix
        """
        self.alpha = alpha
        self.p = p
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box."""
        W = size[3]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, images, labels):
        """
        Apply CutMix to a batch.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels
            
        Returns:
            Mixed images and labels
        """
        if np.random.rand() > self.p:
            return images, labels
        
        batch_size = images.size(0)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Get random bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels


if __name__ == "__main__":
    # Test transforms
    import torch
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test train transforms
    train_tf = get_train_transforms(image_size=456)
    result = train_tf(image=dummy_image)
    print(f"Train transform output shape: {result['image'].shape}")
    
    # Test val transforms
    val_tf = get_val_transforms(image_size=456)
    result = val_tf(image=dummy_image)
    print(f"Val transform output shape: {result['image'].shape}")
    
    # Test TTA transforms
    tta_tf = get_tta_transforms(image_size=456)
    print(f"TTA transforms: {list(tta_tf.keys())}")
    
    # Test MixUp
    mixup = MixUp(alpha=0.4, p=1.0)
    batch_images = torch.randn(4, 3, 456, 456)
    batch_labels = torch.tensor([0.0, 1.0, 2.0, 3.0])
    mixed_images, mixed_labels = mixup(batch_images, batch_labels)
    print(f"MixUp: images shape {mixed_images.shape}, labels {mixed_labels}")
