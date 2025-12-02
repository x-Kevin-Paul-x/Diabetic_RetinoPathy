"""
Diabetic Retinopathy Detection Model

This module provides the main DR detection model combining:
- EfficientNet backbone with GeM pooling
- Regression or classification head
- Loss functions (MSE, Focal, Ordinal)
- PyTorch Lightning integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from omegaconf import DictConfig
import logging

from .backbones import EfficientNetBackbone, create_backbone
from .heads import RegressionHead, ClassificationHead, OrdinalHead, create_head
from ..utils.metrics import (
    quadratic_weighted_kappa,
    RegressionToClassMeter,
    QWKMeter,
    compute_confusion_matrix,
)
from ..utils.threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference: Focal Loss for Dense Object Detection (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [B, C]
            targets: Class indices [B]
            
        Returns:
            Loss scalar
        """
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DRModel(pl.LightningModule):
    """
    Diabetic Retinopathy Detection and Grading Model.
    
    Supports:
    - Regression with threshold optimization (SOTA approach)
    - Classification with Focal Loss
    - Ordinal regression
    - Test-Time Augmentation (TTA)
    - Grad-CAM for explainability
    """
    
    def __init__(
        self,
        # Model architecture
        backbone_name: str = "tf_efficientnet_b5_ns",
        pretrained: bool = True,
        pooling_type: str = "gem",
        gem_p: float = 3.0,
        gem_trainable: bool = True,
        drop_rate: float = 0.4,
        drop_path_rate: float = 0.2,
        # Head configuration
        head_type: str = "regression",  # regression, classification, ordinal
        hidden_dims: List[int] = [512],
        head_dropout: float = 0.3,
        num_classes: int = 5,
        # Loss configuration
        loss_type: str = "smooth_l1",  # smooth_l1, mse, focal, cross_entropy
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[List[float]] = None,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
        max_epochs: int = 30,
        scheduler_type: str = "cosine",
        # Threshold optimization
        optimize_thresholds: bool = True,
        initial_thresholds: List[float] = [0.5, 1.5, 2.5, 3.5],
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Build backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            pooling=pooling_type,
            gem_p=gem_p,
            gem_trainable=gem_trainable,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Build head
        self.head = create_head(
            head_type=head_type,
            in_features=self.backbone.num_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=head_dropout,
        )
        
        # Loss function
        self.loss_fn = self._create_loss_fn()
        
        # Metrics
        self.head_type = head_type
        if head_type == "regression":
            self.train_meter = RegressionToClassMeter(
                num_classes=num_classes,
                thresholds=initial_thresholds
            )
            self.val_meter = RegressionToClassMeter(
                num_classes=num_classes,
                thresholds=initial_thresholds
            )
        else:
            self.train_meter = QWKMeter(num_classes=num_classes)
            self.val_meter = QWKMeter(num_classes=num_classes)
        
        # Threshold optimizer
        self.threshold_optimizer = ThresholdOptimizer(
            num_classes=num_classes,
            initial_thresholds=initial_thresholds
        )
        self.optimized_thresholds = initial_thresholds
        
        # Store validation outputs for threshold optimization
        self.validation_outputs = []
    
    def _create_loss_fn(self) -> nn.Module:
        """Create loss function based on configuration."""
        hp = self.hparams
        
        if hp.loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        elif hp.loss_type == "mse":
            return nn.MSELoss()
        elif hp.loss_type == "focal":
            alpha = None
            if hp.class_weights:
                alpha = torch.tensor(hp.class_weights)
            return FocalLoss(
                alpha=alpha,
                gamma=hp.focal_gamma,
                label_smoothing=hp.label_smoothing
            )
        elif hp.loss_type == "cross_entropy":
            weight = None
            if hp.class_weights:
                weight = torch.tensor(hp.class_weights)
            return nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=hp.label_smoothing
            )
        else:
            raise ValueError(f"Unknown loss type: {hp.loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Predictions (shape depends on head type)
        """
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on head type."""
        if self.head_type == "regression":
            return self.loss_fn(outputs, targets.float())
        elif self.head_type == "ordinal":
            return self.head.compute_loss(outputs, targets)
        else:  # classification
            return self.loss_fn(outputs, targets)
    
    def _get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert outputs to class predictions."""
        if self.head_type == "regression":
            # Use thresholds to convert to classes
            preds = outputs.detach().cpu().numpy()
            classes = np.zeros_like(preds, dtype=int)
            for i, t in enumerate(self.optimized_thresholds):
                classes[preds > t] = i + 1
            return torch.tensor(classes)
        elif self.head_type == "ordinal":
            return self.head.predict(outputs)
        else:  # classification
            return outputs.argmax(dim=-1)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        targets = batch["target"]
        
        outputs = self(images)
        loss = self._compute_loss(outputs, targets)
        
        # Update metrics
        if self.head_type == "regression":
            self.train_meter.update(outputs.detach().cpu(), targets.cpu())
        else:
            preds = self._get_predictions(outputs)
            self.train_meter.update(preds, targets.cpu())
        
        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at end of epoch."""
        qwk = self.train_meter.compute()
        self.log("train_qwk", qwk, prog_bar=True)
        self.train_meter.reset()
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Validation step."""
        images = batch["image"]
        targets = batch["target"]
        
        outputs = self(images)
        loss = self._compute_loss(outputs, targets)
        
        # Store outputs for threshold optimization
        self.validation_outputs.append({
            "outputs": outputs.detach().cpu(),
            "targets": targets.cpu(),
        })
        
        # Update metrics
        if self.head_type == "regression":
            self.val_meter.update(outputs.detach().cpu(), targets.cpu())
        else:
            preds = self._get_predictions(outputs)
            self.val_meter.update(preds, targets.cpu())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        """Compute validation metrics and optimize thresholds."""
        # Compute QWK with current thresholds
        qwk = self.val_meter.compute()
        
        # Optimize thresholds if regression
        if self.head_type == "regression" and self.hparams.optimize_thresholds:
            all_outputs = torch.cat([x["outputs"] for x in self.validation_outputs])
            all_targets = torch.cat([x["targets"] for x in self.validation_outputs])
            
            # Optimize thresholds
            opt_thresholds, opt_qwk = self.threshold_optimizer.optimize(
                all_outputs.numpy(),
                all_targets.numpy(),
                verbose=False
            )
            
            self.optimized_thresholds = opt_thresholds
            self.val_meter.set_thresholds(opt_thresholds)
            
            self.log("val_qwk_optimized", opt_qwk, prog_bar=True)
            logger.info(f"Optimized thresholds: {opt_thresholds}, QWK: {opt_qwk:.4f}")
        
        self.log("val_qwk", qwk, prog_bar=True)
        
        # Reset
        self.val_meter.reset()
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Differential learning rates
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())
        
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.learning_rate * 0.1},
            {"params": head_params, "lr": self.hparams.learning_rate},
        ], weight_decay=self.hparams.weight_decay)
        
        # Scheduler
        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
        elif self.hparams.scheduler_type == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.hparams.learning_rate * 0.01
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_qwk"},
            }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get feature maps for Grad-CAM."""
        return self.backbone.get_features_for_cam(x)
    
    @classmethod
    def from_config(cls, config: DictConfig) -> "DRModel":
        """Create model from Hydra config."""
        model_cfg = config.model
        loss_cfg = config.loss
        training_cfg = config.training
        dataset_cfg = config.dataset
        
        return cls(
            # Model
            backbone_name=model_cfg.backbone,
            pretrained=model_cfg.architecture.pretrained,
            pooling_type=model_cfg.pooling.type,
            gem_p=model_cfg.pooling.get("gem_p", 3.0),
            gem_trainable=model_cfg.pooling.get("gem_trainable", True),
            drop_rate=model_cfg.architecture.drop_rate,
            drop_path_rate=model_cfg.architecture.drop_path_rate,
            # Head
            head_type=model_cfg.head.type,
            hidden_dims=list(model_cfg.head.hidden_dims),
            head_dropout=model_cfg.head.dropout,
            num_classes=dataset_cfg.num_classes,
            # Loss
            loss_type=loss_cfg.primary.type,
            class_weights=list(dataset_cfg.class_weights) if dataset_cfg.get("class_weights") else None,
            # Training
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.optimizer.weight_decay,
            warmup_epochs=training_cfg.scheduler.warmup_epochs,
            max_epochs=training_cfg.epochs,
            scheduler_type=training_cfg.scheduler.name,
            # Threshold
            optimize_thresholds=loss_cfg.threshold_optimization.enabled,
            initial_thresholds=list(loss_cfg.threshold_optimization.initial_thresholds),
        )


if __name__ == "__main__":
    # Test model
    model = DRModel(
        backbone_name="efficientnet_b0",  # Smaller for testing
        pretrained=True,
        head_type="regression",
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 456, 456)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")
    
    # Test with targets
    targets = torch.tensor([1.0, 3.0])
    loss = model._compute_loss(y, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test predictions
    preds = model._get_predictions(y)
    print(f"Predictions: {preds}")
