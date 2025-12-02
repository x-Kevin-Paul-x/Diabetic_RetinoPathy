"""
Classification and Regression Heads for DR Detection

This module provides:
1. RegressionHead: For ordinal regression with continuous output
2. ClassificationHead: For standard multi-class classification
3. OrdinalHead: For ordinal classification (CORAL-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RegressionHead(nn.Module):
    """
    Regression head for DR grading.
    
    Outputs a single scalar prediction that can be converted to
    discrete classes using optimized thresholds.
    
    This approach naturally enforces ordinal relationships via MSE loss.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        output_range: Tuple[float, float] = (0.0, 4.0),
        clip_output: bool = True,
    ):
        """
        Initialize regression head.
        
        Args:
            in_features: Input feature dimension from backbone
            hidden_dims: List of hidden layer dimensions (e.g., [512])
            dropout: Dropout probability
            output_range: Expected output range for clipping
            clip_output: Whether to clip output to range during inference
        """
        super().__init__()
        
        self.output_range = output_range
        self.clip_output = clip_output
        
        # Build MLP layers
        layers = []
        prev_dim = in_features
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
        
        # Initialize final layer
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, in_features]
            
        Returns:
            Regression output [B, 1]
        """
        out = self.head(x)
        
        if self.clip_output and not self.training:
            out = torch.clamp(out, self.output_range[0], self.output_range[1])
        
        return out.squeeze(-1)


class ClassificationHead(nn.Module):
    """
    Classification head for DR grading.
    
    Outputs class logits for 5-class classification.
    Use with Cross-Entropy or Focal Loss.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize classification head.
        
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes (5 for DR)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = in_features
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, in_features]
            
        Returns:
            Class logits [B, num_classes]
        """
        return self.head(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


class OrdinalHead(nn.Module):
    """
    Ordinal regression head (CORAL-style).
    
    Transforms classification into K-1 binary tasks:
    "Is grade > 0?", "Is grade > 1?", etc.
    
    This enforces ordinal relationships while allowing non-uniform
    distances between classes.
    
    Reference: Rank consistent ordinal regression for neural networks (Cao et al., 2020)
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize ordinal head.
        
        Args:
            in_features: Input feature dimension
            num_classes: Number of ordinal classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        # Feature projection
        layers = []
        prev_dim = in_features
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Shared feature dimension to single output
        self.fc = nn.Linear(prev_dim if hidden_dims else in_features, 1, bias=False)
        
        # Ordinal thresholds (K-1 biases)
        self.thresholds = nn.Parameter(torch.zeros(self.num_thresholds))
        
        # Initialize thresholds in ascending order
        with torch.no_grad():
            self.thresholds.copy_(torch.linspace(-2, 2, self.num_thresholds))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, in_features]
            
        Returns:
            Ordinal logits [B, num_thresholds]
            Each logit represents P(grade > k)
        """
        features = self.features(x)
        f = self.fc(features)  # [B, 1]
        
        # Compute logits for each threshold
        logits = f - self.thresholds  # [B, num_thresholds]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input features
            
        Returns:
            Class probabilities [B, num_classes]
        """
        logits = self.forward(x)
        cumulative_probs = torch.sigmoid(logits)  # P(grade > k)
        
        # Convert cumulative to class probabilities
        # P(grade = k) = P(grade > k-1) - P(grade > k)
        ones = torch.ones(x.size(0), 1, device=x.device)
        zeros = torch.zeros(x.size(0), 1, device=x.device)
        
        cumulative = torch.cat([ones, cumulative_probs, zeros], dim=1)
        probs = cumulative[:, :-1] - cumulative[:, 1:]
        
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss.
        
        Args:
            logits: Ordinal logits [B, num_thresholds]
            targets: Target class indices [B]
            
        Returns:
            Loss scalar
        """
        # Create binary targets for each threshold
        # For target = k, binary_targets[j] = 1 if j < k else 0
        binary_targets = torch.zeros_like(logits)
        for j in range(self.num_thresholds):
            binary_targets[:, j] = (targets > j).float()
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='mean')
        
        return loss


class MultiTaskHead(nn.Module):
    """
    Multi-task head combining regression and classification.
    
    Useful for auxiliary losses or ensemble predictions.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared feature layers
        layers = []
        prev_dim = in_features
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Task-specific heads
        self.regression_head = nn.Linear(prev_dim if hidden_dims else in_features, 1)
        self.classification_head = nn.Linear(prev_dim if hidden_dims else in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (regression_output, classification_logits)
        """
        features = self.shared(x)
        
        reg_out = self.regression_head(features).squeeze(-1)
        cls_out = self.classification_head(features)
        
        return reg_out, cls_out


def create_head(
    head_type: str,
    in_features: int,
    num_classes: int = 5,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create head from config.
    
    Args:
        head_type: "regression", "classification", "ordinal", or "multitask"
        in_features: Input feature dimension
        num_classes: Number of classes
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        **kwargs: Additional arguments for specific head types
        
    Returns:
        Head module
    """
    if head_type == "regression":
        return RegressionHead(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_range=kwargs.get("output_range", (0.0, 4.0)),
            clip_output=kwargs.get("clip_output", True),
        )
    elif head_type == "classification":
        return ClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif head_type == "ordinal":
        return OrdinalHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif head_type == "multitask":
        return MultiTaskHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


if __name__ == "__main__":
    # Test heads
    batch_size = 4
    in_features = 2048
    
    x = torch.randn(batch_size, in_features)
    targets = torch.tensor([0, 1, 2, 3])
    
    # Test regression head
    reg_head = RegressionHead(in_features, hidden_dims=[512])
    reg_out = reg_head(x)
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Regression output: {reg_out}")
    
    # Test classification head
    cls_head = ClassificationHead(in_features, num_classes=5, hidden_dims=[512])
    cls_out = cls_head(x)
    print(f"\nClassification output shape: {cls_out.shape}")
    print(f"Predictions: {cls_head.predict(x)}")
    
    # Test ordinal head
    ord_head = OrdinalHead(in_features, num_classes=5, hidden_dims=[512])
    ord_out = ord_head(x)
    print(f"\nOrdinal output shape: {ord_out.shape}")
    print(f"Ordinal predictions: {ord_head.predict(x)}")
    print(f"Ordinal probabilities:\n{ord_head.predict_proba(x)}")
    
    # Test ordinal loss
    loss = ord_head.compute_loss(ord_out, targets)
    print(f"Ordinal loss: {loss.item():.4f}")
