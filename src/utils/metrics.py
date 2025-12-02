"""
Metrics for Diabetic Retinopathy Detection

This module implements:
1. Quadratic Weighted Kappa (QWK) - Primary evaluation metric
2. Confusion matrix utilities
3. Per-class metrics
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union, List
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report


def quadratic_weighted_kappa(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    num_classes: int = 5,
) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK).
    
    QWK measures agreement between predicted and actual ratings,
    with quadratic penalty for distant misclassifications.
    
    Weight matrix: W[i,j] = (i - j)^2 / (N - 1)^2
    
    Args:
        y_true: True labels (0-4 for DR grading)
        y_pred: Predicted labels (0-4)
        num_classes: Number of classes (5 for DR)
        
    Returns:
        QWK score in range [-1, 1], where 1 is perfect agreement
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    # Clip to valid range
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)
    
    # Use sklearn's implementation with quadratic weights
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int = 5,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        normalize: 'true' for row normalization, 'pred' for column, 'all' for total
        
    Returns:
        Confusion matrix array
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    labels = list(range(num_classes))
    
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def compute_per_class_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute per-class precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    return report


class QWKMeter:
    """
    Online Quadratic Weighted Kappa meter for batch processing.
    
    Accumulates predictions and targets, computes QWK on demand.
    """
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
    
    def update(
        self, 
        preds: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update with new batch of predictions and targets.
        
        Args:
            preds: Predicted labels or regression values
            targets: True labels
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        self.predictions.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())
    
    def compute(self) -> float:
        """
        Compute QWK from accumulated predictions.
        
        Returns:
            QWK score
        """
        if len(self.predictions) == 0:
            return 0.0
        
        return quadratic_weighted_kappa(
            np.array(self.targets),
            np.array(self.predictions),
            self.num_classes
        )
    
    @property
    def value(self) -> float:
        """Alias for compute()."""
        return self.compute()


class RegressionToClassMeter(QWKMeter):
    """
    QWK meter for regression outputs.
    
    Converts continuous predictions to discrete classes using thresholds.
    """
    
    def __init__(
        self, 
        num_classes: int = 5,
        thresholds: Optional[List[float]] = None
    ):
        super().__init__(num_classes)
        
        # Default equidistant thresholds
        if thresholds is None:
            thresholds = [0.5, 1.5, 2.5, 3.5]
        
        self.thresholds = thresholds
    
    def set_thresholds(self, thresholds: List[float]):
        """Update thresholds (e.g., after optimization)."""
        self.thresholds = thresholds
    
    def _convert_to_classes(self, values: np.ndarray) -> np.ndarray:
        """Convert continuous values to discrete classes."""
        classes = np.zeros_like(values, dtype=int)
        for i, threshold in enumerate(self.thresholds):
            classes[values > threshold] = i + 1
        return classes
    
    def update(
        self,
        preds: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update with regression predictions.
        
        Args:
            preds: Continuous regression outputs
            targets: True labels (integers)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Store raw predictions for threshold optimization
        self.predictions.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())
    
    def compute(self) -> float:
        """Compute QWK after converting predictions to classes."""
        if len(self.predictions) == 0:
            return 0.0
        
        preds_array = np.array(self.predictions)
        targets_array = np.array(self.targets)
        
        # Convert to classes
        pred_classes = self._convert_to_classes(preds_array)
        
        return quadratic_weighted_kappa(
            targets_array.astype(int),
            pred_classes,
            self.num_classes
        )
    
    def compute_with_thresholds(self, thresholds: List[float]) -> float:
        """Compute QWK with specific thresholds (for optimization)."""
        if len(self.predictions) == 0:
            return 0.0
        
        preds_array = np.array(self.predictions)
        targets_array = np.array(self.targets)
        
        # Convert with provided thresholds
        classes = np.zeros_like(preds_array, dtype=int)
        for i, threshold in enumerate(thresholds):
            classes[preds_array > threshold] = i + 1
        
        return quadratic_weighted_kappa(
            targets_array.astype(int),
            classes,
            self.num_classes
        )


def compute_class_weights(
    labels: Union[np.ndarray, List],
    num_classes: int = 5,
    method: str = 'inverse_freq',
) -> np.ndarray:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Array of labels
        num_classes: Number of classes
        method: 'inverse_freq', 'effective_samples', or 'sqrt_inverse_freq'
        
    Returns:
        Array of class weights
    """
    labels = np.asarray(labels)
    
    # Count samples per class
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    
    if method == 'inverse_freq':
        # Inverse frequency
        weights = 1.0 / counts
    elif method == 'sqrt_inverse_freq':
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(counts)
    elif method == 'effective_samples':
        # Effective number of samples (Class-Balanced Loss)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    return weights


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulated predictions
    y_true = np.array([0, 0, 1, 2, 2, 3, 4, 4, 2, 1])
    y_pred = np.array([0, 1, 1, 2, 3, 3, 4, 3, 2, 1])
    
    print("QWK:", quadratic_weighted_kappa(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(compute_confusion_matrix(y_true, y_pred))
    print("\nPer-class metrics:")
    print(compute_per_class_metrics(y_true, y_pred))
    
    # Test meter
    meter = QWKMeter()
    meter.update(y_pred[:5], y_true[:5])
    meter.update(y_pred[5:], y_true[5:])
    print("\nMeter QWK:", meter.compute())
    
    # Test regression meter
    reg_meter = RegressionToClassMeter()
    reg_preds = y_true.astype(float) + np.random.randn(len(y_true)) * 0.3
    reg_meter.update(reg_preds, y_true)
    print("Regression Meter QWK:", reg_meter.compute())
