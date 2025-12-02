"""
Threshold Optimization for Regression-based DR Grading

This module implements threshold optimization to maximize Quadratic Weighted Kappa (QWK)
when converting continuous regression outputs to discrete class predictions.

This is a key technique from the 1st place APTOS 2019 solution.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Union
import torch
import logging

from .metrics import quadratic_weighted_kappa

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes thresholds for converting regression outputs to discrete classes.
    
    Uses scipy.optimize.minimize to find thresholds that maximize QWK.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        initial_thresholds: Optional[List[float]] = None,
        method: str = 'nelder-mead',
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            num_classes: Number of classes (5 for DR)
            initial_thresholds: Starting thresholds for optimization
            method: Optimization method ('nelder-mead', 'powell', 'L-BFGS-B')
        """
        self.num_classes = num_classes
        self.method = method
        
        # Initialize with equidistant thresholds
        if initial_thresholds is None:
            self.initial_thresholds = [i + 0.5 for i in range(num_classes - 1)]
        else:
            self.initial_thresholds = list(initial_thresholds)
        
        self.optimized_thresholds = None
    
    def _apply_thresholds(
        self,
        predictions: np.ndarray,
        thresholds: List[float]
    ) -> np.ndarray:
        """
        Convert continuous predictions to discrete classes using thresholds.
        
        Args:
            predictions: Continuous values (e.g., 0.0 to 4.0)
            thresholds: List of K-1 thresholds for K classes
            
        Returns:
            Discrete class predictions (0 to K-1)
        """
        classes = np.zeros(len(predictions), dtype=int)
        for i, threshold in enumerate(thresholds):
            classes[predictions > threshold] = i + 1
        return classes
    
    def _objective(
        self,
        thresholds: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Objective function: negative QWK (for minimization).
        
        Args:
            thresholds: Current threshold values
            predictions: Continuous predictions
            targets: True labels
            
        Returns:
            Negative QWK score (we minimize this)
        """
        # Ensure thresholds are sorted
        thresholds = np.sort(thresholds)
        
        # Convert to classes
        pred_classes = self._apply_thresholds(predictions, thresholds)
        
        # Compute QWK (return negative for minimization)
        qwk = quadratic_weighted_kappa(targets, pred_classes, self.num_classes)
        
        return -qwk
    
    def optimize(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        verbose: bool = True
    ) -> Tuple[List[float], float]:
        """
        Find optimal thresholds to maximize QWK.
        
        Args:
            predictions: Continuous regression outputs
            targets: True integer labels
            verbose: Whether to print optimization progress
            
        Returns:
            Tuple of (optimized_thresholds, best_qwk)
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions = predictions.flatten()
        targets = targets.flatten().astype(int)
        
        # Initial QWK with equidistant thresholds
        initial_qwk = -self._objective(
            np.array(self.initial_thresholds), predictions, targets
        )
        
        if verbose:
            logger.info(f"Initial QWK with thresholds {self.initial_thresholds}: {initial_qwk:.4f}")
        
        # Set bounds (thresholds should be between 0 and num_classes)
        bounds = [(0, self.num_classes)] * (self.num_classes - 1)
        
        # Optimize
        if self.method.lower() == 'nelder-mead':
            result = minimize(
                self._objective,
                x0=np.array(self.initial_thresholds),
                args=(predictions, targets),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'disp': verbose}
            )
        elif self.method.lower() == 'powell':
            result = minimize(
                self._objective,
                x0=np.array(self.initial_thresholds),
                args=(predictions, targets),
                method='Powell',
                options={'maxiter': 1000, 'disp': verbose}
            )
        else:
            result = minimize(
                self._objective,
                x0=np.array(self.initial_thresholds),
                args=(predictions, targets),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'disp': verbose}
            )
        
        # Sort thresholds
        self.optimized_thresholds = sorted(result.x.tolist())
        best_qwk = -result.fun
        
        if verbose:
            logger.info(f"Optimized QWK: {best_qwk:.4f}")
            logger.info(f"Optimized thresholds: {self.optimized_thresholds}")
        
        return self.optimized_thresholds, best_qwk
    
    def apply(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        thresholds: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Apply thresholds to convert predictions to classes.
        
        Args:
            predictions: Continuous predictions
            thresholds: Thresholds to use (defaults to optimized or initial)
            
        Returns:
            Discrete class predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        if thresholds is None:
            thresholds = self.optimized_thresholds or self.initial_thresholds
        
        return self._apply_thresholds(predictions.flatten(), thresholds)
    
    def get_thresholds(self) -> List[float]:
        """Return current best thresholds."""
        return self.optimized_thresholds or self.initial_thresholds


class RounderOptimizer(ThresholdOptimizer):
    """
    Alternative optimizer using a different parameterization.
    
    Instead of absolute thresholds, optimizes offsets from class centers.
    This can sometimes converge faster.
    """
    
    def __init__(self, num_classes: int = 5):
        super().__init__(num_classes)
        # Offsets from class centers (0.5, 1.5, 2.5, 3.5)
        self.initial_offsets = [0.0] * (num_classes - 1)
    
    def _offsets_to_thresholds(self, offsets: np.ndarray) -> np.ndarray:
        """Convert offsets to absolute thresholds."""
        base = np.array([i + 0.5 for i in range(len(offsets))])
        return base + offsets
    
    def _objective(
        self,
        offsets: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Objective using offset parameterization."""
        thresholds = self._offsets_to_thresholds(offsets)
        thresholds = np.sort(thresholds)
        
        pred_classes = self._apply_thresholds(predictions, thresholds)
        qwk = quadratic_weighted_kappa(targets, pred_classes, self.num_classes)
        
        return -qwk
    
    def optimize(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        verbose: bool = True
    ) -> Tuple[List[float], float]:
        """Optimize using offset parameterization."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions = predictions.flatten()
        targets = targets.flatten().astype(int)
        
        # Optimize offsets
        result = minimize(
            self._objective,
            x0=np.array(self.initial_offsets),
            args=(predictions, targets),
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        # Convert offsets to thresholds
        self.optimized_thresholds = sorted(
            self._offsets_to_thresholds(result.x).tolist()
        )
        best_qwk = -result.fun
        
        if verbose:
            logger.info(f"Optimized thresholds: {self.optimized_thresholds}")
            logger.info(f"Best QWK: {best_qwk:.4f}")
        
        return self.optimized_thresholds, best_qwk


def grid_search_thresholds(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 5,
    step: float = 0.1,
    verbose: bool = False
) -> Tuple[List[float], float]:
    """
    Grid search for optimal thresholds (slower but guaranteed to find global optimum).
    
    Note: This is O(n^4) for 5 classes, so use small step sizes cautiously.
    
    Args:
        predictions: Continuous predictions
        targets: True labels
        num_classes: Number of classes
        step: Grid step size
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_thresholds, best_qwk)
    """
    from itertools import product
    
    # Create grid for each threshold
    grid = np.arange(0, num_classes, step)
    
    best_qwk = -1.0
    best_thresholds = [0.5, 1.5, 2.5, 3.5]
    
    # Generate all valid threshold combinations
    for thresholds in product(grid, repeat=num_classes - 1):
        # Skip if not sorted
        if list(thresholds) != sorted(thresholds):
            continue
        
        # Apply thresholds
        pred_classes = np.zeros(len(predictions), dtype=int)
        for i, t in enumerate(thresholds):
            pred_classes[predictions > t] = i + 1
        
        # Compute QWK
        qwk = quadratic_weighted_kappa(targets, pred_classes, num_classes)
        
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = list(thresholds)
            if verbose:
                print(f"New best: {best_qwk:.4f} with thresholds {best_thresholds}")
    
    return best_thresholds, best_qwk


if __name__ == "__main__":
    # Test threshold optimization
    np.random.seed(42)
    
    # Simulated data
    n_samples = 1000
    true_labels = np.random.choice([0, 1, 2, 3, 4], size=n_samples, 
                                    p=[0.5, 0.1, 0.25, 0.05, 0.1])
    
    # Simulated noisy predictions
    predictions = true_labels.astype(float) + np.random.randn(n_samples) * 0.5
    predictions = np.clip(predictions, 0, 4)
    
    # Test optimizer
    optimizer = ThresholdOptimizer()
    thresholds, qwk = optimizer.optimize(predictions, true_labels, verbose=True)
    
    print(f"\nFinal thresholds: {thresholds}")
    print(f"Final QWK: {qwk:.4f}")
    
    # Compare with initial thresholds
    pred_initial = optimizer._apply_thresholds(predictions, [0.5, 1.5, 2.5, 3.5])
    qwk_initial = quadratic_weighted_kappa(true_labels, pred_initial)
    print(f"Initial threshold QWK: {qwk_initial:.4f}")
    print(f"Improvement: {qwk - qwk_initial:.4f}")
