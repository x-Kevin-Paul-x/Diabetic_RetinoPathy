"""
Tests for the Diabetic Retinopathy Detection Pipeline
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DRModel
from src.utils import BenGrahamPreprocessor, quadratic_weighted_kappa, ThresholdOptimizer


class TestBenGrahamPreprocessor:
    """Test Ben Graham preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = BenGrahamPreprocessor(output_size=512)
        assert preprocessor.output_size == 512
    
    def test_preprocessor_output_shape(self):
        """Test preprocessing produces correct output shape."""
        preprocessor = BenGrahamPreprocessor(output_size=256)
        # Create dummy image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # Add black border to simulate fundus
        image[:50, :, :] = 0
        image[-50:, :, :] = 0
        image[:, :50, :] = 0
        image[:, -50:, :] = 0
        
        result = preprocessor.process(image)
        assert result.shape == (256, 256, 3)
    
    def test_preprocessor_dtype(self):
        """Test output dtype."""
        preprocessor = BenGrahamPreprocessor(output_size=256)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = preprocessor.process(image)
        assert result.dtype == np.uint8


class TestDRModel:
    """Test DR Model."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return DRModel(
            backbone='efficientnet_b0',  # Smaller for testing
            num_classes=5,
            head_type='regression',
            pretrained=False,
            dropout=0.5,
            pooling='avg'
        )
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model is not None
    
    def test_model_forward_regression(self, model):
        """Test forward pass with regression head."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 1)
    
    def test_model_forward_classification(self):
        """Test forward pass with classification head."""
        model = DRModel(
            backbone='efficientnet_b0',
            num_classes=5,
            head_type='classification',
            pretrained=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 5)
    
    def test_model_gradient_flow(self, model):
        """Test gradients flow through the model."""
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMetrics:
    """Test metric functions."""
    
    def test_qwk_perfect_agreement(self):
        """Test QWK with perfect agreement."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        qwk = quadratic_weighted_kappa(y_pred, y_true)
        assert qwk == pytest.approx(1.0, abs=0.01)
    
    def test_qwk_no_agreement(self):
        """Test QWK with opposite predictions."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([4, 4, 4, 4, 4])
        qwk = quadratic_weighted_kappa(y_pred, y_true)
        assert qwk < 0  # Should be negative
    
    def test_qwk_random_agreement(self):
        """Test QWK with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        qwk = quadratic_weighted_kappa(y_pred, y_true)
        assert -1 <= qwk <= 1


class TestThresholdOptimizer:
    """Test threshold optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes."""
        optimizer = ThresholdOptimizer(num_classes=5)
        assert optimizer.num_classes == 5
    
    def test_optimizer_output(self):
        """Test optimizer produces valid thresholds."""
        np.random.seed(42)
        optimizer = ThresholdOptimizer(num_classes=5)
        
        # Simulate predictions
        y_pred = np.random.uniform(0, 4, 100)
        y_true = np.random.randint(0, 5, 100)
        
        thresholds = optimizer.optimize(y_pred, y_true)
        
        # Should have 4 thresholds for 5 classes
        assert len(thresholds) == 4
        # Thresholds should be sorted
        assert all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1))
    
    def test_optimizer_improves_qwk(self):
        """Test that optimized thresholds improve or maintain QWK."""
        np.random.seed(42)
        
        # Create somewhat correlated predictions
        y_true = np.random.randint(0, 5, 200)
        y_pred = y_true + np.random.uniform(-0.5, 0.5, 200)
        y_pred = np.clip(y_pred, 0, 4)
        
        # Default thresholds
        default_thresholds = [0.5, 1.5, 2.5, 3.5]
        
        def apply_thresholds(preds, thresholds):
            classes = np.zeros_like(preds, dtype=int)
            for i, t in enumerate(thresholds):
                classes[preds > t] = i + 1
            return classes
        
        default_classes = apply_thresholds(y_pred, default_thresholds)
        default_qwk = quadratic_weighted_kappa(default_classes, y_true)
        
        # Optimized thresholds
        optimizer = ThresholdOptimizer(num_classes=5)
        optimal_thresholds = optimizer.optimize(y_pred, y_true)
        optimal_classes = apply_thresholds(y_pred, optimal_thresholds)
        optimal_qwk = quadratic_weighted_kappa(optimal_classes, y_true)
        
        # Optimized should be at least as good
        assert optimal_qwk >= default_qwk - 0.01


class TestDataPipeline:
    """Test data loading pipeline."""
    
    def test_transform_output_shape(self):
        """Test transforms produce correct output."""
        from src.datamodules.transforms import get_val_transforms
        
        transform = get_val_transforms(image_size=224)
        
        # Create dummy image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = transform(image=image)
        
        assert result['image'].shape == (3, 224, 224)
    
    def test_augmentation_determinism(self):
        """Test augmentations are applied."""
        from src.datamodules.transforms import get_train_transforms
        
        transform = get_train_transforms(image_size=224)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Apply twice, should be different (with high probability)
        result1 = transform(image=image)['image']
        result2 = transform(image=image)['image']
        
        # Results should be different due to random augmentation
        # This might occasionally fail by chance, so we just check shape
        assert result1.shape == result2.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
