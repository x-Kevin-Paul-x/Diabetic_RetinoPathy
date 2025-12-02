"""
Inference Script for Diabetic Retinopathy Detection

This script provides:
1. Single image inference
2. Batch inference
3. ONNX export
4. Test-Time Augmentation (TTA)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import json

import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import DRModel
from src.utils.ben_graham import BenGrahamPreprocessor, preprocess_for_inference
from src.datamodules.transforms import get_val_transforms, get_tta_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]


class DRPredictor:
    """
    Diabetic Retinopathy prediction class.
    
    Supports:
    - Single image inference
    - Batch inference
    - Test-Time Augmentation (TTA)
    - ONNX runtime inference
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        device: str = "auto",
        image_size: int = 456,
        use_tta: bool = False,
        thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to PyTorch Lightning checkpoint
            onnx_path: Path to ONNX model (alternative to checkpoint)
            device: Device to use ('auto', 'cuda', 'cpu')
            image_size: Input image size
            use_tta: Whether to use Test-Time Augmentation
            thresholds: Custom thresholds for regression to class conversion
        """
        self.image_size = image_size
        self.use_tta = use_tta
        self.thresholds = thresholds or [0.5, 1.5, 2.5, 3.5]
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.use_onnx = onnx_path is not None
        
        if self.use_onnx:
            self._load_onnx(onnx_path)
        elif checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            raise ValueError("Either checkpoint_path or onnx_path must be provided")
        
        # Preprocessing
        self.preprocessor = BenGrahamPreprocessor(output_size=512)
        self.transform = get_val_transforms(image_size=image_size)
        
        if use_tta:
            self.tta_transforms = get_tta_transforms(image_size=image_size)
        
        logger.info(f"Predictor initialized on {self.device}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from PyTorch Lightning checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        self.model = DRModel.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Get thresholds from model if available
        if hasattr(self.model, 'optimized_thresholds'):
            self.thresholds = self.model.optimized_thresholds
            logger.info(f"Using optimized thresholds: {self.thresholds}")
    
    def _load_onnx(self, onnx_path: str):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")
        
        logger.info(f"Loading ONNX model from {onnx_path}")
        
        # Set providers based on device
        if self.device.type == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        self.model = None
    
    def _preprocess_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and preprocess image."""
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3 and len(image.shape) == 3:
                # Assume BGR if loaded with cv2
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Ben Graham preprocessing
        processed = self.preprocessor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        return processed
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor with transforms."""
        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0)
        return tensor.to(self.device)
    
    def _regression_to_class(self, value: float) -> int:
        """Convert regression value to class using thresholds."""
        for i, threshold in enumerate(self.thresholds):
            if value <= threshold:
                return i
        return len(self.thresholds)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_raw: bool = False,
    ) -> Dict:
        """
        Predict DR grade for a single image.
        
        Args:
            image: Image path or numpy array
            return_raw: Whether to return raw regression value
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        processed = self._preprocess_image(image)
        
        if self.use_tta:
            return self._predict_with_tta(processed, return_raw)
        
        # Convert to tensor
        tensor = self._to_tensor(processed)
        
        # Inference
        if self.use_onnx:
            output = self.onnx_session.run(
                None,
                {"input": tensor.cpu().numpy()}
            )[0]
            raw_value = float(output[0])
        else:
            output = self.model(tensor)
            raw_value = float(output.squeeze().cpu().numpy())
        
        # Convert to class
        predicted_class = self._regression_to_class(raw_value)
        
        result = {
            "class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": self._compute_confidence(raw_value),
        }
        
        if return_raw:
            result["raw_value"] = raw_value
        
        return result
    
    def _predict_with_tta(
        self,
        processed_image: np.ndarray,
        return_raw: bool = False,
    ) -> Dict:
        """Predict with Test-Time Augmentation."""
        predictions = []
        
        for name, transform in self.tta_transforms.items():
            # Apply transform
            transformed = transform(image=processed_image)
            tensor = transformed["image"].unsqueeze(0).to(self.device)
            
            # Inference
            if self.use_onnx:
                output = self.onnx_session.run(
                    None,
                    {"input": tensor.cpu().numpy()}
                )[0]
                predictions.append(float(output[0]))
            else:
                output = self.model(tensor)
                predictions.append(float(output.squeeze().cpu().numpy()))
        
        # Average predictions
        raw_value = np.mean(predictions)
        predicted_class = self._regression_to_class(raw_value)
        
        result = {
            "class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "confidence": self._compute_confidence(raw_value),
            "tta_std": float(np.std(predictions)),
        }
        
        if return_raw:
            result["raw_value"] = raw_value
            result["tta_predictions"] = predictions
        
        return result
    
    def _compute_confidence(self, raw_value: float) -> float:
        """
        Compute prediction confidence based on distance from thresholds.
        
        Higher confidence when prediction is far from threshold boundaries.
        """
        # Find distance to nearest threshold
        min_dist = float('inf')
        for threshold in self.thresholds:
            dist = abs(raw_value - threshold)
            min_dist = min(min_dist, dist)
        
        # Convert to confidence (sigmoid-like)
        confidence = 1.0 - np.exp(-min_dist * 2)
        
        return float(confidence)
    
    def predict_batch(
        self,
        images: List[Union[str, Path]],
        return_raw: bool = False,
    ) -> List[Dict]:
        """
        Predict DR grades for multiple images.
        
        Args:
            images: List of image paths
            return_raw: Whether to return raw values
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image in tqdm(images, desc="Processing"):
            try:
                result = self.predict(image, return_raw=return_raw)
                result["image_path"] = str(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image}: {e}")
                results.append({
                    "image_path": str(image),
                    "error": str(e),
                })
        
        return results
    
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: bool = True,
    ):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic batch size
        """
        if self.use_onnx:
            raise ValueError("Cannot export from ONNX model")
        
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        
        # Dynamic axes for batch size
        if dynamic_axes:
            dynamic_axes_dict = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        else:
            dynamic_axes_dict = None
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
        )
        
        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model exported successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DR Inference")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--input_dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint or ONNX")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--export_onnx", type=str, help="Export model to ONNX")
    
    args = parser.parse_args()
    
    # Determine if ONNX
    is_onnx = args.model.endswith(".onnx")
    
    # Initialize predictor
    predictor = DRPredictor(
        checkpoint_path=None if is_onnx else args.model,
        onnx_path=args.model if is_onnx else None,
        device=args.device,
        use_tta=args.tta,
    )
    
    # Export to ONNX if requested
    if args.export_onnx:
        predictor.export_onnx(args.export_onnx)
        return
    
    # Single image inference
    if args.image:
        result = predictor.predict(args.image, return_raw=True)
        print(f"\nPrediction for {args.image}:")
        print(f"  Class: {result['class']} ({result['class_name']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Raw value: {result['raw_value']:.3f}")
        return
    
    # Batch inference
    if args.input_dir:
        input_dir = Path(args.input_dir)
        images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        
        results = predictor.predict_batch(images, return_raw=True)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        classes = [r.get("class", -1) for r in results if "error" not in r]
        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {args.output}")
        print(f"\nClass distribution:")
        for i, name in enumerate(CLASS_NAMES):
            count = classes.count(i)
            print(f"  {name}: {count} ({count/len(classes)*100:.1f}%)")


if __name__ == "__main__":
    main()
