"""
Ben Graham Preprocessing for Retinal Fundus Images

This module implements the industry-standard preprocessing technique for
diabetic retinopathy detection, originally developed by Ben Graham for
the Kaggle Diabetic Retinopathy Detection competition (2015).

The technique consists of:
1. Crop and center the fundus image (remove black background)
2. Gaussian blur subtraction (local average subtraction) for luminosity normalization
3. Circular masking to remove edge artifacts

Reference: https://www.kaggle.com/code/banzaibanzer/applying-ben-s-preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BenGrahamPreprocessor:
    """
    Ben Graham preprocessing for fundus images.
    
    The algorithm normalizes luminosity and enhances high-frequency features
    (lesions) while suppressing slowly varying background (illumination gradients).
    
    Formula: I_processed = α * I_original + β * G(I_original, σ) + γ
    
    Default parameters (α=4, β=-4, γ=128) act as a high-pass filter.
    """
    
    def __init__(
        self,
        output_size: int = 512,
        alpha: float = 4.0,
        beta: float = -4.0,
        gamma: float = 128.0,
        sigma_ratio: float = 0.1,  # sigma = output_size * sigma_ratio
        crop_black_margin: bool = True,
        apply_circular_mask: bool = True,
        mask_value: int = 0,
    ):
        """
        Initialize Ben Graham preprocessor.
        
        Args:
            output_size: Target size for output image (square)
            alpha: Weight for original image
            beta: Weight for blurred image (negative for subtraction)
            gamma: Constant offset (128 for centering around mid-gray)
            sigma_ratio: Gaussian blur sigma as ratio of image size
            crop_black_margin: Whether to crop black borders
            apply_circular_mask: Whether to apply circular mask at the end
            mask_value: Value for masked pixels (0 = black)
        """
        self.output_size = output_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_ratio = sigma_ratio
        self.crop_black_margin = crop_black_margin
        self.apply_circular_mask = apply_circular_mask
        self.mask_value = mask_value
        
        # Calculate kernel size (must be odd)
        sigma = int(output_size * sigma_ratio)
        self.kernel_size = sigma * 2 + 1
        self.sigma = sigma
    
    def _find_eye_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Find the bounding box of the eye region by thresholding.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (x, y, width, height) for the eye region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find non-black regions
        # Using a low threshold to capture the entire fundus
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return full image if no contours found
            return 0, 0, image.shape[1], image.shape[0]
        
        # Find the largest contour (should be the eye)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return x, y, w, h
    
    def _crop_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to square aspect ratio centered on the eye region.
        
        Args:
            image: Input BGR image
            
        Returns:
            Square cropped image
        """
        if not self.crop_black_margin:
            # Just make it square by cropping to minimum dimension
            h, w = image.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            return image[start_y:start_y + min_dim, start_x:start_x + min_dim]
        
        # Find eye region
        x, y, w, h = self._find_eye_region(image)
        
        # Calculate center of eye region
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determine square size (use max of width and height with padding)
        size = max(w, h)
        size = int(size * 1.05)  # 5% padding
        half_size = size // 2
        
        # Calculate crop bounds
        img_h, img_w = image.shape[:2]
        
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(img_w, center_x + half_size)
        y2 = min(img_h, center_y + half_size)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # If not perfectly square, pad with black
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
        """
        Apply the Ben Graham Gaussian blur subtraction.
        
        Formula: I_processed = α * I_original + β * G(I_original, σ) + γ
        
        Args:
            image: Input BGR image (already cropped and resized)
            
        Returns:
            Processed image with enhanced local contrast
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        
        # Apply the transformation
        # Use cv2.addWeighted for numerical stability
        processed = cv2.addWeighted(
            image, self.alpha,
            blurred, self.beta,
            self.gamma
        )
        
        # Clip to valid range
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        return processed
    
    def _apply_circular_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a circular mask to remove edge artifacts.
        
        Args:
            image: Input image
            
        Returns:
            Image with circular mask applied
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1]) - 2  # Slight inset
        
        # Create circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask
        result = image.copy()
        result[mask == 0] = self.mask_value
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full Ben Graham preprocessing pipeline.
        
        Args:
            image: Input BGR image (numpy array)
            
        Returns:
            Preprocessed image of size (output_size, output_size, 3)
        """
        # Step 1: Crop to square
        cropped = self._crop_to_square(image)
        
        # Step 2: Resize to target size
        resized = cv2.resize(cropped, (self.output_size, self.output_size), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Apply Gaussian blur subtraction
        processed = self._apply_gaussian_filter(resized)
        
        # Step 4: Apply circular mask (optional)
        if self.apply_circular_mask:
            processed = self._apply_circular_mask(processed)
        
        return processed
    
    def process_file(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image (optional)
            
        Returns:
            Processed image array
        """
        input_path = Path(input_path)
        
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Process
        processed = self(image)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            logger.debug(f"Saved processed image to {output_path}")
        
        return processed
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tif', '.tiff'),
        overwrite: bool = False,
    ) -> int:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            extensions: Tuple of valid image extensions
            overwrite: Whether to overwrite existing files
            
        Returns:
            Number of images processed
        """
        from tqdm import tqdm
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        processed_count = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            output_path = output_dir / f"{img_path.stem}.png"  # Save as PNG
            
            if output_path.exists() and not overwrite:
                logger.debug(f"Skipping {img_path.name} (already exists)")
                continue
            
            try:
                self.process_file(img_path, output_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
        
        logger.info(f"Processed {processed_count}/{len(image_files)} images")
        return processed_count


def preprocess_for_inference(
    image: np.ndarray,
    output_size: int = 512,
) -> np.ndarray:
    """
    Quick preprocessing function for inference.
    
    Args:
        image: Input BGR image
        output_size: Target output size
        
    Returns:
        Preprocessed image
    """
    preprocessor = BenGrahamPreprocessor(output_size=output_size)
    return preprocessor(image)


if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Ben Graham Preprocessing for Fundus Images")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output path or directory")
    parser.add_argument("--size", type=int, default=512, help="Output size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    preprocessor = BenGrahamPreprocessor(output_size=args.size)
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        preprocessor.process_directory(input_path, args.output, overwrite=args.overwrite)
    else:
        preprocessor.process_file(input_path, args.output)
        print(f"Processed {input_path} -> {args.output}")
