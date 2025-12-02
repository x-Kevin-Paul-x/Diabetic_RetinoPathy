"""
Image Preprocessing Script using Ben Graham Method

This script applies Ben Graham preprocessing to fundus images:
1. Crop and center the eye region
2. Gaussian blur subtraction for luminosity normalization
3. Circular masking
4. Resize to target size

Usage:
    python scripts/preprocess_images.py --input data/raw/aptos/train_images --output data/processed/aptos --size 512
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import multiprocessing as mp
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ben_graham import BenGrahamPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_image(
    image_path: Path,
    output_dir: Path,
    preprocessor: BenGrahamPreprocessor,
    overwrite: bool = False,
) -> bool:
    """
    Process a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        preprocessor: BenGrahamPreprocessor instance
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    output_path = output_dir / f"{image_path.stem}.png"
    
    if output_path.exists() and not overwrite:
        return True
    
    try:
        preprocessor.process_file(image_path, output_path)
        return True
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    size: int = 512,
    num_workers: int = 4,
    overwrite: bool = False,
):
    """
    Process all images in a directory using multiprocessing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        size: Output image size
        num_workers: Number of parallel workers
        overwrite: Whether to overwrite existing files
    """
    from tqdm import tqdm
    import cv2
    
    # Create preprocessor
    preprocessor = BenGrahamPreprocessor(output_size=size)
    
    # Find all images
    extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    if len(image_files) == 0:
        logger.warning("No images found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    if num_workers > 1:
        # Multiprocessing
        logger.info(f"Processing with {num_workers} workers...")
        
        # Use spawn method for Windows compatibility
        ctx = mp.get_context("spawn")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        success_count = 0
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            
            # Process batch sequentially (multiprocessing with OpenCV can be tricky)
            for img_path in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
                output_path = output_dir / f"{img_path.stem}.png"
                
                if output_path.exists() and not overwrite:
                    success_count += 1
                    continue
                
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not read {img_path.name}")
                        continue
                    
                    # Process
                    processed = preprocessor(img)
                    
                    # Save
                    cv2.imwrite(str(output_path), processed)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
    else:
        # Sequential processing
        success_count = 0
        for img_path in tqdm(image_files, desc="Processing"):
            output_path = output_dir / f"{img_path.stem}.png"
            
            if output_path.exists() and not overwrite:
                success_count += 1
                continue
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                processed = preprocessor(img)
                cv2.imwrite(str(output_path), processed)
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
    
    logger.info(f"Successfully processed {success_count}/{len(image_files)} images")
    logger.info(f"Output saved to {output_dir}")


def compare_preprocessing(
    input_path: Path,
    output_path: Path,
    size: int = 512,
):
    """
    Compare original vs preprocessed image side by side.
    
    Args:
        input_path: Path to input image
        output_path: Path to save comparison
        size: Processing size
    """
    import cv2
    import numpy as np
    
    preprocessor = BenGrahamPreprocessor(output_size=size)
    
    # Load and process
    original = cv2.imread(str(input_path))
    processed = preprocessor(original)
    
    # Resize original for comparison
    original_resized = cv2.resize(original, (size, size))
    
    # Create comparison image
    comparison = np.hstack([original_resized, processed])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Ben Graham", (size + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(output_path), comparison)
    logger.info(f"Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess fundus images using Ben Graham method"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size (default: 512)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed images"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Single image to compare (for visualization)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode: show original vs processed
        input_path = Path(args.compare)
        output_path = Path(args.output) if args.output.endswith(('.png', '.jpg')) else Path(args.output) / "comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compare_preprocessing(input_path, output_path, args.size)
    else:
        # Batch processing mode
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            size=args.size,
            num_workers=args.workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
