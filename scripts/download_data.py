"""
Data Download Script for Diabetic Retinopathy Datasets

This script downloads and prepares the following datasets:
1. APTOS 2019 Blindness Detection (Kaggle)
2. EyePACS (Kaggle 2015) - Optional, for pretraining
3. Messidor-2 - Optional, for external validation
4. IDRiD - Optional, for pixel-level annotations

Supports Kaggle API token from .env file (Kaggle_API_Token)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import zipfile
import shutil
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_kaggle_credentials():
    """Setup Kaggle API credentials from .env file or existing kaggle.json."""
    # First, check for Kaggle_API_Token in environment/.env
    kaggle_token = os.getenv("Kaggle_API_Token")
    
    if kaggle_token:
        # Parse the token - format should be "username:key" or just the key
        # If it's a full token like "KGAT_xxxx", we need username too
        kaggle_username = os.getenv("Kaggle_Username", "")
        
        # Set environment variables that Kaggle API uses
        if ":" in kaggle_token:
            # Format: username:key
            username, key = kaggle_token.split(":", 1)
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        else:
            # Just the key, need username from env
            os.environ["KAGGLE_KEY"] = kaggle_token
            if kaggle_username:
                os.environ["KAGGLE_USERNAME"] = kaggle_username
        
        logger.info("Using Kaggle credentials from .env file")
        return True
    
    # Fall back to checking kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        logger.info("Using Kaggle credentials from ~/.kaggle/kaggle.json")
        return True
    
    logger.error(
        "Kaggle credentials not found!\n"
        "Option 1: Add to .env file:\n"
        "  Kaggle_API_Token=your_api_key\n"
        "  Kaggle_Username=your_username\n"
        "\nOption 2: Use kaggle.json:\n"
        "  1. Go to https://www.kaggle.com/account\n"
        "  2. Click 'Create New API Token'\n"
        "  3. Save kaggle.json to ~/.kaggle/kaggle.json"
    )
    return False


def download_aptos(output_dir: Path, overwrite: bool = False):
    """
    Download APTOS 2019 Blindness Detection dataset.
    
    Dataset info:
    - 3,662 training images
    - 5 classes (0-4 DR severity)
    - Source: Aravind Eye Hospital, India
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    
    dataset_dir = output_dir / "aptos"
    
    if dataset_dir.exists() and not overwrite:
        logger.info(f"APTOS dataset already exists at {dataset_dir}")
        return True
    
    logger.info("Downloading APTOS 2019 dataset...")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download competition data
        api.competition_download_files(
            competition="aptos2019-blindness-detection",
            path=str(output_dir),
            quiet=False,
        )
        
        # Extract
        zip_path = output_dir / "aptos2019-blindness-detection.zip"
        if zip_path.exists():
            logger.info("Extracting APTOS dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            zip_path.unlink()
        
        logger.info(f"APTOS dataset downloaded to {dataset_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download APTOS dataset: {e}")
        logger.info(
            "You can manually download from:\n"
            "https://www.kaggle.com/competitions/aptos2019-blindness-detection/data"
        )
        return False


def download_eyepacs(output_dir: Path, overwrite: bool = False):
    """
    Download EyePACS (Kaggle 2015) dataset.
    
    Dataset info:
    - ~88,702 images (35k train, 53k test)
    - 5 classes (0-4 DR severity)
    - Large dataset for pretraining
    
    Note: This is a large dataset (~80GB), download may take a while.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    
    dataset_dir = output_dir / "eyepacs"
    
    if dataset_dir.exists() and not overwrite:
        logger.info(f"EyePACS dataset already exists at {dataset_dir}")
        return True
    
    logger.info("Downloading EyePACS dataset (this may take a while)...")
    logger.warning("EyePACS is ~80GB. Ensure sufficient disk space.")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download competition data
        api.competition_download_files(
            competition="diabetic-retinopathy-detection",
            path=str(output_dir),
            quiet=False,
        )
        
        # Extract (multiple zip files)
        for zip_file in output_dir.glob("*.zip"):
            if "diabetic" in zip_file.name.lower():
                logger.info(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                zip_file.unlink()
        
        logger.info(f"EyePACS dataset downloaded to {dataset_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download EyePACS dataset: {e}")
        return False


def download_from_url(url: str, output_path: Path):
    """Download file from URL with progress bar."""
    import urllib.request
    from tqdm import tqdm
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def create_sample_data(output_dir: Path):
    """
    Create sample data structure for testing without downloading.
    
    Creates dummy images and CSV for pipeline testing.
    """
    import numpy as np
    import cv2
    import pandas as pd
    
    logger.info("Creating sample data for testing...")
    
    sample_dir = output_dir / "sample"
    images_dir = sample_dir / "train_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images (random fundus-like images)
    np.random.seed(42)
    samples = []
    
    for i in range(50):
        # Create circular fundus-like image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Random background color
        bg_color = np.random.randint(50, 150)
        cv2.circle(img, (256, 256), 230, (bg_color, bg_color + 20, bg_color - 10), -1)
        
        # Random spots (simulating lesions)
        grade = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.15, 0.25, 0.1, 0.1])
        num_spots = grade * 5
        
        for _ in range(num_spots):
            x = np.random.randint(100, 412)
            y = np.random.randint(100, 412)
            r = np.random.randint(3, 15)
            color = (np.random.randint(0, 100), 0, 0)  # Reddish spots
            cv2.circle(img, (x, y), r, color, -1)
        
        # Save image
        img_name = f"sample_{i:04d}"
        cv2.imwrite(str(images_dir / f"{img_name}.png"), img)
        
        samples.append({
            "id_code": img_name,
            "diagnosis": grade,
        })
    
    # Create CSV
    df = pd.DataFrame(samples)
    df.to_csv(sample_dir / "train.csv", index=False)
    
    logger.info(f"Created {len(samples)} sample images in {sample_dir}")
    logger.info(f"Class distribution: {df['diagnosis'].value_counts().to_dict()}")
    
    return True


def prepare_directory_structure(base_dir: Path):
    """Create the expected directory structure."""
    directories = [
        base_dir / "raw",
        base_dir / "processed",
        base_dir / "splits",
    ]
    
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Diabetic Retinopathy Detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aptos", "eyepacs", "sample", "all"],
        default="aptos",
        help="Dataset to download (default: aptos)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    # Create directory structure
    prepare_directory_structure(output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "sample":
        # Create sample data (no download needed)
        success = create_sample_data(output_dir)
    else:
        # Check Kaggle credentials
        if not setup_kaggle_credentials():
            logger.info("\nAlternatively, you can create sample data for testing:")
            logger.info("  python scripts/download_data.py --dataset sample")
            sys.exit(1)
        
        success = True
        
        if args.dataset in ["aptos", "all"]:
            success &= download_aptos(output_dir, args.overwrite)
        
        if args.dataset in ["eyepacs", "all"]:
            success &= download_eyepacs(output_dir, args.overwrite)
    
    if success:
        logger.info("\n" + "="*50)
        logger.info("Download complete!")
        logger.info("="*50)
        logger.info("\nNext steps:")
        logger.info("1. Preprocess images with Ben Graham method:")
        logger.info("   python scripts/preprocess_images.py --input data/raw/aptos/train_images --output data/processed/aptos")
        logger.info("\n2. Start training:")
        logger.info("   python train.py")
    else:
        logger.error("Download failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
