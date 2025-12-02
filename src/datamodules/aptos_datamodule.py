"""
APTOS 2019 Dataset and DataModule for Diabetic Retinopathy Detection

This module provides:
1. APTOSDataset: PyTorch Dataset for APTOS 2019 images
2. APTOSDataModule: PyTorch Lightning DataModule with k-fold support
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from .transforms import get_train_transforms, get_val_transforms, get_tta_transforms


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Blindness Detection.
    
    Supports:
    - Loading preprocessed (Ben Graham) or raw images
    - On-the-fly augmentation with Albumentations
    - Regression or classification targets
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_type: str = "regression",  # "regression" or "classification"
        image_extension: str = ".png",
        return_metadata: bool = False,
    ):
        """
        Initialize APTOS Dataset.
        
        Args:
            dataframe: DataFrame with 'id_code' and 'diagnosis' columns
            image_dir: Directory containing images
            transform: Albumentations transform pipeline
            target_type: "regression" for scalar output, "classification" for one-hot
            image_extension: Image file extension
            return_metadata: Whether to return image ID with sample
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_type = target_type
        self.image_extension = image_extension
        self.return_metadata = return_metadata
        
        # Validate dataframe
        required_cols = ["id_code", "diagnosis"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must have '{col}' column")
        
        # Store labels
        self.labels = self.df["diagnosis"].values
        self.image_ids = self.df["id_code"].values
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - 'image': Tensor [C, H, W]
                - 'target': Tensor (scalar for regression, long for classification)
                - 'id' (optional): Image ID string
        """
        # Get image path
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}{self.image_extension}"
        
        # Try alternative extensions if not found
        if not image_path.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_path = self.image_dir / f"{image_id}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get target
        label = self.labels[idx]
        
        if self.target_type == "regression":
            target = torch.tensor(label, dtype=torch.float32)
        else:  # classification
            target = torch.tensor(label, dtype=torch.long)
        
        sample = {
            "image": image,
            "target": target,
        }
        
        if self.return_metadata:
            sample["id"] = image_id
        
        return sample


class APTOSDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for APTOS 2019 dataset.
    
    Features:
    - Automatic train/val split or k-fold cross-validation
    - Stratified sampling for class imbalance
    - Weighted random sampler option
    - TTA support for validation/test
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        train_csv: str = "train.csv",
        image_dir: str = "train_images",
        processed_dir: Optional[str] = None,  # Ben Graham processed images
        # Split settings
        fold: int = 0,
        num_folds: int = 5,
        val_split: float = 0.2,  # Used if num_folds = 1
        # Training settings
        batch_size: int = 16,
        num_workers: int = 4,
        target_type: str = "regression",
        # Augmentation settings
        image_size: int = 456,
        augmentation_config: Optional[Dict] = None,
        # Sampling
        use_weighted_sampler: bool = True,
        # Reproducibility
        seed: int = 42,
    ):
        """
        Initialize APTOS DataModule.
        
        Args:
            data_dir: Root data directory
            train_csv: Path to training CSV (relative to data_dir)
            image_dir: Path to image directory (relative to data_dir)
            processed_dir: Path to Ben Graham processed images (optional)
            fold: Current fold index (0 to num_folds-1)
            num_folds: Number of folds for CV (1 for simple split)
            val_split: Validation split ratio (used if num_folds=1)
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            target_type: "regression" or "classification"
            image_size: Image size for transforms
            augmentation_config: Optional augmentation config dict
            use_weighted_sampler: Whether to use WeightedRandomSampler
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train_csv = train_csv
        self.image_dir = image_dir
        self.processed_dir = processed_dir
        self.fold = fold
        self.num_folds = num_folds
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_type = target_type
        self.image_size = image_size
        self.augmentation_config = augmentation_config or {}
        self.use_weighted_sampler = use_weighted_sampler
        self.seed = seed
        
        # Will be set in setup()
        self.train_df = None
        self.val_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        
    def prepare_data(self):
        """
        Download or prepare data (called once per node).
        
        Note: Don't assign state here (self.x = y).
        """
        csv_path = self.data_dir / self.train_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training CSV not found at {csv_path}. "
                "Please run scripts/download_data.py first."
            )
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load CSV
        csv_path = self.data_dir / self.train_csv
        df = pd.read_csv(csv_path)
        
        # Determine image directory
        if self.processed_dir:
            img_dir = self.data_dir / self.processed_dir
        else:
            img_dir = self.data_dir / self.image_dir
        
        # Create train/val split
        if self.num_folds > 1:
            # K-fold cross-validation
            skf = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed
            )
            
            splits = list(skf.split(df["id_code"], df["diagnosis"]))
            train_idx, val_idx = splits[self.fold]
            
            self.train_df = df.iloc[train_idx].reset_index(drop=True)
            self.val_df = df.iloc[val_idx].reset_index(drop=True)
        else:
            # Simple stratified split
            from sklearn.model_selection import train_test_split
            
            train_idx, val_idx = train_test_split(
                np.arange(len(df)),
                test_size=self.val_split,
                stratify=df["diagnosis"],
                random_state=self.seed
            )
            
            self.train_df = df.iloc[train_idx].reset_index(drop=True)
            self.val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Calculate class weights for sampler
        train_labels = self.train_df["diagnosis"].values
        class_counts = np.bincount(train_labels, minlength=5)
        self.class_weights = 1.0 / np.maximum(class_counts, 1)
        
        # Create transforms
        train_transform = get_train_transforms(
            image_size=self.image_size,
            **self.augmentation_config
        )
        val_transform = get_val_transforms(image_size=self.image_size)
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = APTOSDataset(
                dataframe=self.train_df,
                image_dir=img_dir,
                transform=train_transform,
                target_type=self.target_type,
            )
            
            self.val_dataset = APTOSDataset(
                dataframe=self.val_df,
                image_dir=img_dir,
                transform=val_transform,
                target_type=self.target_type,
            )
        
        if stage == "validate":
            self.val_dataset = APTOSDataset(
                dataframe=self.val_df,
                image_dir=img_dir,
                transform=val_transform,
                target_type=self.target_type,
            )
    
    def _get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Create weighted random sampler for training."""
        if not self.use_weighted_sampler:
            return None
        
        train_labels = self.train_df["diagnosis"].values
        sample_weights = self.class_weights[train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        sampler = self._get_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader (same as validation)."""
        return self.val_dataloader()
    
    def get_class_distribution(self) -> Dict[str, np.ndarray]:
        """Get class distribution for train and val sets."""
        train_dist = np.bincount(self.train_df["diagnosis"].values, minlength=5)
        val_dist = np.bincount(self.val_df["diagnosis"].values, minlength=5)
        
        return {
            "train": train_dist,
            "val": val_dist,
            "train_pct": train_dist / train_dist.sum() * 100,
            "val_pct": val_dist / val_dist.sum() * 100,
        }


def create_fold_csvs(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    num_folds: int = 5,
    seed: int = 42,
):
    """
    Pre-generate fold CSV files for reproducibility.
    
    Args:
        csv_path: Path to original training CSV
        output_dir: Directory to save fold CSVs
        num_folds: Number of folds
        seed: Random seed
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["id_code"], df["diagnosis"])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_df.to_csv(output_dir / f"fold{fold}_train.csv", index=False)
        val_df.to_csv(output_dir / f"fold{fold}_val.csv", index=False)
        
        print(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")
    
    print(f"\nSaved fold CSVs to {output_dir}")


if __name__ == "__main__":
    # Test DataModule
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    dm = APTOSDataModule(
        data_dir=args.data_dir,
        batch_size=4,
        num_workers=0,
    )
    
    dm.setup("fit")
    
    # Print class distribution
    dist = dm.get_class_distribution()
    print("Class Distribution:")
    print(f"Train: {dist['train']} ({dist['train_pct']})")
    print(f"Val: {dist['val']} ({dist['val_pct']})")
    
    # Test dataloader
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    print(f"\nBatch shape: {batch['image'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Targets: {batch['target']}")
