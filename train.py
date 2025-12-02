"""
Training Script for Diabetic Retinopathy Detection and Grading

A simplified, robust training script using PyTorch Lightning.
Supports both Hydra config and direct CLI arguments.

Usage:
    # Simple training (recommended)
    python train.py --quick
    
    # With custom parameters
    python train.py --quick --epochs 30 --batch-size 16 --backbone efficientnet_b5
    
    # With Hydra config (advanced)
    python train.py training.epochs=50
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class DRLightningModule:
    """PyTorch Lightning module for DR detection - defined before use."""
    pass  # Will be properly defined below


def quick_train(args):
    """
    Quick training mode without Hydra configuration.
    Simpler and more direct for quick experiments.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    
    from src.datamodules import APTOSDataModule
    from src.models import DRModel
    from src.utils import quadratic_weighted_kappa
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    logger.info("=" * 60)
    logger.info("DIABETIC RETINOPATHY DETECTION - TRAINING")
    logger.info("=" * 60)
    
    # Check data exists
    train_csv = data_dir / "aptos" / "train.csv"
    train_images = data_dir / "aptos" / "train_images"
    processed_images = data_dir / "aptos" / "processed"
    
    if not train_csv.exists():
        logger.error(f"Training CSV not found at {train_csv}")
        logger.error("Please download APTOS dataset and extract to data/aptos/")
        sys.exit(1)
    
    if not train_images.exists():
        logger.error(f"Training images not found at {train_images}")
        sys.exit(1)
    
    # Use processed images if available
    use_processed = processed_images.exists() and any(processed_images.iterdir()) if processed_images.exists() else False
    if use_processed:
        logger.info(f"Using preprocessed images from {processed_images}")
    else:
        logger.info("Using raw images (run preprocessing for better results)")
    
    # Load data info
    df = pd.read_csv(train_csv)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")
    
    # Create data module
    logger.info("\nCreating data module...")
    datamodule = APTOSDataModule(
        data_dir=data_dir,
        train_csv="aptos/train.csv",
        image_dir="aptos/train_images",
        processed_dir="aptos/processed" if use_processed else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=0.2,
        seed=args.seed,
    )
    
    # Create model
    logger.info(f"\nCreating model: {args.backbone}")
    model = DRModel(
        backbone_name=f"tf_{args.backbone}_ns",  # e.g., tf_efficientnet_b5_ns
        num_classes=5,
        head_type="regression",
        pretrained=True,
        head_dropout=0.5,
        pooling_type="gem"
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create Lightning module
    lightning_module = DRTrainingModule(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = project_root / "checkpoints" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="dr-{epoch:02d}-{val_qwk:.4f}",
            monitor="val_qwk",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_qwk",
            mode="max",
            patience=args.patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Loggers
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = [
        TensorBoardLogger(save_dir=log_dir, name="tensorboard"),
        CSVLogger(save_dir=log_dir, name="csv_logs"),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate})")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info("=" * 60 + "\n")
    
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Results
    best_qwk = trainer.checkpoint_callback.best_model_score
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation QWK: {best_qwk:.4f}")
    logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
    logger.info("=" * 60)
    
    return best_qwk


# Import pytorch_lightning for the module definition
try:
    import pytorch_lightning as pl
    
    class DRTrainingModule(pl.LightningModule):
        """PyTorch Lightning module for DR detection."""
        
        def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            num_epochs: int = 30,
        ):
            super().__init__()
            self.save_hyperparameters(ignore=["model"])
            self.model = model
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.num_epochs = num_epochs
            
            # Loss function
            self.criterion = nn.MSELoss()
            
            # Store predictions for epoch-level metrics
            self.training_step_outputs = []
            self.validation_step_outputs = []
        
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            images = batch["image"]
            labels = batch["target"]
            outputs = self(images).squeeze()
            
            # Handle single sample batch
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = self.criterion(outputs, labels.float())
            preds = outputs.round().clamp(0, 4).long()
            
            self.training_step_outputs.append({
                "loss": loss.detach(),
                "preds": preds.detach(),
                "labels": labels.detach(),
            })
            
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def on_train_epoch_end(self):
            if self.training_step_outputs:
                all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
                all_labels = torch.cat([x["labels"] for x in self.training_step_outputs])
                
                from src.utils import quadratic_weighted_kappa
                qwk = quadratic_weighted_kappa(
                    all_preds.cpu().numpy(),
                    all_labels.cpu().numpy()
                )
                self.log("train_qwk", qwk, prog_bar=True)
            
            self.training_step_outputs.clear()
        
        def validation_step(self, batch, batch_idx):
            images = batch["image"]
            labels = batch["target"]
            outputs = self(images).squeeze()
            
            # Handle single sample batch
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = self.criterion(outputs, labels.float())
            preds = outputs.round().clamp(0, 4).long()
            
            self.validation_step_outputs.append({
                "loss": loss.detach(),
                "preds": preds.detach(),
                "labels": labels.detach(),
                "outputs": outputs.detach(),
            })
            
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            return loss
        
        def on_validation_epoch_end(self):
            if self.validation_step_outputs:
                all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
                all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
                
                from src.utils import quadratic_weighted_kappa
                qwk = quadratic_weighted_kappa(
                    all_preds.cpu().numpy(),
                    all_labels.cpu().numpy()
                )
                accuracy = (all_preds == all_labels).float().mean().item()
                
                self.log("val_qwk", qwk, prog_bar=True)
                self.log("val_accuracy", accuracy, prog_bar=True)
            
            self.validation_step_outputs.clear()
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }

except ImportError:
    logger.error("pytorch-lightning not installed. Run: pip install pytorch-lightning")
    DRTrainingModule = None


def hydra_train():
    """Training with Hydra configuration management."""
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    
    from src.models import DRModel
    from src.datamodules import APTOSDataModule
    
    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def main(config: DictConfig):
        # Set seed
        pl.seed_everything(config.experiment.seed, workers=True)
        
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
        
        # Data module
        datamodule = APTOSDataModule(
            data_dir=config.paths.data_dir,
            train_csv=config.dataset.train_csv,
            image_dir=config.dataset.train_images,
            processed_dir=config.dataset.processed_images if config.dataset.preprocessing.ben_graham else None,
            batch_size=config.training.batch_size,
            num_workers=config.hardware.num_workers,
            image_size=config.dataset.input_size,
            val_split=config.dataset.validation.split_ratio,
            seed=config.experiment.seed,
        )
        
        # Model
        model = DRModel(
            backbone=config.model.backbone,
            num_classes=config.dataset.num_classes,
            head_type=config.model.head.type,
            pretrained=config.model.pretrained,
            dropout=config.model.head.dropout,
            pooling=config.model.pooling,
        )
        
        # Lightning module
        lightning_module = DRTrainingModule(
            model=model,
            learning_rate=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
            num_epochs=config.training.epochs,
        )
        
        # Callbacks
        checkpoint_dir = Path(config.paths.checkpoints_dir) / config.experiment.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="dr-{epoch:02d}-{val_qwk:.4f}",
                monitor="val_qwk",
                mode="max",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_qwk",
                mode="max",
                patience=config.training.early_stopping.patience,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        
        # Logger
        log_dir = Path(config.paths.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        loggers = [
            TensorBoardLogger(save_dir=log_dir, name="tensorboard"),
            CSVLogger(save_dir=log_dir, name="csv_logs"),
        ]
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            accelerator=config.hardware.accelerator,
            devices=config.hardware.devices,
            precision=config.hardware.precision,
            callbacks=callbacks,
            logger=loggers,
            accumulate_grad_batches=config.training.accumulate_grad_batches,
            gradient_clip_val=config.training.gradient.clip_val,
            log_every_n_steps=config.logging.log_every_n_steps,
        )
        
        # Train
        trainer.fit(lightning_module, datamodule=datamodule)
        
        logger.info(f"Best QWK: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR Detection Model")
    parser.add_argument("--quick", action="store_true", help="Quick training mode (recommended, no Hydra)")
    
    # Quick mode arguments
    parser.add_argument("--backbone", type=str, default="efficientnet_b5", help="Model backbone")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--image-size", type=int, default=456, help="Input image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--accumulate", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args, unknown = parser.parse_known_args()
    
    if args.quick:
        # Quick training mode (recommended)
        quick_train(args)
    else:
        # Hydra training mode
        hydra_train()
