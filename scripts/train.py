#!/usr/bin/env python3
"""
MindScript Training Script
==========================
Run this script to train the MindScript model.

Usage:
    python scripts/train.py --data data/essays.csv --epochs 10 --batch_size 16

Author: [Your Name]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mindscript.models.mindscript_model import MindScriptModel
from mindscript.data.dataset import DataProcessor
from mindscript.training.trainer import MindScriptTrainer
from mindscript.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train MindScript cognitive pattern analysis model"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/essays.csv",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--encoder",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained encoder name"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("🧠 MindScript - Cognitive Pattern Analysis")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Device: {config.device}")
    print("="*60 + "\n")
    
    # Check if data exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please download essays.csv from the data source")
        logger.info("wget https://raw.githubusercontent.com/jcl132/personality-prediction-from-text/master/essays.csv -O data/essays.csv")
        return
    
    # Initialize data processor
    logger.info("Preparing datasets...")
    processor = DataProcessor(tokenizer_name=args.encoder)
    
    train_dataset, val_dataset, test_dataset = processor.prepare_datasets(
        args.data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = MindScriptModel(
        encoder_name=args.encoder,
        num_dimensions=5,
        hidden_dim=768,
        num_attention_heads=8,
        dropout_rate=0.3
    )
    
    # Initialize trainer
    trainer = MindScriptTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=config.device,
        save_dir=args.save_dir,
        patience=args.patience
    )
    
    # Train
    history = trainer.train()
    
    # Final evaluation
    print("\n" + "="*60)
    print("📊 Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(Path(args.save_dir) / "mindscript_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate
    trainer.val_loader = test_loader
    test_loss, test_correlations = trainer.validate()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print("\nTest Correlations:")
    for dim_name, corr in test_correlations.items():
        print(f"  {dim_name}: {corr:.4f}")
    
    avg_corr = sum(test_correlations.values()) / len(test_correlations)
    print(f"\nAverage Test Correlation: {avg_corr:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print(f"Best model saved to: {args.save_dir}/mindscript_best.pt")
    print("="*60)


if __name__ == "__main__":
    main()