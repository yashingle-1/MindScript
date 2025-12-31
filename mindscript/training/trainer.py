"""
MindScript Training Module
==========================
Complete training pipeline with logging and evaluation.

Author: [Your Name]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MindScriptTrainer:
    """
    Training class for MindScript model.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Best model checkpointing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 10,
        device: Optional[torch.device] = None,
        save_dir: str = "models",
        patience: int = 3,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "dimension_correlations": []
        }
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Total training steps: {total_steps}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self._compute_loss(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self._compute_loss(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = {dim: [] for dim in self.model.dimension_names}
        all_labels = {dim: [] for dim in self.model.dimension_names}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self._compute_loss(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                for dim_idx, dim_name in enumerate(self.model.dimension_names):
                    all_preds[dim_name].extend(
                        outputs["dimensions"][dim_name].cpu().numpy()
                    )
                    all_labels[dim_name].extend(
                        labels[:, dim_idx].cpu().numpy()
                    )
        
        # Calculate correlations
        correlations = {}
        for dim_name in self.model.dimension_names:
            corr = np.corrcoef(all_preds[dim_name], all_labels[dim_name])[0, 1]
            correlations[dim_name] = corr if not np.isnan(corr) else 0.0
        
        avg_loss = total_loss / num_batches
        
        return avg_loss, correlations
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute multi-task loss"""
        total_loss = 0
        
        for dim_idx, dim_name in enumerate(self.model.dimension_names):
            dim_pred = outputs["dimensions"][dim_name]
            dim_label = labels[:, dim_idx]
            total_loss += self.criterion(dim_pred, dim_label)
        
        # Average across dimensions
        total_loss = total_loss / len(self.model.dimension_names)
        
        return total_loss
    
    def train(self) -> Dict:
        """Complete training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, correlations = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["dimension_correlations"].append(correlations)
            
            # Print results
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("\nDimension Correlations:")
            for dim_name, corr in correlations.items():
                print(f"  {dim_name}: {corr:.4f}")
            
            avg_corr = np.mean(list(correlations.values()))
            print(f"\nAverage Correlation: {avg_corr:.4f}")
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, "best")
                print("✅ New best model saved!")
            else:
                self.patience_counter += 1
                print(f"⚠️ No improvement ({self.patience_counter}/{self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⛔ Early stopping at epoch {epoch + 1}")
                break
        
        # Save final model
        self._save_checkpoint(epoch, "final")
        
        # Save training history
        self._save_history()
        
        logger.info("Training complete!")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, tag: str):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "num_dimensions": self.model.num_dimensions,
                "hidden_dim": self.model.hidden_dim,
                "dimension_names": self.model.dimension_names
            }
        }
        
        path = self.save_dir / f"mindscript_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def _save_history(self):
        """Save training history"""
        # Convert numpy arrays to lists
        history_json = {
            "train_loss": self.history["train_loss"],
            "val_loss": self.history["val_loss"],
            "dimension_correlations": [
                {k: float(v) for k, v in corr.items()}
                for corr in self.history["dimension_correlations"]
            ]
        }
        
        path = self.save_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(history_json, f, indent=2)
        
        logger.info(f"Training history saved to {path}")