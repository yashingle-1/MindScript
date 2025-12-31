"""
MindScript Configuration
========================
Central configuration for all model and training parameters.

Author: [Your Name]
Project: MindScript - Cognitive Pattern Analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # Base transformer
    encoder_name: str = "distilbert-base-uncased"
    hidden_dim: int = 768
    num_attention_heads: int = 8
    dropout_rate: float = 0.3
    
    # Cognitive dimensions (our unique framework)
    cognitive_dimensions: List[str] = field(default_factory=lambda: [
        "analytical",      # Logical thinking patterns
        "creative",        # Imaginative expression
        "social",          # Interpersonal orientation
        "structured",      # Organization preference
        "emotional"        # Emotional expression
    ])
    
    num_dimensions: int = 5
    
    # Output settings
    use_confidence: bool = True
    use_attention_viz: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    # Basic training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimization
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Data
    max_sequence_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001


@dataclass
class PathConfig:
    """File paths configuration"""
    
    # Directories
    root_dir: Path = Path(".")
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    
    # Files
    raw_data: Path = Path("data/essays.csv")
    processed_data: Path = Path("data/processed/processed_data.pt")
    best_model: Path = Path("models/mindscript_best.pt")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.data_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference settings"""
    
    # Performance
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available()
    use_fp16: bool = True
    
    # Caching
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Output
    return_attention: bool = False
    return_embeddings: bool = False
    confidence_threshold: float = 0.6


class MindScriptConfig:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.paths = PathConfig()
        self.inference = InferenceConfig()
        
        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()},
            "inference": self.inference.__dict__,
            "device": str(self.device)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MindScriptConfig":
        """Create config from dictionary"""
        config = cls()
        # Update configs from dict
        return config


# Global config instance
config = MindScriptConfig()