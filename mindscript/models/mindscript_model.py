"""
MindScript Core Model
=====================
The main neural network for cognitive pattern analysis.

Author: [Your Name]
Architecture: Transformer Encoder + Custom Cognitive Attention + Dimension Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import logging

from .attention_layers import CognitiveAttention, DimensionPooling, ConfidenceEstimator

logger = logging.getLogger(__name__)


class MindScriptModel(nn.Module):
    """
    MindScript: Cognitive Pattern Analysis Model
    
    Architecture:
    1. Pre-trained Transformer Encoder (DistilBERT/BERT)
    2. Custom Cognitive Attention Layer
    3. Dimension-specific Pooling
    4. Multi-head Output for each cognitive dimension
    5. Confidence Estimation
    
    Innovations:
    - Dimension-specific attention biases
    - Learned weighted pooling per dimension
    - Calibrated confidence estimation
    - Attention visualization for explainability
    """
    
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        num_dimensions: int = 5,
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.3,
        dimension_names: Optional[List[str]] = None,
        freeze_encoder_layers: int = 0
    ):
        super().__init__()
        
        self.num_dimensions = num_dimensions
        self.hidden_dim = hidden_dim
        self.dimension_names = dimension_names or [
            "analytical", "creative", "social", "structured", "emotional"
        ]
        
        # Load pre-trained encoder
        logger.info(f"Loading encoder: {encoder_name}")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Get encoder hidden size
        encoder_hidden_size = self.encoder.config.hidden_size
        
        # Projection layer if encoder size differs
        if encoder_hidden_size != hidden_dim:
            self.projection = nn.Linear(encoder_hidden_size, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder layers if specified
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        
        # Custom cognitive attention
        self.cognitive_attention = CognitiveAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            num_dimensions=num_dimensions
        )
        
        # Dimension-specific pooling
        self.dimension_pooling = DimensionPooling(
            hidden_dim=hidden_dim,
            num_dimensions=num_dimensions
        )
        
        # Output heads for each dimension
        self.dimension_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_dimensions)
        ])
        
        # Global pooling for confidence estimation
        self.global_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator(hidden_dim // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"MindScript model initialized with {self._count_parameters():,} parameters")
    
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers"""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        if hasattr(self.encoder, 'transformer'):
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        else:
            return
        
        for layer_idx in range(min(num_layers, len(layers))):
            for param in layers[layer_idx].parameters():
                param.requires_grad = False
    
    def _init_weights(self):
        """Initialize custom layer weights"""
        for head in self.dimension_heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MindScript model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_attention: Return attention weights
            return_embeddings: Return hidden embeddings
            
        Returns:
            Dictionary containing:
            - dimensions: Dict of dimension scores
            - confidence: Confidence score
            - attention_weights: Optional attention weights
            - embeddings: Optional embeddings
        """
        batch_size = input_ids.shape[0]
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output and project
        hidden_states = encoder_outputs.last_hidden_state
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply cognitive attention
        attended_states, attention_weights = self.cognitive_attention(
            hidden_states,
            attention_mask=attention_mask,
            return_attention=return_attention
        )
        
        # Dimension-specific pooling
        dimension_pooled = self.dimension_pooling(
            attended_states,
            attention_mask=attention_mask
        )  # [batch, num_dimensions, hidden_dim]
        
        # Get dimension scores
        dimension_scores = {}
        for dim_idx, dim_name in enumerate(self.dimension_names):
            dim_hidden = dimension_pooled[:, dim_idx, :]  # [batch, hidden_dim]
            score = self.dimension_heads[dim_idx](dim_hidden)  # [batch, 1]
            dimension_scores[dim_name] = score.squeeze(-1)  # [batch]
        
        # Global pooling for confidence
        global_pooled = attended_states.mean(dim=1)  # [batch, hidden_dim]
        global_pooled = self.global_pooler(global_pooled)  # [batch, hidden_dim // 2]
        
        # Estimate confidence
        confidence = self.confidence_estimator(global_pooled).squeeze(-1)  # [batch]
        
        # Prepare output
        output = {
            "dimensions": dimension_scores,
            "confidence": confidence
        }
        
        if return_attention and attention_weights is not None:
            output["attention_weights"] = attention_weights
        
        if return_embeddings:
            output["embeddings"] = attended_states.mean(dim=1)
        
        return output
    
    def predict(
        self,
        text: str,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Predict cognitive dimensions for a single text.
        
        Args:
            text: Input text string
            device: Device to run prediction on
            
        Returns:
            Dictionary with dimension scores and confidence
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        # Convert to Python types
        result = {
            "dimensions": {
                name: float(score.cpu().numpy())
                for name, score in outputs["dimensions"].items()
            },
            "confidence": float(outputs["confidence"].cpu().numpy()),
            "dominant_dimension": max(
                outputs["dimensions"].items(),
                key=lambda x: x[1].item()
            )[0]
        }
        
        return result
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a pre-trained MindScript model"""
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract config
        config = checkpoint.get("config", {})
        config.update(kwargs)
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_pretrained(self, save_path: str, config: Optional[Dict] = None):
        """Save model to path"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "num_dimensions": self.num_dimensions,
                "hidden_dim": self.hidden_dim,
                "dimension_names": self.dimension_names
            }
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")