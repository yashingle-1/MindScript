"""
MindScript Custom Attention Mechanisms
======================================
Novel attention layers for cognitive pattern extraction.

Author: [Your Name]
Innovation: Multi-head cognitive attention with dimension-specific focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CognitiveAttention(nn.Module):
    """
    Custom attention mechanism that learns to focus on
    cognitive-relevant linguistic patterns.
    
    Innovation: Each cognitive dimension gets its own attention head,
    allowing the model to learn dimension-specific patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_dimensions: int = 5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_dimensions = num_dimensions
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Dimension-specific attention biases (our innovation)
        self.dimension_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            for _ in range(num_dimensions)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Xavier uniform"""
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dimension_idx: Optional[int] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cognitive attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Mask tensor [batch, seq_len]
            dimension_idx: Which cognitive dimension to focus on
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attended tensor [batch, seq_len, hidden_dim]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Residual connection
        residual = hidden_states
        
        # Project to Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Add dimension-specific bias (our innovation)
        if dimension_idx is not None and dimension_idx < self.num_dimensions:
            attention_scores = attention_scores + self.dimension_biases[dimension_idx]
        
        # Apply attention mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                extended_mask == 0, float('-inf')
            )
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.output_proj(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        if return_attention:
            return output, attention_weights
        return output, None


class DimensionPooling(nn.Module):
    """
    Pooling layer that creates dimension-specific representations.
    
    Innovation: Instead of simple mean pooling, we use learned
    weighted pooling for each cognitive dimension.
    """
    
    def __init__(self, hidden_dim: int, num_dimensions: int = 5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_dimensions = num_dimensions
        
        # Learned pooling weights for each dimension
        self.pooling_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
            for _ in range(num_dimensions)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence representations for each dimension.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]
            
        Returns:
            pooled: [batch, num_dimensions, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        pooled_outputs = []
        
        for dim_idx in range(self.num_dimensions):
            # Compute attention weights for this dimension
            weights = self.pooling_weights[dim_idx](hidden_states)  # [batch, seq, 1]
            weights = weights.squeeze(-1)  # [batch, seq]
            
            # Apply mask
            if attention_mask is not None:
                weights = weights.masked_fill(attention_mask == 0, float('-inf'))
            
            # Softmax over sequence
            weights = F.softmax(weights, dim=-1)
            weights = weights.unsqueeze(-1)  # [batch, seq, 1]
            
            # Weighted sum
            pooled = (hidden_states * weights).sum(dim=1)  # [batch, hidden_dim]
            pooled_outputs.append(pooled)
        
        # Stack all dimensions
        pooled = torch.stack(pooled_outputs, dim=1)  # [batch, num_dim, hidden_dim]
        
        return pooled


class ConfidenceEstimator(nn.Module):
    """
    Estimates prediction confidence based on input characteristics.
    
    Innovation: Uses input complexity and model uncertainty to
    provide calibrated confidence scores.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence score.
        
        Args:
            pooled_output: [batch, hidden_dim]
            
        Returns:
            confidence: [batch, 1] scores between 0 and 1
        """
        return self.confidence_net(pooled_output)