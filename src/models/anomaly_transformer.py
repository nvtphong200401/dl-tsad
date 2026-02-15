"""Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

Reference: Xu et al., ICLR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class AnomalyTransformer(nn.Module):
    """Anomaly Transformer with Association Discrepancy

    Key innovation: Compares learned series-association with prior-association
    (Gaussian kernel) to detect anomalies.

    Args:
        input_dim: Input dimension (number of channels)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            output: Reconstructed sequence (batch, seq_len, input_dim)
            series_association: Learned associations (batch, seq_len, seq_len)
            prior_association: Prior (Gaussian kernel) (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to d_model
        x_proj = self.input_projection(x)

        # Compute prior association (Gaussian kernel)
        prior_association = self._compute_prior_association(seq_len, x.device)
        prior_association = prior_association.unsqueeze(0).repeat(batch_size, 1, 1)

        # Pass through transformer layers
        series_associations = []
        for layer in self.transformer_layers:
            x_proj, series_assoc = layer(x_proj)
            series_associations.append(series_assoc)

        # Average series associations across layers
        series_association = torch.stack(series_associations).mean(dim=0)

        # Project back to input dimension
        output = self.output_projection(x_proj)

        return output, series_association, prior_association

    def _compute_prior_association(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute Gaussian kernel as prior

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Prior association matrix (seq_len, seq_len)
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        sigma = seq_len / 6  # Hyperparameter from paper
        prior = torch.exp(-distances / (2 * sigma ** 2))
        # Normalize rows
        prior = prior / prior.sum(dim=-1, keepdim=True)
        return prior

    def compute_loss(self,
                    output: torch.Tensor,
                    target: torch.Tensor,
                    series_assoc: torch.Tensor,
                    prior_assoc: torch.Tensor,
                    lambda_assoc: float = 1.0) -> torch.Tensor:
        """Compute association discrepancy loss

        Args:
            output: Model output
            target: Target sequence
            series_assoc: Learned associations
            prior_assoc: Prior associations
            lambda_assoc: Weight for association discrepancy

        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(output, target)

        # Association discrepancy (KL divergence)
        # Add small epsilon for numerical stability
        series_assoc = series_assoc + 1e-8
        prior_assoc = prior_assoc + 1e-8

        assoc_discrepancy = F.kl_div(
            series_assoc.log(),
            prior_assoc,
            reduction='batchmean'
        )

        # During training on normal data, minimize both
        # (low reconstruction error + associations match prior)
        loss = recon_loss + lambda_assoc * assoc_discrepancy

        return loss


class AnomalyTransformerLayer(nn.Module):
    """Single Anomaly Transformer layer"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            output: Transformed tensor (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, seq_len, seq_len)
        """
        # Self-attention with residual
        attn_output, attn_weights = self.attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=True
        )
        x = self.norm1(x + attn_output)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights
