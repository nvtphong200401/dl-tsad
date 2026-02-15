"""AER: Auto-Encoder with Regression for Time Series Anomaly Detection

Reference: Wong et al., IEEE Big Data 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AERModel(nn.Module):
    """BiLSTM Encoder-Decoder + Regressor for AER

    Combines reconstruction (autoencoder) and prediction (regressor)
    with bidirectional scoring for robust anomaly detection.

    Args:
        input_dim: Input dimension (number of channels)
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder (BiLSTM)
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder (BiLSTM)
        self.decoder = nn.LSTM(
            hidden_dim * 2,  # Bidirectional encoder output
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Reconstruction output
        self.recon_fc = nn.Linear(hidden_dim * 2, input_dim)

        # Forward predictor (LSTM)
        self.predictor_forward = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.pred_fc_forward = nn.Linear(hidden_dim, input_dim)

        # Backward predictor (LSTM on reversed sequence)
        self.predictor_backward = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.pred_fc_backward = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            recon: Reconstructed sequence (batch, seq_len, input_dim)
            pred_forward: Forward predictions (batch, seq_len, input_dim)
            pred_backward: Backward predictions (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        encoded, _ = self.encoder(x)

        # Decode (reconstruct)
        decoded, _ = self.decoder(encoded)
        recon = self.recon_fc(decoded)

        # Forward prediction
        pred_f_hidden, _ = self.predictor_forward(x)
        pred_forward = self.pred_fc_forward(pred_f_hidden)

        # Backward prediction (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        pred_b_hidden, _ = self.predictor_backward(x_reversed)
        pred_backward = self.pred_fc_backward(pred_b_hidden)
        pred_backward = torch.flip(pred_backward, dims=[1])  # Flip back

        return recon, pred_forward, pred_backward

    def compute_loss(self,
                    x: torch.Tensor,
                    recon: torch.Tensor,
                    pred_forward: torch.Tensor,
                    pred_backward: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
        """Compute joint loss

        Args:
            x: Original input (batch, seq_len, input_dim)
            recon: Reconstructed sequence
            pred_forward: Forward predictions
            pred_backward: Backward predictions
            alpha: Weight between reconstruction and prediction (0-1)

        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)

        # Forward prediction loss (predict next timestep)
        pred_loss_f = F.mse_loss(pred_forward[:, :-1], x[:, 1:])

        # Backward prediction loss (predict previous timestep)
        pred_loss_b = F.mse_loss(pred_backward[:, 1:], x[:, :-1])

        # Combined prediction loss
        pred_loss = (pred_loss_f + pred_loss_b) / 2

        # Joint loss
        loss = alpha * recon_loss + (1 - alpha) * pred_loss

        return loss
