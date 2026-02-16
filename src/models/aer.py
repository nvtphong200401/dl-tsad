"""AER: Auto-Encoder with Regression for Time Series Anomaly Detection

Reference: Wong et al., IEEE Big Data 2022
Canonical implementation: https://github.com/sintel-dev/Orion (orion.primitives.aer)

Architecture (following the Orion reference):
    - Encoder: BiLSTM → latent vector (final hidden states concatenated)
    - Decoder: RepeatVector(seq_len + 2) → BiLSTM → TimeDistributed(Dense)
    - The decoder output is split into three parts:
        ry: first timestep  → backward regression (predicts timestep before the window)
        y:  middle timesteps → reconstruction   (reconstructs the trimmed input)
        fy: last timestep   → forward regression (predicts timestep after the window)

Training protocol:
    - Input to model:  x_trimmed = x_full[:, 1:-1, :]  (first & last removed)
    - Targets:         ry_target = x_full[:, 0],  y_target = x_full[:, 1:-1],  fy_target = x_full[:, -1]
    - Loss = (reg_ratio/2)*MSE(ry, ry_target) + (1-reg_ratio)*MSE(y, y_target) + (reg_ratio/2)*MSE(fy, fy_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AERModel(nn.Module):
    """Auto-Encoder with Bidirectional Regression (AER)

    A single encoder-decoder that simultaneously reconstructs its input
    and predicts adjacent timesteps via bidirectional regression.

    Args:
        input_dim:  Number of input channels / features
        lstm_units: Hidden units per direction for BiLSTM
                    (default 30, matching the Orion reference)
        num_layers: Number of stacked LSTM layers (default 1)
        dropout:    Dropout rate between LSTM layers
    """

    def __init__(self,
                 input_dim: int,
                 lstm_units: int = 30,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()

        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.latent_dim = lstm_units * 2          # BiLSTM concat

        # ---- Encoder ---- BiLSTM  →  latent vector  ----
        self.encoder = nn.LSTM(
            input_dim, lstm_units, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # ---- Decoder ---- RepeatVector → BiLSTM → Dense  ----
        self.decoder_lstm = nn.LSTM(
            self.latent_dim, lstm_units, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # TimeDistributed(Dense(input_dim))
        self.output_fc = nn.Linear(self.latent_dim, input_dim)

    # -- kept for backward-compat with metadata loading (hidden_dim alias) --
    @property
    def hidden_dim(self):
        return self.lstm_units

    def forward(self, x_trimmed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            x_trimmed: (batch, seq_len, input_dim)
                       The trimmed input (first & last timesteps already removed).

        Returns:
            ry: (batch, input_dim)           — backward regression
            y:  (batch, seq_len, input_dim)  — reconstruction
            fy: (batch, input_dim)           — forward regression
        """
        batch_size, seq_len, _ = x_trimmed.shape
        decode_len = seq_len + 2     # RepeatVector(trimmed_len + 2) = original window size

        # Encode  →  bottleneck vector
        _, (h_n, _) = self.encoder(x_trimmed)
        # h_n: (num_layers * num_directions, batch, lstm_units)
        h_forward = h_n[-2]          # last-layer forward  (batch, lstm_units)
        h_backward = h_n[-1]         # last-layer backward (batch, lstm_units)
        latent = torch.cat([h_forward, h_backward], dim=-1)   # (batch, latent_dim)

        # Decode  →  RepeatVector + BiLSTM + Dense
        repeated = latent.unsqueeze(1).expand(-1, decode_len, -1)  # (batch, decode_len, latent_dim)
        decoded, _ = self.decoder_lstm(repeated)                   # (batch, decode_len, latent_dim)
        output = self.output_fc(decoded)                           # (batch, decode_len, input_dim)

        # Split into [ry, y, fy]
        ry = output[:, 0]            # (batch, input_dim)      backward regression
        y  = output[:, 1:-1]         # (batch, seq_len, input_dim) reconstruction
        fy = output[:, -1]           # (batch, input_dim)      forward regression

        return ry, y, fy

    def compute_loss(self,
                     x_full: torch.Tensor,
                     ry: torch.Tensor,
                     y: torch.Tensor,
                     fy: torch.Tensor,
                     reg_ratio: float = 0.5) -> torch.Tensor:
        """Compute AER loss with proper weighting (matches Orion reference)

        Args:
            x_full:    Original *full* windows (batch, window_size, input_dim)
            ry:        Backward regression output  (batch, input_dim)
            y:         Reconstruction output        (batch, window_size-2, input_dim)
            fy:        Forward regression output    (batch, input_dim)
            reg_ratio: Regression-vs-reconstruction ratio (default 0.5)

        Returns:
            Weighted combined loss scalar
        """
        target_ry = x_full[:, 0]       # first timestep
        target_y  = x_full[:, 1:-1]    # middle portion
        target_fy = x_full[:, -1]      # last timestep

        loss_ry = F.mse_loss(ry, target_ry)
        loss_y  = F.mse_loss(y,  target_y)
        loss_fy = F.mse_loss(fy, target_fy)

        # Reference loss_weights = [reg_ratio/2, 1-reg_ratio, reg_ratio/2]
        loss = (reg_ratio / 2) * loss_ry + (1 - reg_ratio) * loss_y + (reg_ratio / 2) * loss_fy
        return loss
