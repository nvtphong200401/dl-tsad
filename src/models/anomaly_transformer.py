"""Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

Faithful reimplementation based on:
    Xu et al., "Anomaly Transformer: Time Series Anomaly Detection
    with Association Discrepancy", ICLR 2022
    Reference: https://github.com/thuml/Anomaly-Transformer

Key components:
    - TokenEmbedding (Conv1d) + PositionalEmbedding (sinusoidal)
    - AnomalyAttention with learnable sigma for prior association (per-head)
    - Minimax training strategy (loss1 minimizes series-association,
      loss2 maximizes it)
    - Anomaly score = softmax(-series_loss - prior_loss) * recon_error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (fixed, not learned)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float32)
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """1-D convolution embedding (kernel=3, circular padding)."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> conv expects (B, C, L)
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """Token + Positional embedding with dropout."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.value_embedding(x) + self.position_embedding(x))


# ---------------------------------------------------------------------------
# Anomaly Attention (with learnable sigma)
# ---------------------------------------------------------------------------
class AnomalyAttention(nn.Module):
    """Anomaly Attention with learnable prior association.

    The prior is a Gaussian kernel whose sigma is *learned* per-head via a
    linear projection, making it adaptive to the data.
    """

    def __init__(self, win_size: int, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        # Pre-compute pairwise distances |i - j|
        distances = torch.zeros((win_size, win_size), dtype=torch.float32)
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        self.register_buffer('distances', distances)

    def forward(
        self,
        queries: torch.Tensor,   # (B, L, H, d_k)
        keys: torch.Tensor,      # (B, S, H, d_k)
        values: torch.Tensor,    # (B, S, H, d_v)
        sigma: torch.Tensor,     # (B, L, H)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        # Attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        # --- Learnable prior (per-head sigma) ---
        sigma = sigma.transpose(1, 2)                    # (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, S)   # (B, H, L, S)

        prior = self.distances.unsqueeze(0).unsqueeze(0)  # (1, 1, L, S)
        prior = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * \
                torch.exp(-prior ** 2 / (2 * sigma ** 2))

        # Series association (softmax of attention scores)
        series = self.dropout(torch.softmax(attn, dim=-1))

        # Weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", series, values)

        return V.contiguous(), series, prior


class AttentionLayer(nn.Module):
    """Wraps AnomalyAttention with Q/K/V/sigma projections."""

    def __init__(self, attention: AnomalyAttention,
                 d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior = self.inner_attention(
            queries, keys, values, sigma)
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Single Anomaly Transformer encoder layer.

    Uses Conv1d (kernel=1) for FFN as in the original paper.
    """

    def __init__(self, attention_layer: AttentionLayer, d_model: int,
                 d_ff: int = None, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_x, attn, prior = self.attention(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, prior


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class AnomalyTransformer(nn.Module):
    """Anomaly Transformer with Association Discrepancy (ICLR 2022).

    Args:
        win_size:   Window (sequence) length
        input_dim:  Number of input channels
        d_model:    Model hidden dimension
        n_heads:    Number of attention heads
        n_layers:   Number of encoder layers
        d_ff:       FFN hidden dimension (default: 4 * d_model)
        dropout:    Dropout rate
        activation: 'gelu' or 'relu'
    """

    def __init__(
        self,
        win_size: int,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.win_size = win_size

        # Embedding
        self.embedding = DataEmbedding(input_dim, d_model, dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(win_size,
                                     attention_dropout=dropout),
                    d_model, n_heads,
                ),
                d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.projection = nn.Linear(d_model, input_dim, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, L, input_dim)

        Returns:
            output:      (B, L, input_dim) — reconstruction
            series_list: list of (B, H, L, L) — series-associations per layer
            prior_list:  list of (B, H, L, L) — prior-associations per layer
        """
        enc_out = self.embedding(x)

        series_list = []
        prior_list = []
        for layer in self.encoder_layers:
            enc_out, series, prior = layer(enc_out)
            series_list.append(series)
            prior_list.append(prior)

        enc_out = self.norm(enc_out)
        output = self.projection(enc_out)

        return output, series_list, prior_list


# ---------------------------------------------------------------------------
# Loss utilities (minimax strategy from the paper)
# ---------------------------------------------------------------------------
def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Element-wise KL: p * (log p - log q), summed over last dim,
    averaged over heads.

    Args:
        p, q: (B, H, L, L) probability distributions

    Returns:
        (B, L) KL divergence per position (averaged over heads)
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def compute_association_discrepancy(
    series_list: List[torch.Tensor],
    prior_list: List[torch.Tensor],
    win_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute series_loss and prior_loss across all layers.

    Implements the *asymmetric* KL computation from the original paper:
    series_loss detaches prior, prior_loss detaches series.

    Args:
        series_list: List of (B, H, L, L) per layer
        prior_list:  List of (B, H, L, L) per layer
        win_size:    Window size (for normalizing prior)

    Returns:
        series_loss, prior_loss: scalar tensors
    """
    series_loss = 0.0
    prior_loss = 0.0

    for u in range(len(prior_list)):
        # Normalize prior along last dim
        prior_norm = prior_list[u] / torch.unsqueeze(
            torch.sum(prior_list[u], dim=-1), dim=-1
        ).repeat(1, 1, 1, win_size)

        series_loss += (
            torch.mean(my_kl_loss(series_list[u], prior_norm.detach()))
            + torch.mean(my_kl_loss(prior_norm.detach(), series_list[u]))
        )
        prior_loss += (
            torch.mean(my_kl_loss(prior_norm, series_list[u].detach()))
            + torch.mean(my_kl_loss(series_list[u].detach(), prior_norm))
        )

    series_loss /= len(prior_list)
    prior_loss /= len(prior_list)
    return series_loss, prior_loss


def compute_anomaly_score(
    x: torch.Tensor,
    output: torch.Tensor,
    series_list: List[torch.Tensor],
    prior_list: List[torch.Tensor],
    win_size: int,
    temperature: float = 50.0,
) -> np.ndarray:
    """Per-timestep anomaly score (for inference).

    Score = softmax(-series_loss - prior_loss) * recon_error

    Args:
        x:           (B, L, D) input
        output:      (B, L, D) reconstruction
        series_list: per-layer series associations
        prior_list:  per-layer prior associations
        win_size:    window size
        temperature: scaling factor for KL scores

    Returns:
        scores: (B, L) numpy array
    """
    criterion = nn.MSELoss(reduction='none')
    recon_error = torch.mean(criterion(x, output), dim=-1)  # (B, L)

    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior_list)):
        prior_norm = prior_list[u] / torch.unsqueeze(
            torch.sum(prior_list[u], dim=-1), dim=-1
        ).repeat(1, 1, 1, win_size)

        if u == 0:
            series_loss = my_kl_loss(
                series_list[u], prior_norm.detach()) * temperature
            prior_loss = my_kl_loss(
                prior_norm, series_list[u].detach()) * temperature
        else:
            series_loss += my_kl_loss(
                series_list[u], prior_norm.detach()) * temperature
            prior_loss += my_kl_loss(
                prior_norm, series_list[u].detach()) * temperature

    metric = torch.softmax(-series_loss - prior_loss, dim=-1)  # (B, L)
    score = metric * recon_error
    return score.detach().cpu().numpy()
