"""
Stage 04 – Temporal Models

Implements three main architectures and one static baseline:
  1. LSTMClassifier      — bidirectional LSTM over keypoint sequences
  2. TransformerClassifier — multi-head self-attention encoder
  3. CNNLSTMClassifier   — 1-D CNN for local patterns + LSTM for sequence
  4. MLPClassifier       — static baseline (mean-pooled keypoints)
"""

import math
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


# ── 1. LSTM ───────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM that reads a (T, input_dim) keypoint sequence and
    outputs a form-quality class logit.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        out, (h, _) = self.lstm(x)
        # Concatenate last forward + backward hidden states
        h_fwd = h[-2]   # (B, hidden_dim) — last layer, forward
        h_bwd = h[-1]   # (B, hidden_dim) — last layer, backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden_dim)
        return self.classifier(h_cat)


# ── 2. Transformer ────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder that attends over the temporal keypoint sequence.
    A learnable [CLS] token aggregates the sequence for classification.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        d_model: int = config.HIDDEN_DIM,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_LAYERS,
        ff_dim: int = config.FF_DIM,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        max_seq_len: int = config.SEQUENCE_LEN + 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = self.input_proj(x)                           # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, T+1, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                              # (B, T+1, d_model)
        cls_out = x[:, 0]                                # (B, d_model) — CLS token
        return self.classifier(cls_out)


# ── 3. CNN + LSTM ─────────────────────────────────────────────────────────────

class CNNLSTMClassifier(nn.Module):
    """
    1-D convolution captures local pose patterns (e.g., elbow angle at release);
    LSTM models the temporal arc of the pushup rep.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        cnn_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            cnn_channels * 2, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = x.permute(0, 2, 1)          # (B, input_dim, T) for Conv1d
        x = self.cnn(x)                  # (B, cnn_channels*2, T)
        x = x.permute(0, 2, 1)          # (B, T, cnn_channels*2)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.classifier(h_cat)


# ── 4. MLP (static baseline) ──────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Simple MLP that operates on mean-pooled keypoints.
    Used as a static baseline — ignores temporal order.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) → mean pool over time
        x = x.mean(dim=1)    # (B, input_dim)
        return self.net(x)


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "transformer": TransformerClassifier,
    "cnn_lstm": CNNLSTMClassifier,
    "mlp": MLPClassifier,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
