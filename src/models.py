"""
models.py

All the model architectures we tried:
- MLP (mean pool baseline, ignores time)
- LSTM (bidirectional)
- CNN + LSTM (our best model)
- Transformer (didn't work great with small data)
"""

import math
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


# ── MLP ───────────────────────────────────────────────────────────────────────
# simplest possible model — just averages all frames and runs it through FC layers
# spoiler: it doesn't work because it throws away all the temporal info

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
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

    def forward(self, x):
        # x: (B, T, 66) — average over frames first, then classify
        x = x.mean(dim=1)
        return self.net(x)


# ── LSTM ──────────────────────────────────────────────────────────────────────
# reads the sequence frame by frame in both directions
# then uses the final hidden state from each direction for classification

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
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

    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h, _) = self.lstm(x)
        # h shape is (num_layers * 2, B, hidden_dim) because bidirectional
        # we want the last forward and last backward hidden states
        h_fwd = h[-2]  # last layer, forward direction
        h_bwd = h[-1]  # last layer, backward direction
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.classifier(h_cat)


# ── CNN + LSTM ────────────────────────────────────────────────────────────────
# runs 1D conv first to pick up local patterns (e.g. "elbow bending over 5 frames")
# then feeds that into the LSTM so it sees both local and global structure
# this turned out to be the best model

class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        cnn_channels=64,
        kernel_size=3,
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
            cnn_channels * 2,
            hidden_dim,
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

    def forward(self, x):
        # conv1d wants (B, channels, T) but we have (B, T, channels)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # back to (B, T, cnn_channels*2)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.classifier(h_cat)


# ── Transformer ───────────────────────────────────────────────────────────────
# this code was AI generated
# the positional encoding and CLS token pattern is straight from the BERT paper
# we tried this but it underperformed with only 100 clips — needs way more data

class PositionalEncoding(nn.Module):
    # this code was AI generated
    def __init__(self, d_model, max_len=512, dropout=0.1):
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

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    # this code was AI generated
    def __init__(
        self,
        input_dim=config.INPUT_DIM,
        d_model=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        ff_dim=config.FF_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        max_seq_len=config.SEQUENCE_LEN + 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # learnable CLS token prepended to the sequence for classification
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

    def forward(self, x):
        x = self.input_proj(x)                             # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                     # (B, T+1, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = x[:, 0]   # read off the CLS token
        return self.classifier(cls_out)


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mlp": MLPClassifier,
    "lstm": LSTMClassifier,
    "cnn_lstm": CNNLSTMClassifier,
    "transformer": TransformerClassifier,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}', pick from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
