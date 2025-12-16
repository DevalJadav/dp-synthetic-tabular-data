# models/dp_labeler.py
# DP-trained labeler to assign the target column (income) to generated features.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DPLabeler(nn.Module):
    """
    Small MLP classifier producing a single logit.
    We train it with DP-SGD (noise added to gradients) in training scripts.
    """
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [
                spectral_norm(nn.Linear(d, hidden)),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = hidden
        layers += [spectral_norm(nn.Linear(d, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B,)
