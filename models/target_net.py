# models/target_net.py
import torch
import torch.nn as nn

class TargetNet(nn.Module):
    """
    Simple MLP classifier to predict income (binary) from features.
    Output: logits (use sigmoid at inference).
    """
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
