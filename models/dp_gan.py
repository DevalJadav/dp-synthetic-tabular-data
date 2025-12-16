# models/dp_gan.py
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, final_act: nn.Module | None = None):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        if final_act is not None:
            layers.append(final_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dp_gan(
    data_dim: int,
    noise_dim: int,
    gen_hidden: List[int],
    disc_hidden: List[int],
) -> Tuple[nn.Module, nn.Module]:
    G = MLP(noise_dim, gen_hidden, data_dim, final_act=nn.Tanh())
    D = MLP(data_dim, disc_hidden, 1, final_act=None)
    return G, D


def add_dp_noise(model: nn.Module, max_grad_norm: float, noise_multiplier: float, batch_size: int) -> None:
    """
    Minimal DP-SGD style: clip each parameter gradient then add Gaussian noise.
    """
    # clip global norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    std = noise_multiplier * max_grad_norm / max(1, int(batch_size))
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.add_(torch.randn_like(p.grad) * std)
