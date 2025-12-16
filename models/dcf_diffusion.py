# models/dcf_diffusion.py
# Latent diffusion for tabular data with an autoencoder + denoiser.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, use_spectral_norm: bool = False):
        super().__init__()
        lin1 = nn.Linear(dim, dim)
        lin2 = nn.Linear(dim, dim)
        if use_spectral_norm:
            lin1 = spectral_norm(lin1)
            lin2 = spectral_norm(lin2)
        self.lin1 = lin1
        self.lin2 = lin2
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.lin1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.view(-1, 1))


class MLPStack(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int,
                 dropout: float = 0.1, use_spectral_norm: bool = True):
        super().__init__()
        lin_in = nn.Linear(in_dim, width)
        lin_out = nn.Linear(width, out_dim)
        if use_spectral_norm:
            lin_in = spectral_norm(lin_in)
            lin_out = spectral_norm(lin_out)
        self.in_proj = lin_in
        self.out_proj = lin_out
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([
            ResidualBlock(width, dropout=dropout, use_spectral_norm=use_spectral_norm)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        for b in self.blocks:
            h = b(h)
        return self.out_proj(h)


class TabularDCF(nn.Module):
    """
    Autoencoder + latent diffusion denoiser.

    Implements:
      - loss(x) -> dict(total, diff, recon)
      - sample(n=..., device=...) -> returns decoded x_hat (feature space)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        time_emb_dim: int = 64,
        width: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
        T: int = 100,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.use_spectral_norm = bool(use_spectral_norm)

        self.T = int(T)
        self.recon_weight = float(recon_weight)

        # AE
        self.encoder = MLPStack(self.input_dim, self.latent_dim, self.width, self.depth, self.dropout, self.use_spectral_norm)
        self.decoder = MLPStack(self.latent_dim, self.input_dim, self.width, max(1, self.depth), self.dropout, self.use_spectral_norm)

        # Denoiser in latent
        self.time_embed = TimeEmbedding(time_emb_dim=self.time_emb_dim)
        self.denoiser = MLPStack(self.latent_dim + self.time_emb_dim, self.latent_dim, self.width, self.depth, self.dropout, self.use_spectral_norm)

        # schedule buffers (created lazily on correct device)
        self.register_buffer("_betas", torch.empty(0), persistent=False)
        self.register_buffer("_alphas", torch.empty(0), persistent=False)
        self.register_buffer("_alpha_cumprod", torch.empty(0), persistent=False)

    def _ensure_schedule(self, device: torch.device):
        if self._betas.numel() == self.T and self._betas.device == device:
            return

        betas = torch.linspace(1e-4, 0.02, self.T, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self._betas = betas
        self._alphas = alphas
        self._alpha_cumprod = alpha_cumprod

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # predict noise in latent
        t_emb = self.time_embed(t)
        h = torch.cat([z_t, t_emb], dim=1)
        return self.denoiser(h)

    def loss(self, x: torch.Tensor) -> dict:
        """
        x: [B, input_dim]
        returns {"total":..., "diff":..., "recon":...}
        """
        device = x.device
        self._ensure_schedule(device)

        # AE recon
        z0 = self.encode(x)
        x_rec = self.decode(z0)
        recon = F.mse_loss(x_rec, x)

        # diffusion in latent
        B = x.size(0)
        t_idx = torch.randint(0, self.T, (B,), device=device)
        a_bar = self._alpha_cumprod[t_idx].view(B, 1)

        noise = torch.randn_like(z0)
        z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise

        # t normalized to [0,1] for the time embedding
        t = t_idx.float() / float(max(1, self.T - 1))
        noise_hat = self.forward(z_t, t)

        diff = F.mse_loss(noise_hat, noise)
        total = diff + self.recon_weight * recon

        return {"total": total, "diff": diff, "recon": recon}

    @torch.no_grad()
    def sample(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """
        Sample x_hat in feature space: [n, input_dim]
        """
        device = torch.device(device)
        self._ensure_schedule(device)
        self.eval()

        z = torch.randn(n, self.latent_dim, device=device)

        for t in reversed(range(self.T)):
            t_idx = torch.full((n,), t, device=device, dtype=torch.long)
            a = self._alphas[t_idx].view(n, 1)
            a_bar = self._alpha_cumprod[t_idx].view(n, 1)
            b = self._betas[t_idx].view(n, 1)

            t_norm = t_idx.float() / float(max(1, self.T - 1))
            eps_hat = self.forward(z, t_norm)

            # DDPM update in latent
            z = (1.0 / torch.sqrt(a)) * (z - (b / torch.sqrt(1.0 - a_bar)) * eps_hat)

            if t > 0:
                z = z + torch.sqrt(b) * torch.randn_like(z)

        x_hat = self.decode(z)
        return x_hat


# Backward-compatible alias (older code expects DCFDiffusion)
DCFDiffusion = TabularDCF
