# training/dp_ctgan_train.py
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import MODELS_DIR, DP_CTGAN_CONFIG, TARGET_COLUMN
from .data_utils import load_adult, preprocess_tabular, reconstruct_from_synthetic, enforce_binary_label_balance
from .evaluation import evaluate_synthetic_utility
from models.dp_gan import make_dp_gan


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _rand_seed() -> int:
    return int.from_bytes(os.urandom(4), "little")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dp_ctgan(device: Optional[str] = None) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # Seeds:
    #  - seed: training randomness (can vary per run)
    #  - split_seed: MUST match preprocess_tabular split
    #  - eval_seed: controls evaluation split + synth sampling
    # -------------------------------------------------
    seed = int(DP_CTGAN_CONFIG.get("seed", -1))
    if seed < 0:
        seed = _rand_seed()

    split_seed = int(DP_CTGAN_CONFIG.get("split_seed", 0))  # preprocess_tabular uses random_state=0 by default

    eval_seed = int(DP_CTGAN_CONFIG.get("eval_seed", -1))
    if eval_seed < 0:
        eval_seed = _rand_seed()

    _set_seed(seed)
    print(f"[INFO] DP-CTGAN device={device} | seed={seed} | split_seed={split_seed} | eval_seed={eval_seed}")

    # -------------------------
    # Load + preprocess
    # -------------------------
    df = load_adult(balanced=True).reset_index(drop=True)

    # IMPORTANT: keep split deterministic + aligned across your project
    X_train, X_test, metadata = preprocess_tabular(df, test_size=0.2, random_state=split_seed)

    data_dim = int(metadata["data_dim"])
    noise_dim = int(DP_CTGAN_CONFIG.get("noise_dim", 64))

    # -------------------------
    # Model
    # -------------------------
    G, D = make_dp_gan(
        data_dim=data_dim,
        noise_dim=noise_dim,
        gen_hidden=DP_CTGAN_CONFIG["generator_dim"],
        disc_hidden=DP_CTGAN_CONFIG["discriminator_dim"],
    )
    G.to(device)
    D.to(device)

    lr = float(DP_CTGAN_CONFIG["lr"])
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    batch_size = int(DP_CTGAN_CONFIG["batch_size"])
    epochs = int(DP_CTGAN_CONFIG["epochs"])

    max_grad_norm = float(DP_CTGAN_CONFIG.get("max_grad_norm", 1.0))
    noise_multiplier = float(DP_CTGAN_CONFIG.get("noise_multiplier", 0.0))

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    # -------------------------
    # Training loop (simple GAN)
    # DP-ish: clip + add noise to gradients
    # -------------------------
    G.train()
    D.train()

    for ep in range(1, epochs + 1):
        perm = torch.randperm(X_train_t.size(0), device=device)
        X_shuf = X_train_t[perm]

        d_losses = []
        g_losses = []

        for i in range(0, X_shuf.size(0), batch_size):
            real = X_shuf[i : i + batch_size]
            if real.size(0) < 2:
                continue

            # ---- train D ----
            z = torch.randn(real.size(0), noise_dim, device=device)
            fake = G(z).detach()

            d_real = D(real).view(-1)
            d_fake = D(fake).view(-1)

            lossD = (
                F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
                F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            )

            optD.zero_grad(set_to_none=True)
            lossD.backward()

            # clip + DP-ish noise
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_grad_norm)
            if noise_multiplier > 0:
                for p in D.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * (noise_multiplier * max_grad_norm))

            optD.step()

            # ---- train G ----
            z2 = torch.randn(real.size(0), noise_dim, device=device)
            fake2 = G(z2)
            d_fake2 = D(fake2).view(-1)

            lossG = F.binary_cross_entropy_with_logits(d_fake2, torch.ones_like(d_fake2))

            optG.zero_grad(set_to_none=True)
            lossG.backward()

            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
            if noise_multiplier > 0:
                for p in G.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * (noise_multiplier * max_grad_norm))

            optG.step()

            d_losses.append(float(lossD.detach().item()))
            g_losses.append(float(lossG.detach().item()))

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[DP-CTGAN][Epoch {ep:03d}/{epochs}] D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f}")

    # -------------------------
    # Save checkpoint
    # -------------------------
    run_name = f"dp_ctgan_{_timestamp()}"
    out_path = MODELS_DIR / f"{run_name}.pt"

    ckpt = {
        "type": "dp_ctgan",
        "noise_dim": noise_dim,
        "metadata": metadata,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "seed": seed,
        "split_seed": split_seed,
        "eval_seed": eval_seed,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
    }
    torch.save(ckpt, out_path)

    # -------------------------
    # Evaluate (utility)
    # -------------------------
    # NOTE: keep the *real* dataset consistent; evaluation randomness comes from eval_seed
    df_real = load_adult(balanced=True).reset_index(drop=True)

    # Sampling for synth eval should be controlled to compare runs fairly
    _set_seed(eval_seed)

    n_eval = int(DP_CTGAN_CONFIG.get("n_synth_eval", 10000))
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_eval, noise_dim, device=device)
        Xs = G(z).detach().cpu().numpy()

    df_synth = reconstruct_from_synthetic(Xs, metadata)

    # prevent crash if income collapses
    df_synth = enforce_binary_label_balance(df_synth)

    metrics = evaluate_synthetic_utility(
        df_real=df_real,
        df_synth=df_synth,
        target_column=TARGET_COLUMN,
        n_synth=n_eval,
        eval_seed=eval_seed,
    )

    entry = {
        "run_name": run_name,
        "type": "dp_ctgan",
        "model_path": str(out_path),
        "metrics": metrics,
        "timestamp": _timestamp(),
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "epsilon": None,
        "seed": seed,
        "split_seed": split_seed,
        "eval_seed": eval_seed,
    }
    return entry
