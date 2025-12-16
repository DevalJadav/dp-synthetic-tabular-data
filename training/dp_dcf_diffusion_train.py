# training/dp_dcf_diffusion_train.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .config import DP_DCF_CONFIG, MODELS_DIR, TARGET_COLUMN
from .data_utils import (
    load_adult,
    preprocess_tabular_conditional,   # ✅ IMPORTANT
    fit_tabular_metadata,             # ✅ train-only metadata
    transform_with_metadata,          # ✅ transform using that metadata
    reconstruct_from_synthetic,
)
from .evaluation import evaluate_synth_vs_real

from models.dcf_diffusion import TabularDCF


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rand_seed() -> int:
    return int.from_bytes(os.urandom(4), "little")


def train_dp_dcf_diffusion(device: Optional[str] = None) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # training randomness
    seed = int(DP_DCF_CONFIG.get("seed", -1))
    if seed < 0:
        seed = _rand_seed()
    _set_seed(seed)

    # keep split stable (so you can compare runs); set -1 to randomize split too
    split_seed = int(DP_DCF_CONFIG.get("split_seed", 0))
    if split_seed < 0:
        split_seed = _rand_seed()

    # sampling randomness for eval
    sample_seed = int(DP_DCF_CONFIG.get("sample_seed", -1))
    if sample_seed < 0:
        sample_seed = _rand_seed()

    print(
        f"[INFO] DP DCF-Diffusion device: {device} | "
        f"seed={seed} | split_seed={split_seed} | sample_seed={sample_seed}"
    )

    # -------------------------
    # Load data
    # -------------------------
    df = load_adult(balanced=True, random_state=0).reset_index(drop=True)
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"TARGET_COLUMN '{TARGET_COLUMN}' missing from dataset.")

    y = df[TARGET_COLUMN].astype(str).values

    # -------------------------
    # Split ONCE (authoritative split)
    # -------------------------
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=split_seed,
        stratify=y if pd.Series(y).value_counts().min() >= 2 else None,
        shuffle=True,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train[TARGET_COLUMN].astype(str).values
    if pd.Series(y_train).nunique() < 2:
        raise RuntimeError(f"Conditional training needs >=2 labels, got: {sorted(pd.Series(y_train).unique())}")

    # -------------------------
    # Preprocess for CONDITIONAL DCF
    # -> excludes income from features
    # We fit metadata on TRAIN ONLY for stable feature space.
    # -------------------------
    # This call is mainly to keep metadata consistent with your pipeline,
    # but we will use the explicit fit/transform below as the source of truth.
    _Xtr_tmp, _Xte_tmp, _meta_tmp = preprocess_tabular_conditional(df, test_size=0.2, random_state=split_seed)

    # Fit metadata on train ONLY (no leakage)
    # IMPORTANT: include_target_as_feature=False for conditional
    metadata = fit_tabular_metadata(df_train, include_target_as_feature=False)

    X_train = transform_with_metadata(df_train, metadata)
    X_test = transform_with_metadata(df_test, metadata)

    data_dim = int(X_train.shape[1])
    X_train_t = torch.tensor(X_train, dtype=torch.float32)

    # -------------------------
    # Hyperparams
    # -------------------------
    latent_dim = int(DP_DCF_CONFIG.get("latent_dim", 64))
    time_emb_dim = int(DP_DCF_CONFIG.get("time_emb_dim", 64))
    width = int(DP_DCF_CONFIG.get("hidden", 256))
    depth = int(DP_DCF_CONFIG.get("depth", 3))
    dropout = float(DP_DCF_CONFIG.get("dropout", 0.1))
    use_spectral_norm = bool(DP_DCF_CONFIG.get("use_spectral_norm", True))
    T = int(DP_DCF_CONFIG.get("T", 100))
    recon_weight = float(DP_DCF_CONFIG.get("recon_weight", 1.0))

    lr = float(DP_DCF_CONFIG["lr"])
    epochs = int(DP_DCF_CONFIG["epochs"])
    batch_size = int(DP_DCF_CONFIG["batch_size"])
    max_grad_norm = float(DP_DCF_CONFIG["max_grad_norm"])
    noise_multiplier = float(DP_DCF_CONFIG["noise_multiplier"])

    # -------------------------
    # Conditional training: one model per label
    # -------------------------
    label_values = sorted(pd.Series(y_train).unique().tolist())

    # label distribution in train (for sampling proportions)
    train_dist = pd.Series(y_train).value_counts(normalize=True).to_dict()
    train_dist = {str(k): float(v) for k, v in train_dist.items()}

    state_dicts: Dict[str, Dict[str, Any]] = {}

    for lab in label_values:
        mask = (y_train == str(lab))
        X_lab_t = X_train_t[mask]

        print(f"[INFO] Training conditional DCF for label='{lab}' | n={int(X_lab_t.shape[0])}")

        if int(X_lab_t.shape[0]) < max(8, batch_size // 4):
            raise RuntimeError(
                f"Too few samples for label '{lab}' after balancing/split. "
                f"Have {int(X_lab_t.shape[0])}. "
                f"Reduce batch_size or disable balanced=True."
            )

        # allow smaller label batches by using an effective batch size
        eff_bs = min(batch_size, int(X_lab_t.shape[0]))
        loader = DataLoader(
            TensorDataset(X_lab_t),
            batch_size=eff_bs,
            shuffle=True,
            drop_last=True if eff_bs >= 8 else False,
        )

        model = TabularDCF(
            input_dim=data_dim,
            latent_dim=latent_dim,
            time_emb_dim=time_emb_dim,
            width=width,
            depth=depth,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            T=T,
            recon_weight=recon_weight,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            last = None

            for (xb,) in loader:
                xb = xb.to(device)

                loss_dict = model.loss(xb)
                loss = loss_dict["total"]

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # DP-ish: clip + noise (approx; not true per-sample DP)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.add_(torch.randn_like(p.grad) * (noise_multiplier * max_grad_norm))

                opt.step()
                last = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}

            if last is not None and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0 or epoch == epochs - 1):
                print(
                    f"[{lab}] [Epoch {epoch+1:03d}/{epochs}] "
                    f"total={last['total']:.4f} diff={last['diff']:.4f} recon={last['recon']:.4f}"
                )

        state_dicts[str(lab)] = model.state_dict()

    # -------------------------
    # Save checkpoint (conditional)
    # -------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dp_dcf_diffusion_{ts}"
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{run_name}.pt")

    torch.save(
    {
        "conditional": True,
        "labels": label_values,
        "train_dist": train_dist,
        "model_state_dicts": state_dicts,
        "metadata": {
            **metadata,
            "data_dim": data_dim,
            "latent_dim": latent_dim,
            "time_emb_dim": time_emb_dim,
            "width": width,
            "depth": depth,
            "dropout": dropout,
            "use_spectral_norm": use_spectral_norm,
            "T": T,
            "recon_weight": recon_weight,
            "split": {"test_size": 0.2, "random_state": split_seed},
            "include_target_as_feature": False,
            "decode_seed": sample_seed,   # ✅ ADD HERE
        },
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
    },
    model_path,
    )

    print(f"[INFO] Saved DP DCF-Diffusion (conditional): {model_path}")

    # -------------------------
    # Evaluation: generate synth WITH target column
    # -------------------------
    n_synth = int(DP_DCF_CONFIG.get("n_synth_eval", 10000))

    # deterministic sampling if you keep sample_seed fixed
    _set_seed(sample_seed)

    # allocate per-label counts sum exactly to n_synth
    counts: list[tuple[str, int]] = []
    remaining = n_synth
    for i, lab in enumerate(label_values):
        if i == len(label_values) - 1:
            n_lab = remaining
        else:
            p = float(train_dist.get(str(lab), 1.0 / len(label_values)))
            n_lab = max(1, int(round(n_synth * p)))
            min_left = (len(label_values) - i - 1) * 1
            n_lab = min(n_lab, remaining - min_left)
        counts.append((str(lab), n_lab))
        remaining -= n_lab

    synth_parts = []
    for lab, n_lab in counts:
        model = TabularDCF(
            input_dim=data_dim,
            latent_dim=latent_dim,
            time_emb_dim=time_emb_dim,
            width=width,
            depth=depth,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            T=T,
            recon_weight=recon_weight,
        ).to(device)

        model.load_state_dict(state_dicts[str(lab)])
        model.eval()

        with torch.no_grad():
            x_hat = model.sample(n=n_lab, device=device).detach().cpu().numpy()

        x_hat = np.clip(x_hat, -3.0, 3.0)

        # decode categoricals properly (sample mode in your new data_utils)
        df_synth_lab = reconstruct_from_synthetic(
            x_hat,
            metadata,
            categorical_sampling={"workclass": 0.9, "occupation": 0.9},  # optional temps
            categorical_mode="sample",
        )

        # add label back
        df_synth_lab[TARGET_COLUMN] = str(lab)
        synth_parts.append(df_synth_lab)

    df_synth = (
        pd.concat(synth_parts, ignore_index=True)
        .sample(frac=1.0, random_state=sample_seed)
        .reset_index(drop=True)
    )

    # evaluate vs real FULL df (your evaluation function decides how)
    metrics = evaluate_synth_vs_real(df_real=df, df_synth=df_synth, target_column=TARGET_COLUMN)

    entry = {
        "run_name": run_name,
        "type": "dp_dcf_diffusion",
        "model_path": str(model_path),
        "metrics": metrics,
        "timestamp": ts,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "epsilon": None,
        "seed": seed,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "conditional": True,
        "labels": label_values,
    }

    # append to registry.json (list format)
    registry_file = MODELS_DIR / "registry.json"
    try:
        existing = json.loads(registry_file.read_text()) if registry_file.exists() and registry_file.read_text().strip() else []
    except Exception:
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append(entry)
    registry_file.write_text(json.dumps(existing, indent=2))

    return entry
