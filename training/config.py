# training/config.py
from __future__ import annotations
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Put your file here:
#   E:\Deep Learning Application\data\raw\adult.csv
ADULT_CSV_PATH = RAW_DATA_DIR / "adult.csv"

# Backward-compatible alias (your data_utils imports this)
RAW_CSV_PATH = ADULT_CSV_PATH

# ----------------------------
# Dataset schema
# ----------------------------
TARGET_COLUMN = "income"

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

SENSITIVE_COLUMNS = ["race", "sex", "native_country"]

# ----------------------------
# Training configs
# ----------------------------
CTGAN_CONFIG = {
    "epochs": 50,
    "batch_size": 512,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256),
    "lr": 2e-4,
}

DP_CTGAN_CONFIG = {
    "batch_size": 512,
    "epochs": 50,
    "lr": 1e-4,
    "generator_dim": [256, 256],
    "discriminator_dim": [256, 256],
    "max_grad_norm": 1.0,
    "noise_multiplier": 0.8,
    "n_synth_eval": 10000,
}

# ----------------------------
# DP DCF-Diffusion config
# ----------------------------
DP_DCF_CONFIG = {
    "batch_size": 512,
    "epochs": 150,
    "lr": 2e-4,
    "hidden": 512,          # used by your diffusion model
    "depth": 3,
    "dropout": 0.1,
    "latent_dim": 64,
    "time_emb_dim": 64,
    "T": 100,
    "recon_weight": 1.0,

    "max_grad_norm": 1.0,
    "noise_multiplier": 0.6,
    "n_synth_eval": 10000,

    "seed": -1,        # random training every run
    "split_seed": 0,   # MUST match preprocess_tabular split
    "sample_seed": -1, # random sampling/eval every run
}

