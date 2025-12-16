# training/ctgan_train.py

import json
import pickle
from datetime import datetime

from ctgan import CTGAN

from .config import (
    CTGAN_CONFIG,
    MODELS_DIR,
    CATEGORICAL_COLUMNS,
)
from .data_utils import load_adult
from .evaluation import evaluate_synthetic_utility


def train_ctgan(run_name: str = None):
    # 1. Load dataset
    df = load_adult()

    # 2. Discrete columns = all categoricals (including income)
    discrete_columns = CATEGORICAL_COLUMNS

    # 3. Create CTGAN model (IMPORTANT: pac=1 so batch_size can be anything)
    ctgan = CTGAN(
        batch_size=CTGAN_CONFIG["batch_size"],
        generator_dim=CTGAN_CONFIG["generator_dim"],
        discriminator_dim=CTGAN_CONFIG["discriminator_dim"],
        generator_lr=CTGAN_CONFIG["lr"],
        discriminator_lr=CTGAN_CONFIG["lr"],
        epochs=CTGAN_CONFIG["epochs"],
        pac=1,          # <-- FIX: avoid batch_size % pac assertion
        verbose=True,
    )

    print("[INFO] Fitting CTGAN on Adult dataset...")
    # You can either pass epochs here or rely on the one in the constructor.
    ctgan.fit(df, discrete_columns)

    # 4. Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"ctgan_{timestamp}"

    model_path = MODELS_DIR / f"{run_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(ctgan, f)

    print(f"[INFO] CTGAN saved: {model_path}")

    # 5. Evaluate utility (synthetic → real)
    metrics = evaluate_synthetic_utility(
        df_real=df,
        synthesizer=ctgan,
        target_column="income",
        n_synth=10000,
    )

    registry_entry = {
        "run_name": run_name,
        "type": "ctgan",
        "model_path": str(model_path),
        "metrics": metrics,
        "timestamp": timestamp,
    }

    # 6. Append to registry.json
    registry_file = MODELS_DIR / "registry.json"
    try:
        if registry_file.exists() and registry_file.read_text().strip():
            existing = json.loads(reg_file.read_text())
        else:
            existing = []
    except Exception:
        existing = []

    existing.append(registry_entry)
    registry_file.write_text(json.dumps(existing, indent=2))
    print(f"[INFO] Registry updated: {registry_file}")
    print("[METRICS]", metrics)
    return registry_entry


if __name__ == "__main__":
    train_ctgan()
