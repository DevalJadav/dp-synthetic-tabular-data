# training/data_utils.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    ADULT_CSV_PATH,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)

# ----------------------------
# Load Adult dataset
# ----------------------------
def load_adult(balanced: bool = True, random_state: int = 0) -> pd.DataFrame:
    if not ADULT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Adult CSV not found at {ADULT_CSV_PATH}\n"
            f"Put it here: {ADULT_CSV_PATH}"
        )

    df = pd.read_csv(ADULT_CSV_PATH)

    # many Adult csvs have spaces in column names
    df.columns = df.columns.astype(str).str.strip()

    # strip categorical values
    for c in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # OPTIONAL: balance income classes by downsampling majority
    if balanced and TARGET_COLUMN in df.columns:
        vc = df[TARGET_COLUMN].value_counts(dropna=False)
        if len(vc) >= 2:
            n_min = int(vc.min())
            df = (
                df.groupby(TARGET_COLUMN, group_keys=False)
                  .sample(n=n_min, random_state=random_state)
                  .reset_index(drop=True)
            )

    return df


# ----------------------------
# Core: preprocess -> standardized numeric + one-hot categoricals
# Supports optional inclusion of TARGET_COLUMN as feature
# ----------------------------
def preprocess_tabular(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0,
    include_target_as_feature: bool = True,   # ✅ key switch
):
    """
    Returns:
      X_train, X_test, metadata

    metadata includes:
      - num_cols, cat_cols, cat_levels, feature_slices
      - scaler_mean, scaler_scale
      - y_train/y_test binary for utility eval (based on df[TARGET_COLUMN])
      - pos_label / neg_label
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")

    # labels for stratification + evaluation (NOT necessarily in features)
    y = df[TARGET_COLUMN].astype(str).str.strip()

    # which cols we will one-hot
    gen_cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if include_target_as_feature and (TARGET_COLUMN not in gen_cat_cols):
        gen_cat_cols = gen_cat_cols + [TARGET_COLUMN]

    gen_num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    used_cols = gen_num_cols + gen_cat_cols
    Xdf = df[used_cols].copy()

    # clean categoricals
    for c in gen_cat_cols:
        Xdf[c] = Xdf[c].astype(str).str.strip().replace({"nan": "?"}).fillna("?")

    # stratify only if safe
    strat = None
    counts = y.value_counts()
    if len(counts) >= 2 and counts.min() >= 2:
        strat = y

    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        Xdf, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
        shuffle=True,
    )

    # numeric scaling
    scaler = StandardScaler()
    if gen_num_cols:
        X_num_train = scaler.fit_transform(X_train_df[gen_num_cols].astype(float))
        X_num_test = scaler.transform(X_test_df[gen_num_cols].astype(float))
    else:
        X_num_train = np.zeros((len(X_train_df), 0), dtype=np.float32)
        X_num_test = np.zeros((len(X_test_df), 0), dtype=np.float32)

    # one-hot categoricals
    cat_levels: Dict[str, list] = {}
    X_cat_train_parts = []
    X_cat_test_parts = []

    for c in gen_cat_cols:
        levels = sorted(X_train_df[c].astype(str).unique().tolist())
        cat_levels[c] = levels

        tr = pd.get_dummies(X_train_df[c], prefix=c)
        te = pd.get_dummies(X_test_df[c], prefix=c)

        want_cols = [f"{c}_{lvl}" for lvl in levels]
        tr = tr.reindex(columns=want_cols, fill_value=0)
        te = te.reindex(columns=want_cols, fill_value=0)

        X_cat_train_parts.append(tr.to_numpy(dtype=np.float32))
        X_cat_test_parts.append(te.to_numpy(dtype=np.float32))

    X_cat_train = np.concatenate(X_cat_train_parts, axis=1) if X_cat_train_parts else np.zeros((len(X_train_df), 0), dtype=np.float32)
    X_cat_test = np.concatenate(X_cat_test_parts, axis=1) if X_cat_test_parts else np.zeros((len(X_test_df), 0), dtype=np.float32)

    X_train = np.concatenate([X_num_train, X_cat_train], axis=1).astype(np.float32)
    X_test = np.concatenate([X_num_test, X_cat_test], axis=1).astype(np.float32)

    # binary label encoding for evaluation (independent of include_target_as_feature)
    uniq = sorted(y.unique().tolist())
    neg_label, pos_label = (uniq[0], uniq[1]) if len(uniq) == 2 else ("<=50K", ">50K")
    y_train = (y_train_s == pos_label).astype(np.int64).to_numpy()
    y_test = (y_test_s == pos_label).astype(np.int64).to_numpy()

    # slices
    feature_slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for c in gen_num_cols:
        feature_slices[c] = (cursor, cursor + 1)
        cursor += 1
    for c in gen_cat_cols:
        k = len(cat_levels[c])
        feature_slices[c] = (cursor, cursor + k)
        cursor += k

    metadata = {
        "num_cols": gen_num_cols,
        "cat_cols": gen_cat_cols,
        "cat_levels": cat_levels,
        "feature_slices": feature_slices,
        "scaler_mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
        "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
        "data_dim": int(X_train.shape[1]),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
        "pos_label": pos_label,
        "neg_label": neg_label,
        "include_target_as_feature": bool(include_target_as_feature),
        "split": {"test_size": float(test_size), "random_state": int(random_state)},
    }

    return X_train, X_test, metadata


# ----------------------------
# BEST PRACTICE FOR YOUR CONDITIONAL DCF
# (one model per income label)
# -> EXCLUDE income from features, add it back after sampling.
# ----------------------------
def preprocess_tabular_conditional(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0,
):
    """
    Same as preprocess_tabular, but forces include_target_as_feature=False.
    Use this for conditional DCF training (one model per label).
    """
    return preprocess_tabular(
        df=df,
        test_size=test_size,
        random_state=random_state,
        include_target_as_feature=False,
    )


# ----------------------------
# Reconstruction helper
# ----------------------------
def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def reconstruct_from_synthetic(
    x: np.ndarray,
    metadata: Dict[str, Any],
    categorical_sampling: Optional[Dict[str, float]] = None,
    categorical_mode: str = "sample",  # ✅ "sample" or "argmax"
) -> pd.DataFrame:
    """
    Reconstruct dataframe from model output vector.

    IMPORTANT FIX:
    - For categorical blocks, apply SOFTMAX first.
    - Then either sample (recommended) or argmax.
    """
    categorical_sampling = categorical_sampling or {}
    num_cols = metadata["num_cols"]
    cat_cols = metadata["cat_cols"]
    cat_levels = metadata["cat_levels"]
    feature_slices = metadata["feature_slices"]

    df_out: Dict[str, Any] = {}

    # unscale numeric
    mean = np.array(metadata.get("scaler_mean", []), dtype=float) if metadata.get("scaler_mean") else None
    scale = np.array(metadata.get("scaler_scale", []), dtype=float) if metadata.get("scaler_scale") else None

    for i, c in enumerate(num_cols):
        s, e = feature_slices[c]
        col = x[:, s:e].reshape(-1)
        if mean is not None and scale is not None and i < len(mean) and i < len(scale) and scale[i] != 0:
            col = col * scale[i] + mean[i]
        df_out[c] = col

    # categorical decode
    decode_seed = metadata.get("decode_seed", 0)
    rng = np.random.default_rng(int(decode_seed))
    for c in cat_cols:
        s, e = feature_slices[c]
        logits = x[:, s:e]
        levels = cat_levels[c]
        k = len(levels)
        if k == 0:
            df_out[c] = ["?"] * len(x)
            continue

        temp = float(categorical_sampling.get(c, 1.0))
        temp = max(1e-6, temp)

        # softmax probabilities
        probs = _softmax(logits / temp, axis=1)

        if categorical_mode == "argmax":
            idx = np.argmax(probs, axis=1)
        else:
            # sample from probs
            idx = np.array([rng.choice(k, p=probs[i]) for i in range(probs.shape[0])], dtype=int)

        df_out[c] = [levels[int(i)] for i in idx]

    return pd.DataFrame(df_out)


# ----------------------------
# Utility helpers
# ----------------------------
def append_income_from_probs(
    df_synth: pd.DataFrame,
    probs: np.ndarray,
    pos_label: str = ">50K",
    neg_label: str = "<=50K",
) -> pd.DataFrame:
    p = probs.reshape(-1)
    out = df_synth.copy()
    out[TARGET_COLUMN] = np.where(p >= 0.5, pos_label, neg_label)
    return out


def enforce_binary_label_balance(df_synth: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df_synth.columns:
        return df_synth
    vc = df_synth[TARGET_COLUMN].value_counts()
    if len(vc) < 2:
        out = df_synth.copy()
        idx = out.sample(frac=0.2, random_state=0).index
        only = vc.index[0]
        other = ">50K" if only != ">50K" else "<=50K"
        out.loc[idx, TARGET_COLUMN] = other
        return out
    return df_synth


# ============================
# fit/transform helpers
# (kept, but aligned)
# ============================
def fit_tabular_metadata(df_train: pd.DataFrame, include_target_as_feature: bool = True) -> Dict[str, Any]:
    """
    Fit scaler + categorical levels on df_train ONLY.
    Returns metadata compatible with reconstruct_from_synthetic().
    """
    df_train = df_train.copy()
    df_train.columns = df_train.columns.astype(str).str.strip()

    # decide which columns to include
    num_cols = [c for c in NUMERIC_COLUMNS if c in df_train.columns]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df_train.columns]
    if include_target_as_feature and TARGET_COLUMN in df_train.columns:
        if TARGET_COLUMN not in cat_cols:
            cat_cols = cat_cols + [TARGET_COLUMN]

    used_cols = [c for c in (num_cols + cat_cols) if c in df_train.columns]
    df_train = df_train[used_cols]

    # Fill missing categoricals
    for c in cat_cols:
        df_train[c] = df_train[c].astype(str).str.strip().replace({"nan": "?"}).fillna("?")

    scaler = StandardScaler()
    if num_cols:
        scaler.fit(df_train[num_cols].astype(float))
    else:
        scaler.mean_ = np.array([])
        scaler.scale_ = np.array([])

    cat_levels: Dict[str, list] = {}
    feature_slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    for c in num_cols:
        feature_slices[c] = (cursor, cursor + 1)
        cursor += 1

    for c in cat_cols:
        levels = sorted(df_train[c].astype(str).unique().tolist())
        cat_levels[c] = levels
        k = len(levels)
        feature_slices[c] = (cursor, cursor + k)
        cursor += k

    metadata: Dict[str, Any] = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_levels": cat_levels,
        "scaler_mean": scaler.mean_.tolist() if num_cols else [],
        "scaler_scale": scaler.scale_.tolist() if num_cols else [],
        "feature_slices": feature_slices,
        "data_dim": int(cursor),
        "include_target_as_feature": bool(include_target_as_feature),
    }
    return metadata


def transform_with_metadata(df: pd.DataFrame, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Transform df into the same standardized + one-hot format using pre-fitted metadata.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    num_cols = metadata.get("num_cols", [])
    cat_cols = metadata.get("cat_cols", [])
    cat_levels = metadata.get("cat_levels", {})

    used_cols = [c for c in (num_cols + cat_cols) if c in df.columns]
    df = df[used_cols]

    # Fill missing categoricals
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": "?"}).fillna("?")

    # numeric
    if num_cols:
        mean = np.array(metadata.get("scaler_mean", []), dtype=float)
        scale = np.array(metadata.get("scaler_scale", []), dtype=float)
        X_num = df[num_cols].astype(float).to_numpy()
        X_num = (X_num - mean) / (scale + 1e-12)
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)

    # categoricals
    X_cat_parts = []
    for c in cat_cols:
        levels = cat_levels.get(c, [])
        k = len(levels)
        if c not in df.columns or k == 0:
            X_cat_parts.append(np.zeros((len(df), k), dtype=np.float32))
            continue

        want_cols = [f"{c}_{lvl}" for lvl in levels]
        onehot = pd.get_dummies(df[c], prefix=c)
        onehot = onehot.reindex(columns=want_cols, fill_value=0)
        X_cat_parts.append(onehot.to_numpy(dtype=np.float32))

    X_cat = np.concatenate(X_cat_parts, axis=1) if X_cat_parts else np.zeros((len(df), 0), dtype=np.float32)
    X = np.concatenate([X_num.astype(np.float32), X_cat.astype(np.float32)], axis=1).astype(np.float32)
    return X
