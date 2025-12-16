# backend/api.py
import json
import pickle
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from training.config import MODELS_DIR, DP_CTGAN_CONFIG, TARGET_COLUMN
from training.data_utils import load_adult, reconstruct_from_synthetic
from agents.agent_pipeline import orchestrate_training


# -----------------------------
# Auto-delete __pycache__ (dev)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
for p in ROOT.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
sys.dont_write_bytecode = True


# -----------------------------
# Paths for results
# -----------------------------
RESULTS_DIR = ROOT / "results"
SYNTH_DIR = RESULTS_DIR / "synth"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="DP Synthetic Tabular Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Small helpers
# -----------------------------
def _pick_first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _load_torch(path: Path) -> Dict[str, Any]:
    # weights_only exists in newer torch; on older versions it will error
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fix_exact_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Ensure df has exactly n rows (sample or pad)."""
    if len(df) > n:
        return df.sample(n, random_state=0).reset_index(drop=True)
    if len(df) < n:
        extra = df.sample(n - len(df), replace=True, random_state=0)
        return (
            pd.concat([df, extra], ignore_index=True)
            .sample(frac=1.0, random_state=0)
            .reset_index(drop=True)
        )
    return df.reset_index(drop=True)


# -----------------------------
# Registry helpers (robust)
# -----------------------------
def _load_registry_json() -> Any:
    reg_path = MODELS_DIR / "registry.json"
    if not reg_path.exists() or not reg_path.read_text().strip():
        return {}
    return json.loads(reg_path.read_text())


def _registry_entries(reg_json: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      1) list of entries
      2) dict with keys {dp_ctgan: {...}, dp_dcf_diffusion: {...}, best_model: {...}}
    """
    if isinstance(reg_json, list):
        return [e for e in reg_json if isinstance(e, dict) and e.get("run_name")]
    if isinstance(reg_json, dict):
        entries = []
        for v in reg_json.values():
            if isinstance(v, dict) and v.get("run_name") and v.get("model_path"):
                entries.append(v)
        return entries
    return []


def _find_entry(run_name: str) -> Dict[str, Any]:
    reg_json = _load_registry_json()
    entries = _registry_entries(reg_json)
    for e in entries:
        if e.get("run_name") == run_name:
            return e
    raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")


@app.get("/models")
def list_models():
    reg = _load_registry_json()
    if isinstance(reg, dict):
        out = []
        for v in reg.values():
            if isinstance(v, dict) and v.get("run_name") and v.get("model_path"):
                out.append(v)
        return out
    return reg if isinstance(reg, list) else []


# -----------------------------
# Post-processing helpers
# -----------------------------
def _quantile_map_to_real(synth_col: pd.Series, real_col: pd.Series) -> pd.Series:
    s = pd.to_numeric(synth_col, errors="coerce")
    r = pd.to_numeric(real_col, errors="coerce").dropna().values
    if len(r) < 10:
        return synth_col
    ranks = s.rank(method="average", pct=True)
    ranks = ranks.fillna(0.5).clip(0.0, 1.0).values
    mapped = np.quantile(r, ranks)
    return pd.Series(mapped, index=synth_col.index)


def _rebalance_target_like_real(df_synth: pd.DataFrame, df_real: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    if target not in df_synth.columns or target not in df_real.columns:
        return df_synth

    real_dist = df_real[target].astype(str).value_counts(normalize=True)
    df_synth = df_synth.copy()
    df_synth[target] = df_synth[target].astype(str)

    parts = []
    for cls, p in real_dist.items():
        want = int(round(n * float(p)))
        sub = df_synth[df_synth[target] == str(cls)]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(want, replace=True, random_state=0))

    if not parts:
        return df_synth

    out = pd.concat(parts, ignore_index=True)

    if len(out) > n:
        out = out.sample(n, random_state=0)
    elif len(out) < n:
        out = pd.concat(
            [out, df_synth.sample(n - len(out), replace=True, random_state=0)],
            ignore_index=True,
        )

    return out.sample(frac=1.0, random_state=0).reset_index(drop=True)


def _smooth_numeric_like_real(
    df_synth: pd.DataFrame,
    df_real: pd.DataFrame,
    numeric_cols: List[str],
    noise_scale: float = 0.02,
) -> pd.DataFrame:
    """Reduce peaky numeric distributions by light post-noise scaled to real variance."""
    out = df_synth.copy()

    for col in numeric_cols:
        if col not in out.columns or col not in df_real.columns:
            continue

        std = pd.to_numeric(df_real[col], errors="coerce").std()
        if not np.isfinite(std) or std == 0:
            continue

        noise = np.random.normal(0, std * noise_scale, size=len(out))
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float) + noise

        lo = pd.to_numeric(df_real[col], errors="coerce").min()
        hi = pd.to_numeric(df_real[col], errors="coerce").max()
        if np.isfinite(lo) and np.isfinite(hi):
            out[col] = out[col].clip(lo, hi)

    return out


def _classwise_quantile_map(df_synth: pd.DataFrame, df_real: pd.DataFrame, col: str) -> pd.Series:
    """Quantile-map numeric column to real distribution, separately per income class."""
    if TARGET_COLUMN not in df_synth.columns or TARGET_COLUMN not in df_real.columns:
        return _quantile_map_to_real(df_synth[col], df_real[col])

    out = df_synth[col].copy()
    real_classes = df_real[TARGET_COLUMN].astype(str).unique()

    for cls in real_classes:
        rs = df_real[df_real[TARGET_COLUMN].astype(str) == str(cls)]
        ss = df_synth[df_synth[TARGET_COLUMN].astype(str) == str(cls)]
        if len(rs) < 50 or len(ss) < 50:
            continue
        out.loc[ss.index] = _quantile_map_to_real(ss[col], rs[col]).values

    return out


def _select_best_subset(df_synth: pd.DataFrame, df_real: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df_synth.copy()

    if TARGET_COLUMN in out.columns and TARGET_COLUMN in df_real.columns:
        out = _rebalance_target_like_real(out, df_real, TARGET_COLUMN, n=min(len(out), max(n, 1000)))

    out = _smooth_numeric_like_real(
        out,
        df_real,
        numeric_cols=["age", "hours_per_week", "capital_gain", "capital_loss"],
        noise_scale=0.02,
    )

    return _fix_exact_n(out, n)


# -----------------------------
# Integrity repair (IMPORTANT)
# -----------------------------
_EDU_TO_NUM = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def _integrity_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce Adult-dataset logical constraints to stop impossible combinations
    that destroy classifier metrics.
    """
    out = df.copy()

    # relationship <-> sex consistency
    if "relationship" in out.columns and "sex" in out.columns:
        rel = out["relationship"].astype(str)
        out.loc[rel.eq("Husband"), "sex"] = "Male"
        out.loc[rel.eq("Wife"), "sex"] = "Female"

    # education -> education_num consistency
    if "education" in out.columns and "education_num" in out.columns:
        mapped = out["education"].astype(str).map(_EDU_TO_NUM)
        m = mapped.notna()
        out.loc[m, "education_num"] = mapped[m].astype(int)

    # never-worked sanity
    if "workclass" in out.columns:
        wc = out["workclass"].astype(str)
        if TARGET_COLUMN in out.columns:
            out.loc[wc.eq("Never-worked"), TARGET_COLUMN] = "<=50K"
        if "hours_per_week" in out.columns:
            out.loc[wc.eq("Never-worked"), "hours_per_week"] = out.loc[wc.eq("Never-worked"), "hours_per_week"].clip(0, 40)

    return out


# -----------------------------
# Utility: clamp + repair (final types)
# -----------------------------
def _repair_df(
    df_synth: pd.DataFrame,
    df_real: pd.DataFrame,
    n: Optional[int] = None,
    rebalance_target: bool = True,
) -> pd.DataFrame:
    df_synth = df_synth.copy()
    if n is None:
        n = len(df_synth)

    # 1) classwise calibration (numeric)
    for col in ["age", "hours_per_week"]:
        if col in df_synth.columns and col in df_real.columns:
            df_synth[col] = _classwise_quantile_map(df_synth, df_real, col)

    # 2) optional rebalance (label)
    if rebalance_target and TARGET_COLUMN in df_synth.columns and TARGET_COLUMN in df_real.columns:
        df_synth = _rebalance_target_like_real(df_synth, df_real, TARGET_COLUMN, int(n))

    # 3) smoothing
    df_synth = _smooth_numeric_like_real(
        df_synth,
        df_real,
        numeric_cols=["age", "hours_per_week", "capital_gain", "capital_loss"],
        noise_scale=0.02,
    )

    # 4) enforce integer columns
    int_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss", "fnlwgt"]
    for c in int_cols:
        if c in df_synth.columns:
            df_synth[c] = pd.to_numeric(df_synth[c], errors="coerce").fillna(0)
            df_synth[c] = np.rint(df_synth[c]).astype("int64")

    # bounds
    if "age" in df_synth.columns:
        df_synth["age"] = df_synth["age"].clip(17, 90)
    if "hours_per_week" in df_synth.columns:
        df_synth["hours_per_week"] = df_synth["hours_per_week"].clip(1, 99)
    if "fnlwgt" in df_synth.columns:
        df_synth["fnlwgt"] = df_synth["fnlwgt"].clip(lower=1)
    if "capital_gain" in df_synth.columns:
        df_synth["capital_gain"] = df_synth["capital_gain"].clip(lower=0)
    if "capital_loss" in df_synth.columns:
        df_synth["capital_loss"] = df_synth["capital_loss"].clip(lower=0)

    # 5) integrity rules (the big fix)
    df_synth = _integrity_repair(df_synth)

    # 6) exact n rows
    return _fix_exact_n(df_synth, int(n))


# -----------------------------
# Train orchestrator
# -----------------------------
@app.post("/orchestrate/train_best")
def orchestrate_train_best(device: Optional[str] = None):
    return orchestrate_training(device=device)


# -----------------------------
# Internal generator that returns FULL DF
# -----------------------------
def _allocate_label_counts(labels: List[str], train_dist: Dict[str, float], n: int) -> List[tuple[str, int]]:
    """
    Allocate exact counts per label that sum to n.
    """
    labels = [str(x) for x in labels]
    if not labels:
        return []

    # normalize dist
    dist = {str(k): float(v) for k, v in (train_dist or {}).items()}
    if not dist:
        dist = {lab: 1.0 / len(labels) for lab in labels}

    total = sum(dist.get(lab, 0.0) for lab in labels)
    if total <= 0:
        dist = {lab: 1.0 / len(labels) for lab in labels}
    else:
        dist = {lab: dist.get(lab, 0.0) / total for lab in labels}

    counts: List[tuple[str, int]] = []
    remaining = int(n)

    for i, lab in enumerate(labels):
        if i == len(labels) - 1:
            c = max(0, remaining)
        else:
            p = float(dist.get(lab, 1.0 / len(labels)))
            c = int(round(n * p))
            # ensure we leave at least 0 for others
            c = max(0, min(c, remaining))
        counts.append((lab, c))
        remaining -= c

    # if rounding made total < n, pad the largest class
    s = sum(c for _, c in counts)
    if s < n:
        lab0, c0 = counts[0]
        counts[0] = (lab0, c0 + (n - s))

    # if total > n (rare), trim from first
    s = sum(c for _, c in counts)
    if s > n:
        lab0, c0 = counts[0]
        counts[0] = (lab0, max(0, c0 - (s - n)))

    return counts


def _generate_df(run_name: str, n: int, device: str) -> pd.DataFrame:
    entry = _find_entry(run_name)
    model_path = Path(entry["model_path"])
    mtype = entry["type"]

    try:
        df_real = load_adult(balanced=True)
    except TypeError:
        df_real = load_adult()

    # -------------------------
    # CTGAN (pickle)
    # -------------------------
    if mtype == "ctgan":
        with open(model_path, "rb") as f:
            ctgan = pickle.load(f)
        df_synth = ctgan.sample(int(n))
        return _repair_df(df_synth, df_real, int(n), rebalance_target=True)

    # -------------------------
    # DP-CTGAN (torch)
    # -------------------------
    if mtype == "dp_ctgan":
        ckpt = _load_torch(model_path)
        metadata = ckpt.get("metadata", {})

        data_dim = _pick_first(metadata, "data_dim", default=None)
        if data_dim is None:
            data_dim = _pick_first(ckpt, "data_dim", default=None)
        if data_dim is None:
            raise HTTPException(status_code=500, detail="DP-CTGAN checkpoint missing data_dim.")

        noise_dim = _pick_first(ckpt, "noise_dim", default=None)
        if noise_dim is None:
            noise_dim = _pick_first(metadata, "noise_dim", default=64)

        from models.dp_gan import make_dp_gan

        G, _D = make_dp_gan(
            data_dim=int(data_dim),
            noise_dim=int(noise_dim),
            gen_hidden=DP_CTGAN_CONFIG["generator_dim"],
            disc_hidden=DP_CTGAN_CONFIG["discriminator_dim"],
        )

        g_state = _pick_first(ckpt, "G_state_dict", "G_state", default=None)
        if g_state is None:
            raise HTTPException(status_code=500, detail="DP-CTGAN checkpoint missing generator weights.")

        G.load_state_dict(g_state)
        G.to(device).eval()

        gen_n = int(n) * 5
        with torch.no_grad():
            z = torch.randn(gen_n, int(noise_dim), device=device)
            synth = G(z).detach().cpu().numpy()

        synth = np.clip(synth, -3.0, 3.0)
        df_synth = reconstruct_from_synthetic(synth, metadata)

        df_synth = _select_best_subset(df_synth, df_real, n=int(n))
        return _repair_df(df_synth, df_real, n=int(n), rebalance_target=True)

    # -------------------------
    # DP DCF-Diffusion (torch)
    # -------------------------
    if mtype == "dp_dcf_diffusion":
        from models.dcf_diffusion import TabularDCF

        ckpt = _load_torch(model_path)
        metadata = ckpt.get("metadata", {})
        data_dim = int(metadata.get("data_dim"))

        is_conditional = bool(ckpt.get("conditional", False)) or ("model_state_dicts" in ckpt and "labels" in ckpt)

        # -------- conditional ----------
        if is_conditional:
            labels = ckpt.get("labels", [])
            state_dicts = ckpt.get("model_state_dicts", {})
            train_dist = ckpt.get("train_dist", {})

            if not labels or not isinstance(state_dicts, dict):
                raise HTTPException(status_code=500, detail="Conditional DCF checkpoint missing labels/state_dicts.")

            counts = _allocate_label_counts([str(x) for x in labels], train_dist, int(n))

            dfs = []
            with torch.no_grad():
                for lab, n_lab in counts:
                    if n_lab <= 0:
                        continue

                    model = TabularDCF(
                        input_dim=data_dim,
                        latent_dim=int(metadata.get("latent_dim", 64)),
                        time_emb_dim=int(metadata.get("time_emb_dim", 64)),
                        width=int(metadata.get("width", 256)),
                        depth=int(metadata.get("depth", 3)),
                        dropout=float(metadata.get("dropout", 0.1)),
                        use_spectral_norm=bool(metadata.get("use_spectral_norm", True)),
                        T=int(metadata.get("T", 100)),
                        recon_weight=float(metadata.get("recon_weight", 1.0)),
                    ).to(device)

                    sd = state_dicts.get(str(lab)) or state_dicts.get(lab)
                    if sd is None:
                        raise HTTPException(status_code=500, detail=f"Missing state_dict for label={lab}")

                    model.load_state_dict(sd)
                    model.eval()

                    x_hat = model.sample(n=int(n_lab), device=device).detach().cpu().numpy()
                    x_hat = np.clip(x_hat, -3.0, 3.0)

                    df_lab = reconstruct_from_synthetic(x_hat, metadata)
                    df_lab[TARGET_COLUMN] = str(lab)  # force label
                    dfs.append(df_lab)

            if not dfs:
                raise HTTPException(status_code=500, detail="Conditional DCF produced no samples.")

            df_synth = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
            df_synth = _fix_exact_n(df_synth, int(n))

            # IMPORTANT: don't rebalance again (conditional sampling already handled it)
            return _repair_df(df_synth, df_real, n=int(n), rebalance_target=False)

        # -------- non-conditional ----------
        if "model_state_dict" not in ckpt:
            raise HTTPException(status_code=500, detail="DCF checkpoint missing model_state_dict.")

        model = TabularDCF(
            input_dim=data_dim,
            latent_dim=int(metadata.get("latent_dim", 64)),
            time_emb_dim=int(metadata.get("time_emb_dim", 64)),
            width=int(metadata.get("width", 256)),
            depth=int(metadata.get("depth", 3)),
            dropout=float(metadata.get("dropout", 0.1)),
            use_spectral_norm=bool(metadata.get("use_spectral_norm", True)),
            T=int(metadata.get("T", 100)),
            recon_weight=float(metadata.get("recon_weight", 1.0)),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        gen_n = int(n) * 5
        with torch.no_grad():
            x_hat = model.sample(n=gen_n, device=device).detach().cpu().numpy()

        x_hat = np.clip(x_hat, -3.0, 3.0)
        df_synth = reconstruct_from_synthetic(x_hat, metadata)

        df_synth = _select_best_subset(df_synth, df_real, n=int(n))
        return _repair_df(df_synth, df_real, n=int(n), rebalance_target=True)

    raise HTTPException(status_code=400, detail=f"Unknown model type: {mtype}")


# -----------------------------
# Public generate endpoint (preview + save CSV)
# -----------------------------
@app.post("/generate")
def generate_synthetic(run_name: str, n: int = 1000, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df_synth = _generate_df(run_name=run_name, n=int(n), device=device)

    out_csv = SYNTH_DIR / f"{run_name}.csv"
    df_synth.to_csv(out_csv, index=False)

    return {
        "run_name": run_name,
        "n_rows": int(len(df_synth)),
        "saved_csv": str(out_csv),
        "data_preview": df_synth.head(20).to_dict("records"),
    }


# -----------------------------
# Visualize endpoint (REAL full counts)
# -----------------------------
@app.post("/visualize")
def visualize(run_name: str, n_synth: int = 10000, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        df_real = load_adult(balanced=True)
    except TypeError:
        df_real = load_adult()

    df_synth = _generate_df(run_name=run_name, n=int(n_synth), device=device)

    def numeric_hist(col: str, bins: int = 20):
        if col not in df_real.columns or col not in df_synth.columns:
            return {"bins": [], "real_counts": [], "synth_counts": []}

        r = pd.to_numeric(df_real[col], errors="coerce").dropna()
        s = pd.to_numeric(df_synth[col], errors="coerce").dropna()
        if len(r) == 0 or len(s) == 0:
            return {"bins": [], "real_counts": [], "synth_counts": []}

        edges = np.histogram_bin_edges(r, bins=bins)
        real_counts, _ = np.histogram(r, bins=edges)
        synth_counts, _ = np.histogram(s, bins=edges)

        mids = ((edges[:-1] + edges[1:]) / 2.0).tolist()
        return {
            "bins": [float(x) for x in mids],
            "real_counts": [int(x) for x in real_counts.tolist()],
            "synth_counts": [int(x) for x in synth_counts.tolist()],
        }

    def cat_hist(col: str):
        if col not in df_real.columns and col not in df_synth.columns:
            return {"categories": [], "real_counts": [], "synth_counts": []}

        r = df_real[col].astype(str).fillna("?").value_counts() if col in df_real.columns else pd.Series(dtype=int)
        s = df_synth[col].astype(str).fillna("?").value_counts() if col in df_synth.columns else pd.Series(dtype=int)

        cats = sorted(set(r.index.tolist()) | set(s.index.tolist()))
        return {
            "categories": cats,
            "real_counts": [int(r.get(k, 0)) for k in cats],
            "synth_counts": [int(s.get(k, 0)) for k in cats],
        }

    return {
        "run_name": run_name,
        "n_synth": int(n_synth),
        "numeric": {
            "age": numeric_hist("age", bins=20),
            "hours_per_week": numeric_hist("hours_per_week", bins=20),
        },
        "categorical": {
            "income": cat_hist("income"),
        },
    }
