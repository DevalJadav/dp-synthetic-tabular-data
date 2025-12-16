# training/evaluation.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _rand_seed() -> int:
    # reproducible per-process randomness
    return int(np.random.randint(0, 2**31 - 1))


def safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # keep original y type (Series) for downstream .values/.astype etc.
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)

    if len(unique) < 2 or counts.min() < 2:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



def _encode_align(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode both, align to SAME columns.
    Use OUTER join to keep union of columns (safer than left).
    """
    A = pd.get_dummies(df_a, drop_first=False)
    B = pd.get_dummies(df_b, drop_first=False)
    A2, B2 = A.align(B, join="outer", axis=1, fill_value=0)
    return A2, B2


def rebalance_like_real(
    df_synth: pd.DataFrame,
    df_real: pd.DataFrame,
    target: str,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Optional helper (not used by default in evaluate) — made non-deterministic via random_state.
    """
    if target not in df_synth.columns or target not in df_real.columns:
        return df_synth

    real_dist = df_real[target].astype(str).value_counts(normalize=True)
    df_synth = df_synth.copy()
    df_synth[target] = df_synth[target].astype(str)

    n = len(df_synth)
    parts = []
    for cls, p in real_dist.items():
        want = int(round(n * float(p)))
        sub = df_synth[df_synth[target] == str(cls)]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(want, replace=True, random_state=random_state))

    if not parts:
        return df_synth

    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=random_state)
    elif len(out) < n:
        out = pd.concat(
            [out, df_synth.sample(n - len(out), replace=True, random_state=random_state)],
            ignore_index=True,
        )

    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _build_clf() -> Any:
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="saga",
            max_iter=5000,
            n_jobs=-1,
        ),
    )


def evaluate_synthetic_utility(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    target_column: str = "income",
    n_synth: int = 10000,
    eval_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Utility metric:
      - Train LogisticRegression on REAL-TRAIN → evaluate on REAL-TEST
      - Train LogisticRegression on SYNTH (optionally sampled to n_synth) → evaluate on REAL-TEST

    Key fix: eval_seed is random by default so you don't get identical metrics every run.
    """
    if eval_seed is None:
        eval_seed = _rand_seed()

    df_real = df_real.copy()
    df_synth = df_synth.copy()

    # strip whitespace column names + values
    df_real.columns = df_real.columns.astype(str).str.strip()
    df_synth.columns = df_synth.columns.astype(str).str.strip()

    if target_column not in df_real.columns:
        return {"error": f"'{target_column}' missing in real data", "eval_seed": int(eval_seed)}

    if target_column not in df_synth.columns:
        return {
            "eval_seed": int(eval_seed),
            "real_train_real_test": None,
            "synth_train_real_test": {
                "acc": None,
                "f1": None,
                "auc": None,
                "note": "target column missing in synthetic data",
            },
        }

    df_real[target_column] = df_real[target_column].astype(str).str.strip()
    df_synth[target_column] = df_synth[target_column].astype(str).str.strip()

    # pick a stable pos_label based on REAL labels
    real_labels = sorted(df_real[target_column].unique().tolist())
    pos_label = real_labels[-1] if len(real_labels) >= 2 else None

    # -------------------------
    # 1) Real → Real
    # -------------------------
    X = df_real.drop(columns=[target_column])
    y = df_real[target_column]

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.2, random_state=int(eval_seed))
    Xtr_enc, Xte_enc = _encode_align(X_train, X_test)

    clf = _build_clf()
    clf.fit(Xtr_enc, y_train)

    y_pred = clf.predict(Xte_enc)
    if pos_label is not None and len(clf.classes_) == 2 and hasattr(clf, "predict_proba"):
        # probability for pos_label
        pos_idx = int(np.where(clf.classes_ == pos_label)[0][0]) if pos_label in clf.classes_ else 1
        y_prob = clf.predict_proba(Xte_enc)[:, pos_idx]
        y_true_bin = (np.asarray(y_test) == pos_label).astype(int)
        auc = float(roc_auc_score(y_true_bin, y_prob))
        f1 = float(f1_score(y_test, y_pred, average="binary", pos_label=pos_label))
    else:
        auc = None
        f1 = None

    real_metrics = {
        "acc": float(accuracy_score(y_test, y_pred)),
        "f1": f1,
        "auc": auc,
    }

    # -------------------------
    # 2) Synth → Real
    # -------------------------
    # (optional) sample exactly n_synth from synth to reduce bias + add controlled randomness
    if n_synth is not None and len(df_synth) > 0:
        take = int(min(max(1, n_synth), len(df_synth)))
        df_synth_train = df_synth.sample(take, replace=(take > len(df_synth)), random_state=int(eval_seed)).reset_index(drop=True)
    else:
        df_synth_train = df_synth

    y_s = df_synth_train[target_column]
    if y_s.nunique() < 2 or pos_label is None:
        synth_metrics = {
            "acc": None,
            "f1": None,
            "auc": None,
            "note": "synthetic labels collapsed OR real pos_label missing",
        }
    else:
        Xs = df_synth_train.drop(columns=[target_column])
        Xs_enc, Xte_enc2 = _encode_align(Xs, X_test)

        clf2 = _build_clf()
        clf2.fit(Xs_enc, y_s)

        y_pred2 = clf2.predict(Xte_enc2)

        if len(clf2.classes_) == 2 and hasattr(clf2, "predict_proba"):
            pos_idx2 = int(np.where(clf2.classes_ == pos_label)[0][0]) if pos_label in clf2.classes_ else 1
            y_prob2 = clf2.predict_proba(Xte_enc2)[:, pos_idx2]
            y_true_bin2 = (np.asarray(y_test) == pos_label).astype(int)
            auc2 = float(roc_auc_score(y_true_bin2, y_prob2))
            f12 = float(f1_score(y_test, y_pred2, average="binary", pos_label=pos_label))
        else:
            auc2 = None
            f12 = None

        synth_metrics = {
            "acc": float(accuracy_score(y_test, y_pred2)),
            "f1": f12,
            "auc": auc2,
        }

    real_dist = df_real[target_column].value_counts(normalize=True).to_dict()
    synth_dist = df_synth[target_column].value_counts(normalize=True).to_dict()
    dup_ratio = float(df_synth.duplicated().mean()) if len(df_synth) else 0.0

    return {
        "eval_seed": int(eval_seed),
        "real_train_real_test": real_metrics,
        "synth_train_real_test": synth_metrics,
        "quality": {
            "real_label_dist": real_dist,
            "synth_label_dist": synth_dist,
            "synth_duplicate_ratio": dup_ratio,
        },
    }


# Backward-compatible alias
def evaluate_synth_vs_real(df_real, df_synth, target_column: str = "income", **kwargs):
    return evaluate_synthetic_utility(
        df_real=df_real,
        df_synth=df_synth,
        target_column=target_column,
        **kwargs,
    )
