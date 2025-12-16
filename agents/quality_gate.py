# agents/quality_gate.py
from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np

def _label_imbalance(dist: Dict[str, float]) -> float:
    """0 = perfect balance, higher = more skew. Works for 2+ classes."""
    if not dist or len(dist) < 2:
        return 10.0  # collapsed or missing
    p = np.array(list(dist.values()), dtype=float)
    p = p / (p.sum() + 1e-9)
    return float(np.abs(p - (1.0 / len(p))).sum())

class QualityGateAgent:
    """
    Hard gates (must pass):
    - synth AUC must be >= min_auc
    - label imbalance must be <= max_imbalance
    - duplicates must be <= max_dup_ratio
    """
    def __init__(
        self,
        min_auc: float = 0.55,
        max_imbalance: float = 0.60,
        max_dup_ratio: float = 0.05,
    ):
        self.min_auc = float(min_auc)
        self.max_imbalance = float(max_imbalance)
        self.max_dup_ratio = float(max_dup_ratio)

    def check(self, entry: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        m = entry.get("metrics") or {}
        synth = (m.get("synth_train_real_test") or {})
        q = (m.get("quality") or {})

        auc = synth.get("auc")
        dup = q.get("synth_duplicate_ratio", 0.0)
        dist = q.get("synth_label_dist") or {}

        imbalance = _label_imbalance(dist)

        reasons = []
        if auc is None or float(auc) < self.min_auc:
            reasons.append(f"auc<{self.min_auc} (got {auc})")
        if float(imbalance) > self.max_imbalance:
            reasons.append(f"label_imbalance>{self.max_imbalance} (got {imbalance:.3f})")
        if float(dup) > self.max_dup_ratio:
            reasons.append(f"dup_ratio>{self.max_dup_ratio} (got {dup})")

        ok = (len(reasons) == 0)
        return ok, {
            "ok": ok,
            "auc": auc,
            "label_imbalance": float(imbalance),
            "dup_ratio": float(dup),
            "reasons": reasons,
        }
