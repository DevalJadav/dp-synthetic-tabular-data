# agents/agent_pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from training.config import MODELS_DIR, TARGET_COLUMN, SENSITIVE_COLUMNS
from training.data_utils import load_adult
from training.dp_ctgan_train import train_dp_ctgan
from training.dp_dcf_diffusion_train import train_dp_dcf_diffusion


REG_PATH = MODELS_DIR / "registry.json"


# ---------------------------
# Agent 1: Registry Agent
# ---------------------------
class RegistryAgent:
    def load(self) -> List[Dict[str, Any]]:
        if not REG_PATH.exists() or not REG_PATH.read_text().strip():
            return []
        try:
            obj = json.loads(REG_PATH.read_text())
        except Exception:
            return []

        # registry.json MUST be a LIST for your HTML UI
        if isinstance(obj, list):
            return [e for e in obj if isinstance(e, dict) and e.get("run_name")]

        # If it was a dict, convert safely
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                if isinstance(v, dict) and v.get("run_name"):
                    out.append(v)
            return out

        return []

    def save(self, entries: List[Dict[str, Any]]) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        REG_PATH.write_text(json.dumps(entries, indent=2))

    def upsert(self, entry: Dict[str, Any]) -> None:
        reg = self.load()
        reg = [e for e in reg if e.get("run_name") != entry.get("run_name")]
        reg.insert(0, entry)  # newest first
        self.save(reg)


# ---------------------------
# Agent 2: Training Agent
# ---------------------------
class TrainingAgent:
    def run(self, device: Optional[str] = None) -> Dict[str, Any]:
        ct = train_dp_ctgan(device=device)
        dcf = train_dp_dcf_diffusion(device=device)
        return {"dp_ctgan": ct, "dp_dcf_diffusion": dcf}


# ---------------------------
# Agent 3: Fairness Agent (light summary)
# ---------------------------
class FairnessAgent:
    def demographic_parity_diff(self, df: pd.DataFrame, target: str, group: str) -> Optional[float]:
        if target not in df.columns or group not in df.columns:
            return None

        y = df[target].astype(str)
        if y.nunique() < 2:
            return None

        # choose a positive label deterministically
        pos = sorted(y.unique())[-1]

        df2 = df[[target, group]].copy()
        df2["_y"] = (df2[target].astype(str) == pos).astype(int)

        rates = df2.groupby(group)["_y"].mean().dropna()
        if len(rates) < 2:
            return None
        return float(rates.max() - rates.min())

    def run(self, df_synth: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        for col in SENSITIVE_COLUMNS:
            out[col] = self.demographic_parity_diff(df_synth, TARGET_COLUMN, col)
        return {"demographic_parity_diff": out}


# ---------------------------
# Agent 3.5: Quality Gate Agent
# ---------------------------
class QualityGateAgent:
    def __init__(self, min_auc: float = 0.52, max_imbalance: float = 0.35, max_dup_ratio: float = 0.05):
        self.min_auc = float(min_auc)
        self.max_imbalance = float(max_imbalance)
        self.max_dup_ratio = float(max_dup_ratio)

    def _label_imbalance(self, dist: Dict[str, float]) -> float:
        if not dist or len(dist) < 2:
            return 2.0  # collapsed / missing
        p = np.array(list(dist.values()), dtype=float)
        p = p / (p.sum() + 1e-9)
        return float(np.abs(p - (1.0 / len(p))).sum())

    def check(self, entry: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        m = entry.get("metrics") or {}
        synth = (m.get("synth_train_real_test") or {})
        q = (m.get("quality") or {})

        auc = synth.get("auc")
        auc_val = float(auc) if auc is not None else None

        dup = float(q.get("synth_duplicate_ratio", 0.0))
        dist = q.get("synth_label_dist") or {}
        imb = self._label_imbalance(dist)

        reasons = []
        if auc_val is None or auc_val < self.min_auc:
            reasons.append(f"auc<{self.min_auc} (got {auc})")
        if imb > self.max_imbalance:
            reasons.append(f"label_imbalance>{self.max_imbalance} (got {imb:.3f})")
        if dup > self.max_dup_ratio:
            reasons.append(f"dup_ratio>{self.max_dup_ratio} (got {dup:.3f})")

        ok = (len(reasons) == 0)
        report = {
            "ok": ok,
            "auc": auc_val,
            "label_imbalance": float(imb),
            "dup_ratio": float(dup),
            "reasons": reasons,
            "thresholds": {
                "min_auc": self.min_auc,
                "max_imbalance": self.max_imbalance,
                "max_dup_ratio": self.max_dup_ratio,
            },
        }
        return ok, report


# ---------------------------
# Agent 4: Selection Agent
# ---------------------------
class SelectionAgent:
    def pick_best(self, dp_ctgan_entry: Dict[str, Any], dp_dcf_entry: Dict[str, Any]) -> Dict[str, Any]:
        ct_gate = (dp_ctgan_entry.get("metrics") or {}).get("quality_gate") or {}
        dcf_gate = (dp_dcf_entry.get("metrics") or {}).get("quality_gate") or {}

        ct_ok = bool(ct_gate.get("ok", False))
        dcf_ok = bool(dcf_gate.get("ok", False))

        if ct_ok and not dcf_ok:
            dp_ctgan_entry.setdefault("metrics", {})["selection_note"] = "ctgan_passed_gate_only"
            return dp_ctgan_entry
        if dcf_ok and not ct_ok:
            dp_dcf_entry.setdefault("metrics", {})["selection_note"] = "dcf_passed_gate_only"
            return dp_dcf_entry

        # If neither/both pass, pick higher AUC
        ct_auc = ((dp_ctgan_entry.get("metrics") or {}).get("synth_train_real_test") or {}).get("auc") or 0.0
        dcf_auc = ((dp_dcf_entry.get("metrics") or {}).get("synth_train_real_test") or {}).get("auc") or 0.0

        chosen = dp_ctgan_entry if float(ct_auc) >= float(dcf_auc) else dp_dcf_entry
        chosen.setdefault("metrics", {})["selection_note"] = "picked_higher_auc"
        return chosen


# ---------------------------
# Orchestrator function used by API
# ---------------------------
def orchestrate_training(device: Optional[str] = None) -> Dict[str, Any]:
    reg_agent = RegistryAgent()
    train_agent = TrainingAgent()
    gate_agent = QualityGateAgent(min_auc=0.52, max_imbalance=0.35, max_dup_ratio=0.05)
    select_agent = SelectionAgent()

    results = train_agent.run(device=device)

    for k in ["dp_ctgan", "dp_dcf_diffusion"]:
        entry = results.get(k)
        if not entry:
            continue

        entry.setdefault("metrics", {})
        entry["metrics"].setdefault("fairness", {"demographic_parity_diff": {c: None for c in SENSITIVE_COLUMNS}})

        ok, gate_report = gate_agent.check(entry)
        entry["metrics"]["quality_gate"] = gate_report

        reg_agent.upsert(entry)

    dp_ct = results["dp_ctgan"]
    dp_dcf = results["dp_dcf_diffusion"]
    best = select_agent.pick_best(dp_ct, dp_dcf)

    return {
        "dp_ctgan": dp_ct,
        "dp_dcf_diffusion": dp_dcf,
        "best_model": {
            "run_name": best.get("run_name"),
            "type": best.get("type"),
            "score_note": "quality_gate_then_auc",
        },
        "registry_path": str(REG_PATH),
    }
