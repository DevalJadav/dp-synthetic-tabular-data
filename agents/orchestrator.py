# agents/orchestrator.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch

from training.config import MODELS_DIR
from training.dp_ctgan_train import train_dp_ctgan
from training.dp_dcf_diffusion_train import train_dp_dcf_diffusion


REGISTRY_PATH = MODELS_DIR / "registry.json"
BEST_PATH = MODELS_DIR / "best_model.json"


def _read_registry_list() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists() or not REGISTRY_PATH.read_text().strip():
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _write_registry_list(entries: List[Dict[str, Any]]) -> None:
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2))


def _score_entry(entry: Dict[str, Any]) -> float:
    """
    Higher is better. Use synth_train_real_test AUC if present, else accuracy, else -inf.
    """
    m = (entry.get("metrics") or {})
    s = (m.get("synth_train_real_test") or {})
    auc = s.get("auc")
    acc = s.get("acc")
    if isinstance(auc, (int, float)):
        return float(auc)
    if isinstance(acc, (int, float)):
        return float(acc)
    return float("-inf")


def _pick_best(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [e for e in entries if isinstance(e, dict) and e.get("run_name") and e.get("type")]
    if not valid:
        return None
    return max(valid, key=_score_entry)


def orchestrate_training(device: Optional[str] = None) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Any] = {
        "device": device,
        "trained": {},
        "best_model": None,
    }

    registry = _read_registry_list()

    # -------------------
    # Train DP-CTGAN
    # -------------------
    try:
        dp_ctgan_entry = train_dp_ctgan(device=device)
        registry.append(dp_ctgan_entry)
        results["trained"]["dp_ctgan"] = dp_ctgan_entry
    except Exception as e:
        results["trained"]["dp_ctgan"] = {"status": "failed", "error": str(e)}

    # -------------------
    # Train DP DCF-Diffusion
    # -------------------
    try:
        dp_dcf_entry = train_dp_dcf_diffusion(device=device)
        registry.append(dp_dcf_entry)
        results["trained"]["dp_dcf_diffusion"] = dp_dcf_entry
    except Exception as e:
        results["trained"]["dp_dcf_diffusion"] = {"status": "failed", "error": str(e)}

    # Save registry
    _write_registry_list(registry)

    # Pick best and save separately
    best = _pick_best(registry)
    results["best_model"] = best
    if best is not None:
        BEST_PATH.write_text(json.dumps(best, indent=2))

    return results
