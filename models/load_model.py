import torch
import json
from pathlib import Path
import pandas as pd

from training.data_utils import reconstruct_from_synthetic
from models.dcf_diffusion import DCFDiffusionModel
from models.dp_gan import make_dp_gan   # For CTGAN / DP-CTGAN


MODELS_DIR = Path("models")


def load_model_and_generate(run_name, n):
    registry = json.loads((MODELS_DIR / "registry.json").read_text())

    row = next((r for r in registry if r["run_name"] == run_name), None)
    if row is None:
        raise ValueError("Model not found in registry: " + run_name)

    ckpt = torch.load(row["model_path"], map_location="cpu")
    metadata = ckpt["metadata"]

    # -------------- DIFFUSION MODEL --------------
    if row["type"] == "dp_dcf_diffusion":
        dim = metadata["data_dim"]
        model = DCFDiffusionModel(dim).cpu()
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        synth = model.sample(n).detach().numpy()
        synth = synth.clip(-3, 3)

        return reconstruct_from_synthetic(synth, metadata)

    # -------------- GAN MODELS --------------
    else:
        dim = metadata["data_dim"]
        G, D = make_dp_gan(dim, noise_dim=64)
        G.load_state_dict(ckpt["G_state_dict"])
        G.eval()

        z = torch.randn(n, 64)
        synth = G(z).detach().numpy().clip(-3, 3)
        return reconstruct_from_synthetic(synth, metadata)