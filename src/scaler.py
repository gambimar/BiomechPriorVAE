"""Load existing scaler artifacts on the requested device."""

import io
import pickle
from pathlib import Path

import torch


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=True
            )
        return super().find_class(module, name)


def load_scaler(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load a trusted scaler pickle, including legacy CUDA tensor artifacts."""
    with Path(path).open("rb") as stream:
        scaler = _CPUUnpickler(stream).load()
    if not isinstance(scaler, dict) or not {"mean_", "scale_"} <= scaler.keys():
        raise ValueError("Scaler must be a dictionary with 'mean_' and 'scale_' keys")
    result = {
        key: torch.as_tensor(scaler[key], dtype=torch.float32, device=device)
        for key in ("mean_", "scale_")
    }
    if result["mean_"].ndim != 1 or result["mean_"].shape != result["scale_"].shape:
        raise ValueError("Scaler mean_ and scale_ must be matching feature vectors")
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Scaler values must be finite")
    if not (result["scale_"] > 0).all():
        raise ValueError("Scaler scale_ values must be positive")
    return result
