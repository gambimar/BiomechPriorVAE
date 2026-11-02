"""Smoke-test the published 50-D prior using a full simulation-state input."""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_published_prior() -> None:
    """Check that the checkpoint produces a finite objective and gradient."""
    model_path = REPO_ROOT / "result/model/BiomechPriorVAE_best_50.pth"
    scaler_path = REPO_ROOT / "result/model/scaler_50.pkl"
    if not model_path.is_file() or not scaler_path.is_file():
        pytest.skip("Published model/scaler are not available in this checkout")

    from src.vaemodel import VAEModelWrapper

    wrapper = VAEModelWrapper(model_path, scaler_path, num_dofs=50, latent_dim=24)
    state = np.zeros(123, dtype=np.float32)
    error = wrapper.reconstruct(state)
    gradient = wrapper.reconstruct_withgrad(state)
    assert np.isfinite(error).all()
    assert gradient.shape == state.shape
    assert np.isfinite(gradient).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-v"]))
