"""Inference regressions using small synthetic artifacts and the published model."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src import vaemodel
from src.scaler import load_scaler
from src.vaetrainer import BiomechPriorVAE


@pytest.fixture
def artifacts(tmp_path):
    def create(num_dofs=50, scale=1.0, state_keys=None):
        torch.manual_seed(17)
        model_path = tmp_path / f'model_{num_dofs}.pth'
        scaler_path = tmp_path / f'scaler_{num_dofs}.pkl'
        torch.save(BiomechPriorVAE(num_dofs, latent_dim=3, hidden_dim=8).state_dict(), model_path)
        with scaler_path.open('wb') as stream:
            pickle.dump({'mean_': np.zeros(num_dofs), 'scale_': np.full(num_dofs, scale)}, stream)
        return vaemodel.VAEModelWrapper(
            model_path, scaler_path, num_dofs=num_dofs, latent_dim=3,
            hidden_dim=8, state_keys=state_keys,
        )
    return create


@pytest.mark.parametrize('width', [123, 135])
def test_batched_score_and_gradient_match_individual_frames(artifacts, width):
    wrapper = artifacts()
    frames = np.random.default_rng(11).normal(0, 0.2, (3, width)).astype(np.float32)
    expected_scores = [wrapper.reconstruct(frame) for frame in frames]
    expected_gradients = np.stack([wrapper.reconstruct_withgrad(frame) for frame in frames]) / len(frames)
    np.testing.assert_allclose(wrapper.reconstruct(frames), np.mean(expected_scores), rtol=2e-5)
    np.testing.assert_allclose(wrapper.reconstruct_withgrad(frames), expected_gradients, rtol=3e-4, atol=2e-6)
    assert all(parameter.grad is None for parameter in wrapper.model.parameters())


@pytest.mark.parametrize('width', [123, 135])
def test_input_gradient_matches_finite_differences(artifacts, width):
    wrapper = artifacts()
    frame = np.random.default_rng(19).normal(0, 0.1, width).astype(np.float32)
    gradient = wrapper.reconstruct_withgrad(frame)
    for index in (0, 3, 10, 27, 37, 54, 62, 78, width - 1):
        plus, minus = frame.copy(), frame.copy()
        plus[index] += 1e-3
        minus[index] -= 1e-3
        numeric = (wrapper.reconstruct(plus) - wrapper.reconstruct(minus)) / 2e-3
        np.testing.assert_allclose(gradient[index], numeric, rtol=0.03, atol=3e-3)


def test_score_does_not_build_graph_and_gradient_overrides_no_grad(artifacts):
    wrapper = artifacts()
    modes = []
    handle = wrapper.model.encoder.register_forward_pre_hook(lambda *args: modes.append(torch.is_grad_enabled()))
    frame = np.zeros(123, dtype=np.float32)
    wrapper.reconstruct(frame)
    with torch.no_grad():
        wrapper.reconstruct_withgrad(frame)
    handle.remove()
    assert modes == [False, True]


def test_clamped_variance_has_zero_derivative(artifacts):
    wrapper = artifacts(scale=1e-5)

    class VarianceOnlyModel:
        def encode(self, value):
            return value, value

        def decode(self, value):
            return torch.zeros_like(value), torch.exp(value)

    wrapper.model = VarianceOnlyModel()
    frame = np.zeros(123, dtype=np.float32)
    # At zero angles the squared residual is zero. Variances remain below eps.
    assert wrapper.reconstruct_withgrad(frame)[0] == 0


@pytest.mark.parametrize('num_dofs', [23, 25, 46, 48, 50, 52, 69])
def test_training_layouts_are_supported(artifacts, num_dofs):
    wrapper = artifacts(num_dofs)
    assert wrapper.mask.sum() == num_dofs
    assert np.isfinite(wrapper.reconstruct(np.zeros(123, dtype=np.float32)))


def test_explicit_layout_preserves_feature_order(artifacts):
    wrapper = artifacts(27, state_keys=('force', 'pos'))
    assert wrapper.feature_indices[:4].tolist() == [54, 55, 56, 57]
    assert wrapper.feature_indices[-1].item() == 26
    with pytest.raises(ValueError, match='select'):
        artifacts(27, state_keys=('pos',))


@pytest.mark.parametrize('frame', [np.zeros(54), np.zeros((1, 2, 123)), np.zeros((0, 123)), np.full(123, np.nan)])
def test_invalid_input_fails_before_network(artifacts, frame):
    with pytest.raises(ValueError):
        artifacts().reconstruct(frame)


def test_failed_reinitialization_clears_previous_model(artifacts):
    vaemodel._vae_instance = artifacts()
    try:
        assert not vaemodel.initialize_vae('missing.pth', 'missing.pkl')
        with pytest.raises(RuntimeError, match='initialized'):
            vaemodel.reconstruct(np.zeros(123))
    finally:
        vaemodel._vae_instance = None


def test_published_scaler_loads_without_original_storage_loader(monkeypatch):
    def forbid_storage(*args, **kwargs):
        raise AssertionError('legacy device-dependent storage loader was called')

    monkeypatch.setattr(torch.storage, '_load_from_bytes', forbid_storage)
    root = Path(__file__).resolve().parents[1]
    scaler = load_scaler(root / 'result/model/scaler_50.pkl')
    assert scaler['mean_'].shape == (50,)
    assert scaler['mean_'].device.type == 'cpu'


def test_published_model_cpu_matlab_bridge():
    root = Path(__file__).resolve().parents[1]
    assert vaemodel.initialize_vae(
        root / 'result/model/BiomechPriorVAE_best_50.pth',
        root / 'result/model/scaler_50.pkl', num_dofs=50, latent_dim=24,
    )
    try:
        frames = np.zeros((123, 2), dtype=np.float64)
        score = vaemodel.reconstruct(frames)
        gradient = vaemodel.reconstruct_withgrad(frames)
        assert np.isfinite(score)
        assert gradient.shape == (2, 123)
        assert np.isfinite(gradient).all()
    finally:
        vaemodel._vae_instance = None


@pytest.mark.parametrize('scaler', [
    {'mean_': [0, 0], 'scale_': [1]},
    {'mean_': [0], 'scale_': [0]},
    {'mean_': [float('nan')], 'scale_': [1]},
])
def test_scaler_rejects_invalid_statistics(tmp_path, scaler):
    path = tmp_path / 'invalid.pkl'
    with path.open('wb') as stream:
        pickle.dump(scaler, stream)
    with pytest.raises(ValueError):
        load_scaler(path)
