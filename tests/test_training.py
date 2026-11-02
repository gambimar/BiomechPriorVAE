"""Regression tests for data normalization and sequence-shaped VAE losses."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import vaetrainer


class Frames(Dataset):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.values = torch.tensor([0., 2., 7., 20., 100., 200.])

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index].reshape(1, 1)
        return {'pos': value, 'vel': value * 2}, {}, 0, 0


def test_scaler_matches_all_training_frames_and_uses_passed_paths(monkeypatch):
    captured = []

    def dataset(*args, **kwargs):
        frames = Frames(*args, **kwargs)
        captured.append(frames)
        return frames

    monkeypatch.setattr(vaetrainer, 'AddBiomechanicsDataset', dataset)
    train, val, scaler = vaetrainer.load_data('chosen-data', 'chosen-geometry', batch_size=2)
    values = np.array([captured[0][i][0]['pos'].item() for i in train.dataset.indices])
    expected = np.stack((values, values * 2), axis=-1)
    np.testing.assert_allclose(scaler['mean_'].numpy(), expected.mean(axis=0))
    np.testing.assert_allclose(scaler['scale_'].numpy(), expected.std(axis=0), rtol=1e-6)
    assert all(value.device.type == 'cpu' for value in scaler.values())
    assert captured[0].args == ('chosen-data',)
    assert captured[0].kwargs['geometry_folder'] == 'chosen-geometry'
    assert captured[0].kwargs['skip_loading_skeletons'] is True
    assert len(val.dataset) == 2


def test_scaler_is_invariant_to_partial_batch_size(monkeypatch):
    monkeypatch.setattr(vaetrainer, 'AddBiomechanicsDataset', Frames)
    scalers = [vaetrainer.load_data('data', '', batch_size=size, state_keys=('pos',))[2]
               for size in (1, 3, 4)]
    for scaler in scalers[1:]:
        torch.testing.assert_close(scaler['mean_'], scalers[0]['mean_'])
        torch.testing.assert_close(scaler['scale_'], scalers[0]['scale_'])


def test_supplied_scaler_skips_refitting(monkeypatch):
    class UnreadFrames(Frames):
        def __getitem__(self, index):
            raise AssertionError('An existing scaler must not trigger another data scan')

    monkeypatch.setattr(vaetrainer, 'AddBiomechanicsDataset', UnreadFrames)
    scaler = {'mean_': torch.zeros(2), 'scale_': torch.ones(2)}
    assert vaetrainer.load_data('data', '', scaler=scaler)[2] is scaler


@pytest.mark.parametrize('train_split', [0, 1, -0.5, 2])
def test_invalid_split_fails_before_loading(train_split):
    with pytest.raises(ValueError, match='train_split'):
        vaetrainer.load_data('missing', '', train_split=train_split)


def test_kl_sums_latent_axis_and_averages_frames():
    x = torch.ones(2, 3, 4)
    mu = torch.ones(2, 3, 5)
    logvar = torch.zeros_like(mu)
    total, recon, kl = vaetrainer.vae_loss(x, (torch.zeros_like(x), torch.ones_like(x)), mu, logvar, 0.2)
    assert recon.item() == pytest.approx(2.)
    assert kl.item() == pytest.approx(2.5)
    assert total.item() == pytest.approx(0.7)
    flat = vaetrainer.vae_loss(x.flatten(0, 1),
                              (torch.zeros(6, 4), torch.ones(6, 4)),
                              mu.flatten(0, 1), logvar.flatten(0, 1), 0.2)
    for a, b in zip((total, recon, kl), flat):
        torch.testing.assert_close(a, b)


class ConstantVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        recon = torch.zeros_like(x) + self.bias
        latent = torch.zeros((*x.shape[:-1], 3), device=x.device) + self.bias
        return (recon, torch.ones_like(recon)), latent, torch.zeros_like(latent)


def test_validation_weights_short_final_batch_by_frames():
    trainer = vaetrainer.BiomechPriorVAETrainer(ConstantVAE(), device='cpu', state_keys=('pos',))
    dataset = Frames()
    expected = 0.05 * dataset.values.square().mean().item()
    for batch_size in (1, 4, 6):
        metrics = trainer.validate(DataLoader(dataset, batch_size=batch_size))
        assert metrics['loss'] == pytest.approx(expected)


def test_training_and_test_run_with_explicit_features_and_cpu_scaler():
    model = vaetrainer.BiomechPriorVAE(num_dofs=1, latent_dim=2, hidden_dim=8)
    trainer = vaetrainer.BiomechPriorVAETrainer(model, device='cpu', state_keys=('pos',))
    loader = DataLoader(Frames(), batch_size=4)
    scaler = {'mean_': torch.tensor([20.]), 'scale_': torch.tensor([30.])}
    metrics = trainer.train_epoch(loader, torch.optim.Adam(model.parameters()), scaler=scaler)
    assert all(np.isfinite(value) for value in metrics.values())
    result = trainer.test(loader, scaler)
    assert result[0]['recon'].shape == (4, 1, 1)
    assert result[0]['original'][0, 0, 0] == pytest.approx(-2 / 3)


def test_empty_loaders_have_clear_errors():
    trainer = vaetrainer.BiomechPriorVAETrainer(ConstantVAE(), device='cpu')
    with pytest.raises(ValueError, match='Validation loader is empty'):
        trainer.validate([])
    with pytest.raises(ValueError, match='Training loader is empty'):
        trainer.train_epoch([], torch.optim.Adam(trainer.model.parameters()))


def test_default_training_configuration_saves_and_reloads_portable_artifacts(tmp_path, monkeypatch):
    class FullFrames(Frames):
        forbid_reads = False

        def __getitem__(self, index):
            assert not self.forbid_reads, 'Reloading must reuse the saved scaler'
            data, *metadata = super().__getitem__(index)
            return {key: value.repeat(1, 23) for key, value in data.items()}, *metadata

    original_init = vaetrainer.BiomechPriorVAE.__init__
    monkeypatch.setattr(vaetrainer, 'AddBiomechanicsDataset', FullFrames)
    monkeypatch.setattr(vaetrainer.BiomechPriorVAE, '__init__',
                        lambda self, num_dofs, latent_dim: original_init(self, num_dofs, latent_dim, hidden_dim=8))
    monkeypatch.setattr(vaetrainer.BiomechPriorVAETrainer, 'visualize_history', lambda self: None)
    model, _ = vaetrainer.train_model('data', '', str(tmp_path), num_epochs=1)
    assert model.num_dofs == 46
    FullFrames.forbid_reads = True
    restored, _, scaler = vaetrainer.test_model(
        'data', '', str(tmp_path / 'BiomechPriorVAE_best_46.pth'), str(tmp_path / 'scaler_46.pkl'))
    assert restored.num_dofs == 46
    assert scaler['mean_'].shape == (46,)
    assert all(tensor.device.type == 'cpu' for tensor in scaler.values())
