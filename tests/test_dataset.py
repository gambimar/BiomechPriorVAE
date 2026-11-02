"""Exercise frame extraction with a small stand-in for the Nimble disk API."""

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.addBiomechanicsDataset import AddBiomechanicsDataset, ground_force_features, target_dof_names


class Skeleton:
    def __init__(self, names):
        self.names = names

    def getNumDofs(self):
        return len(self.names)

    def getDofByIndex(self, index):
        return SimpleNamespace(getName=lambda: self.names[index])


class Subject:
    names = list(reversed(target_dof_names))
    length = 5
    missing = [0] * 5
    truncate = False

    def __init__(self, path):
        self.path = path

    def getNumDofs(self):
        return len(self.names)

    def getNumJoints(self):
        return len(self.names)

    def getGroundForceBodies(self):
        return ['calcn_r', 'calcn_l']

    def getNumTrials(self):
        return 1

    def getTrialLength(self, trial):
        return self.length

    def getMissingGRF(self, trial):
        return self.missing

    def readSkel(self, processing_pass, geometry):
        return Skeleton(self.names)

    def getMassKg(self):
        return 1.

    def getHeightM(self):
        return 2.

    def getTrialName(self, trial):
        return 'walk01'

    def readFrames(self, trial, start, count, stride, **kwargs):
        frames = []
        for index in range(start, min(start + count * stride, self.length), stride):
            values = np.arange(len(self.names), dtype=float) + index
            frame_pass = SimpleNamespace(pos=values, vel=values * 2, tau=values * 3,
                                         groundContactForce=np.array([3., 10., 4., 5., 20., 12.]) * 9.81)
            frames.append(SimpleNamespace(processingPasses=[frame_pass]))
        return frames[:-1] if self.truncate else frames


@pytest.fixture
def dataset_factory(tmp_path, monkeypatch):
    file = tmp_path / 'subject.b3d'
    file.write_bytes(b'0' * 11000)
    monkeypatch.setitem(sys.modules, 'nimblephysics', SimpleNamespace(
        biomechanics=SimpleNamespace(SubjectOnDisk=Subject,
                                     MissingGRFReason=SimpleNamespace(notMissingGRF=0))))
    monkeypatch.setattr(Subject, 'missing', [0] * 5)
    monkeypatch.setattr(Subject, 'names', list(reversed(target_dof_names)))
    monkeypatch.setattr(Subject, 'length', 5)
    monkeypatch.setattr(Subject, 'truncate', False)

    def create(**kwargs):
        return AddBiomechanicsDataset(str(file), geometry_folder='',
                                      skip_loading_skeletons=True, **kwargs)
    return create


def test_single_frame_dataset_includes_last_two_frames(dataset_factory):
    dataset = dataset_factory(window_size=1)
    assert len(dataset) == 5
    data, labels, _, _ = dataset[4]
    assert data['pos'].shape == (1, 23)
    assert data['force'].shape == (1, 4)
    torch.testing.assert_close(data['force'], torch.tensor([[10., 5., 20., 13.]]))
    assert labels['trialname'].item() == 0


def test_strided_window_shapes_and_name_based_ankles(dataset_factory):
    dataset = dataset_factory(window_size=5, stride=2)
    assert len(dataset) == 1
    data, _, _, _ = dataset[0]
    assert all(value.shape[0] == 3 for value in data.values())
    ankle_columns = [target_dof_names.index('ankle_angle_r'), target_dof_names.index('ankle_angle_l')]
    torch.testing.assert_close(data['M_ankle'], data['tau'][:, ankle_columns])
    torch.testing.assert_close(data['force'], torch.tensor([[10., 5., 20., 13.]]).repeat(3, 1))


def test_missing_grf_filters_only_sampled_frames(dataset_factory, monkeypatch):
    monkeypatch.setattr(Subject, 'missing', [0, 1, 0, 0, 0])
    dataset = dataset_factory(window_size=3, stride=2)
    assert dataset.windows == [(0, 0, 0), (0, 0, 2)]


def test_short_read_raises_instead_of_inserting_fake_zeros(dataset_factory, monkeypatch):
    dataset = dataset_factory(window_size=1)
    monkeypatch.setattr(Subject, 'truncate', True)
    with pytest.raises(RuntimeError, match='Expected 1 frames'):
        dataset[0]


def test_missing_required_joint_rejected_at_initialization(dataset_factory, monkeypatch):
    monkeypatch.setattr(Subject, 'names', target_dof_names[:-1])
    with pytest.raises(ValueError, match='Required DOF'):
        dataset_factory(window_size=1)


def test_invalid_window_rejected(dataset_factory):
    with pytest.raises(ValueError, match='positive'):
        dataset_factory(window_size=0)


def test_empty_directory_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'nimblephysics', SimpleNamespace())
    with pytest.raises(ValueError, match='No usable .b3d'):
        AddBiomechanicsDataset(str(tmp_path), window_size=1, geometry_folder='')


def test_ground_forces_follow_named_feet_and_ignore_pelvis():
    force = torch.tensor([[5., 20., 12., 999., 999., 999., 3., 10., 4.]]) * 9.81
    actual = ground_force_features(force, ['calcn_l', 'pelvis', 'calcn_r'], mass_kg=1.)
    torch.testing.assert_close(actual, torch.tensor([[10., 5., 20., 13.]]))


def test_ground_forces_sum_multiple_contact_bodies_per_foot():
    force = torch.tensor([[1., 10., 0., 2., 20., 0., 0., 15., 4.]]) * 9.81
    actual = ground_force_features(force, ['heel_r', 'toe_r', 'calcn_l'], mass_kg=1.)
    torch.testing.assert_close(actual, torch.tensor([[30., 3., 15., 4.]]))


def test_ground_forces_exclude_sided_nonfoot_contacts():
    force = torch.tensor([[0., 1., 0., 0., 10., 0., 0., 1., 0.]]) * 9.81
    actual = ground_force_features(force, ['calcn_r', 'hand_r', 'calcn_l'], mass_kg=1.)
    torch.testing.assert_close(actual, torch.tensor([[1., 0., 1., 0.]]))


def test_worker_pickle_restores_readers_and_can_be_pickled_again(dataset_factory):
    dataset = dataset_factory(window_size=1)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.windows == dataset.windows
    torch.testing.assert_close(restored[4][0]['pos'], dataset[4][0]['pos'])
    assert restored.skeletons == []
    assert len(pickle.loads(pickle.dumps(restored))) == len(dataset)


def test_invalid_trial_metadata_fails_instead_of_silently_dropping_data(dataset_factory, monkeypatch):
    def missing_flags(self, trial):
        raise OSError('broken header')

    monkeypatch.setattr(Subject, 'getMissingGRF', missing_flags)
    with pytest.raises(RuntimeError, match='Error accessing trial 0'):
        dataset_factory(window_size=1)
