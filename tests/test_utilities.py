"""Regression checks for conversion, visualisation and export utilities."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fake_nimble(monkeypatch):
    nimble = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "nimblephysics", nimble)
    return nimble


@pytest.mark.parametrize("converter_name", ["B3DConverter", "Gait3dB3DConverter"])
def test_converter_defaults_to_remaining_trial(fake_nimble, converter_name):
    module = importlib.import_module("scripts.b3dconverter")
    converter = getattr(module, converter_name)("")
    if converter_name == "Gait3dB3DConverter":
        converter.target_dof_indices = [2, 0]
    calls = []

    def read_frames(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(processingPasses=[SimpleNamespace(pos=np.arange(3))])]

    subject = SimpleNamespace(getTrialLength=lambda trial: 8, readFrames=read_frames)
    result = converter.convert_single_trial(subject, 0, start_frame=3)
    assert calls[0]["numFramesToRead"] == 5
    np.testing.assert_array_equal(result, [[2, 0]] if converter_name.startswith("Gait") else [[0, 1, 2]])
    assert converter.convert_single_trial(subject, 0, start_frame=8).size == 0
    with pytest.raises(ValueError, match="start_frame"):
        converter.convert_single_trial(subject, 0, start_frame=-1)


def test_converter_rejects_incomplete_dof_mapping(fake_nimble):
    from scripts.b3dconverter import Gait3dB3DConverter

    converter = Gait3dB3DConverter("")
    converter.dof_names = converter.target_dof_names[:-1]
    with pytest.raises(ValueError, match="missing DOFs"):
        converter.dof_mapping()


def test_static_visualisation_yields_cpu(fake_nimble, monkeypatch):
    module = importlib.import_module("src.datavisualize")
    renders, sleeps = [], []
    fake_nimble.NimbleGUI = lambda: SimpleNamespace(
        serve=lambda port: None,
        nativeAPI=lambda: SimpleNamespace(renderSkeleton=lambda skeleton: renders.append(skeleton)),
    )
    # An earlier import may hold another test's placeholder module.
    monkeypatch.setattr(module, "nimble", fake_nimble)

    def sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(module.time, "sleep", sleep)
    visualizer = module.PoseVisualizer.__new__(module.PoseVisualizer)
    visualizer.skeleton = object()
    visualizer.set_pose = lambda positions: None
    visualizer.visualize_pose(np.zeros(33))
    assert len(renders) == 1
    assert sleeps == [0.01]


@pytest.mark.parametrize("name", [
    "scripts.export_latent_embedding",
    "scripts.sample_stratified_thumbnails",
    "scripts.sample_full_dofs_thumbnails",
    "scripts.vtp_to_ply",
    "scripts.test",
])
def test_utilities_import_without_datasets_or_side_effects(name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "src.data.addBiomechanicsDataset", None)
    monkeypatch.setitem(sys.modules, "src.vaemodel", None)
    module = importlib.import_module(name)
    assert callable(getattr(module, "main", getattr(module, "convert_vtp_to_ply", None)))
    assert Path.cwd() == tmp_path
    assert not list(tmp_path.iterdir())


def test_small_embedding_uses_valid_perplexity(monkeypatch):
    import sklearn.manifold
    from scripts.export_utils import embed_latents

    kwargs_seen = []

    class FakeTSNE:
        def __init__(self, **kwargs):
            kwargs_seen.append(kwargs)

        def fit_transform(self, values):
            return values[:, :2]

    monkeypatch.setattr(sklearn.manifold, "TSNE", FakeTSNE)
    result = embed_latents(np.ones((3, 24)))
    assert result.shape == (3, 2)
    assert kwargs_seen[0]["perplexity"] == 2
    with pytest.raises(ValueError, match="At least two"):
        embed_latents(np.zeros((1, 24)))


def test_geometry_conversion_reports_write_failure(monkeypatch, tmp_path):
    from scripts.vtp_to_ply import convert_vtp_to_ply

    (tmp_path / "mesh.vtp").write_text("placeholder")
    reader = SimpleNamespace(SetFileName=lambda name: None, Update=lambda: None, GetOutputPort=lambda: None)
    writer = SimpleNamespace(SetFileName=lambda name: None, SetInputConnection=lambda port: None, Write=lambda: 0)
    monkeypatch.setitem(sys.modules, "vtk", SimpleNamespace(vtkXMLPolyDataReader=lambda: reader, vtkPLYWriter=lambda: writer))
    with pytest.raises(OSError, match="Failed to write"):
        convert_vtp_to_ply(tmp_path)


@pytest.mark.parametrize("name", [
    "scripts.export_latent_embedding",
    "scripts.sample_stratified_thumbnails",
    "scripts.sample_full_dofs_thumbnails",
])
def test_exports_run_without_gradients_and_respect_sample_limit(name, monkeypatch, tmp_path):
    import json
    import torch

    module = importlib.import_module(name)
    dof_names = [
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
        "lumbar_extension", "lumbar_bending", "lumbar_rotation", "arm_flex_r", "arm_add_r",
        "arm_rot_r", "elbow_flex_r", "pro_sup_r", "arm_flex_l", "arm_add_l", "arm_rot_l",
        "elbow_flex_l", "pro_sup_l",
    ]
    frame_pass = SimpleNamespace(pos=np.zeros(23), vel=np.zeros(23), groundContactForce=np.zeros(6))
    skeleton = SimpleNamespace(
        getNumDofs=lambda: 23,
        getDofByIndex=lambda i: SimpleNamespace(getName=lambda: dof_names[i]),
    )
    subject = SimpleNamespace(
        readSkel=lambda *args: skeleton,
        getTrialName=lambda trial: "walk",
        getMassKg=lambda: 70,
        getGroundForceBodies=lambda: ["calcn_r", "calcn_l"],
        readFrames=lambda *args, **kwargs: [SimpleNamespace(processingPasses=[frame_pass])],
    )

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.subjects = [subject]
            self.windows = [(0, 0, i) for i in range(3)]

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return ({
                "pos": torch.zeros((1, 23)), "vel": torch.zeros((1, 23)), "force": torch.zeros((1, 4)),
            }, {"trialname": torch.tensor(0)}, 0, 0)

    encode_calls = []

    class Wrapper:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(encode=self.encode)

        def _preprocess_torch(self, data):
            return data

        def encode(self, data):
            assert not torch.is_grad_enabled()
            assert data.shape[-1] == 50
            encode_calls.append(data.shape)
            return data[..., :24], data[..., :24]

    monkeypatch.setitem(sys.modules, "src.data.addBiomechanicsDataset", SimpleNamespace(
        AddBiomechanicsDataset=Dataset, trial_id_mapping={"walk": 0}, target_dof_names=dof_names,
        classify_trial=lambda trialname: "walk",
        ground_force_features=lambda force, names, mass: torch.zeros(4),
    ))
    monkeypatch.setitem(sys.modules, "src.vaemodel", SimpleNamespace(VAEModelWrapper=Wrapper))
    monkeypatch.setattr(module, "parse_export_args", lambda _: SimpleNamespace(data_path="unused", num_workers=0))
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "embed_latents", lambda values: np.zeros((len(values), 2)))
    if hasattr(module, "TARGET_TOTAL"):
        monkeypatch.setattr(module, "TARGET_TOTAL", 2)
    module.main()
    outputs = list((tmp_path / "result").glob("*.json"))
    assert len(outputs) == 1
    points = json.loads(outputs[0].read_text())
    assert len(points) == (3 if name.endswith("export_latent_embedding") else 2)
    assert encode_calls
    if name.endswith("sample_full_dofs_thumbnails"):
        assert len(encode_calls) == 1
        assert encode_calls[0][0] == len(points)
