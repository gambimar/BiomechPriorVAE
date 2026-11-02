"""Compare CPU inference against an existing Git revision without changing files."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vaemodel import VAEModelWrapper
from src.scaler import load_scaler


def median_latency(wrapper, frames: np.ndarray, gradient: bool, repeats: int) -> float:
    """Return median milliseconds per call after warming the model."""
    for _ in range(5):
        wrapper.reconstruct(frames, getgrad=gradient)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        wrapper.reconstruct(frames, getgrad=gradient)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main() -> None:
    """Benchmark the published model with a fixed synthetic 50-node batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='e0e9d3e')
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    source = subprocess.run(
        ['git', 'show', f'{args.baseline_ref}:src/vaemodel.py'], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    baseline = types.ModuleType('baseline_vaemodel')
    exec(compile(source, 'baseline_vaemodel.py', 'exec'), baseline.__dict__)
    torch.set_num_threads(1)
    paths = (ROOT / 'result/model/BiomechPriorVAE_best_50.pth', ROOT / 'result/model/scaler_50.pkl')
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        # Remap the bundled CUDA scaler so the historical loader also runs on CPU hosts.
        portable_scaler = Path(directory) / 'scaler.pkl'
        with portable_scaler.open('wb') as stream:
            pickle.dump(load_scaler(paths[1]), stream)
        current = VAEModelWrapper(paths[0], portable_scaler, num_dofs=50, latent_dim=24)
        previous = baseline.VAEModelWrapper(paths[0], portable_scaler, num_dofs=50, latent_dim=24)
    frames = np.random.default_rng(42).normal(0, 0.1, (50, 123)).astype(np.float32)
    np.testing.assert_allclose(current.reconstruct(frames), previous.reconstruct(frames), rtol=1e-6)
    result = {'device': 'cpu', 'torch_threads': 1, 'batch_size': 50, 'repeats': args.repeats}
    for gradient in (False, True):
        kind = 'gradient' if gradient else 'objective'
        old_ms = median_latency(previous, frames, gradient, args.repeats)
        new_ms = median_latency(current, frames, gradient, args.repeats)
        result[kind] = {'baseline_ms': old_ms, 'current_ms': new_ms, 'speedup': old_ms / new_ms}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
