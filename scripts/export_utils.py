"""Common command-line and embedding helpers for latent-space exports."""

import argparse
import os

import numpy as np


def parse_export_args(description: str) -> argparse.Namespace:
    """Read the dataset location and portable DataLoader worker count."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-path",
        default=os.environ.get("ADDBIOMECHANICS_PATH", "/home/public/data/AddBiomechanicsDataset/train/With_Arm/"),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def embed_latents(latents: np.ndarray) -> np.ndarray:
    """Embed at least two latent vectors, including small validation datasets."""
    from sklearn.manifold import TSNE

    latents = np.asarray(latents)
    if latents.ndim != 2 or latents.shape[0] < 2:
        raise ValueError("At least two latent vectors are required for t-SNE")
    return TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=min(30, latents.shape[0] - 1),
    ).fit_transform(latents)
