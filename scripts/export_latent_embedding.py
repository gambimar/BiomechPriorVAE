"""One-off export of a 2D t-SNE embedding of the BiomechPriorVAE latent space,
colored by AddBiomechanics trial-type label, for the project website's
Latent-space panel. Mirrors notebook/latent_space_analysis.ipynb cells 6 + 12,
but against the num_dofs=50 ("q_dot_F") checkpoint that matches
plot/figure00.py's published architecture, and writes plain JSON instead of a
matplotlib figure. Does not modify any existing file in the repo.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_utils import parse_export_args, embed_latents

NUM_DOFS = 50
LATENT_DIM = 24
STATE_KEYS = ["pos", "vel", "force"]
MAX_BATCHES = 300
BATCH_SIZE = 32

@torch.inference_mode()
def main() -> None:
    """Export samples using paths relative to the repository."""
    args = parse_export_args(__doc__)
    torch.manual_seed(0)
    from src.data.addBiomechanicsDataset import AddBiomechanicsDataset, trial_id_mapping
    from src.vaemodel import VAEModelWrapper

    dataset = AddBiomechanicsDataset(
        args.data_path,
        1,
        '',
        testing_with_short_dataset=False,
        skip_loading_skeletons=True,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = VAEModelWrapper(
        REPO_ROOT / f"result/model/BiomechPriorVAE_best_{NUM_DOFS}.pth",
        REPO_ROOT / f"result/model/scaler_{NUM_DOFS}.pkl",
        num_dofs=NUM_DOFS,
        latent_dim=LATENT_DIM,
        device=device,
    )

    mu_list, label_list = [], []
    for batch_idx, (data, label_, seq_len, _) in enumerate(tqdm(dataloader)):
        if batch_idx >= MAX_BATCHES:
            break
        data_cat = torch.concat([data[key] for key in STATE_KEYS], dim=-1).to(device)
        data_pre = vae_model._preprocess_torch(data_cat)
        mu_, _ = vae_model.model.encode(data_pre)
        mu_list.append(mu_.detach().cpu())
        label_list.append(label_['trialname'].detach().cpu())

    if not mu_list:
        raise ValueError('No valid dataset frames were available for export')
    z = torch.cat(mu_list, dim=0).numpy().reshape(-1, LATENT_DIM)
    labels_lin = torch.cat(label_list, dim=0).numpy().astype(int).reshape(-1)

    inv_map = {v: k for k, v in trial_id_mapping.items()}
    label_names = [inv_map[int(i)] for i in labels_lin]

    print(f"Embedding {z.shape[0]} points, latent dim {z.shape[1]}")
    print("Label counts:", {k: int((np.array(label_names) == k).sum()) for k in trial_id_mapping})

    z2 = embed_latents(z)

    points = [
        {"x": float(z2[i, 0]), "y": float(z2[i, 1]), "label": label_names[i]}
        for i in range(z2.shape[0])
    ]

    out_path = REPO_ROOT / "result/latent_embedding_export.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(points, f)
    print(f"Wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()
