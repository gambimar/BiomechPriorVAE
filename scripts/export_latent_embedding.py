"""One-off export of a 2D t-SNE embedding of the BiomechPriorVAE latent space,
colored by AddBiomechanics trial-type label, for the project website's
Latent-space panel. Mirrors notebook/latent_space_analysis.ipynb cells 6 + 12,
but against the num_dofs=50 ("q_dot_F") checkpoint that matches
plot/figure00.py's published architecture, and writes plain JSON instead of a
matplotlib figure. Does not modify any existing file in the repo.
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

while not os.path.exists('.git'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

from src.data.addBiomechanicsDataset import AddBiomechanicsDataset, trial_id_mapping
from src.vaemodel import VAEModelWrapper

NUM_DOFS = 50
LATENT_DIM = 24
STATE_KEYS = ["pos", "vel", "force"]
MAX_BATCHES = 300
BATCH_SIZE = 32

dataset = AddBiomechanicsDataset(
    "/home/public/data/AddBiomechanicsDataset/train/With_Arm/",
    1,
    '',
    testing_with_short_dataset=False,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae_model = VAEModelWrapper(
    f"result/model/BiomechPriorVAE_best_{NUM_DOFS}.pth",
    f"result/model/scaler_{NUM_DOFS}.pkl",
    num_dofs=NUM_DOFS,
    latent_dim=LATENT_DIM,
    device=device,
)

mu_list, label_list = [], []
for batch_idx, (data, label_, seq_len, _) in enumerate(tqdm(dataloader)):
    data_cat = torch.concat([data[key] for key in STATE_KEYS], dim=-1).to(device)
    data_pre = vae_model._preprocess_torch(data_cat)
    mu_, _ = vae_model.model.encode(data_pre)
    mu_list.append(mu_.detach().cpu())
    label_list.append(label_['trialname'].detach().cpu())
    if batch_idx >= MAX_BATCHES:
        break

z = np.array(torch.concat(mu_list, dim=0)).squeeze()
labels_lin = np.array(torch.concat(label_list, dim=0)).astype(int).squeeze()

inv_map = {v: k for k, v in trial_id_mapping.items()}
label_names = [inv_map[int(i)] for i in labels_lin]

print(f"Embedding {z.shape[0]} points, latent dim {z.shape[1]}")
print("Label counts:", {k: int((np.array(label_names) == k).sum()) for k in trial_id_mapping})

reducer = TSNE(n_components=2, init='pca', random_state=0, perplexity=30)
z2 = reducer.fit_transform(z)

points = [
    {"x": float(z2[i, 0]), "y": float(z2[i, 1]), "label": label_names[i]}
    for i in range(z2.shape[0])
]

out_path = "result/latent_embedding_export.json"
with open(out_path, "w") as f:
    json.dump(points, f)
print(f"Wrote {len(points)} points to {out_path}")
