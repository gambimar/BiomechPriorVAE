"""Select ~500 stratified samples (by trial_id_mapping category) from the
AddBiomechanics dataset for the website's latent-space hover thumbnails.
Saves each sample's t-SNE (x,y), label, and raw joint-angle subset needed to
pose a SKEL mesh (hip/knee/ankle/lumbar only -- SKEL's arm DOF parametrization
doesn't map 1:1 onto AddBiomechanics' arm_flex/add/rot, so arms are left
neutral, matching how figure01's own icons already do it).
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

from src.data.addBiomechanicsDataset import AddBiomechanicsDataset, trial_id_mapping, target_dof_names
from src.vaemodel import VAEModelWrapper

NUM_DOFS = 50
LATENT_DIM = 24
STATE_KEYS = ["pos", "vel", "force"]
BATCH_SIZE = 32
TARGET_TOTAL = 1500
CAP_PER_LABEL = 350  # common categories capped here; rare ones just take all found
MAX_BATCHES = 12000

# Only these map 1:1 (or via rename) onto SKEL's pose_param_names.
POSE_DOFS = [
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
]
POSE_DOF_IDX = [target_dof_names.index(d) for d in POSE_DOFS]

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

inv_map = {v: k for k, v in trial_id_mapping.items()}
counts = {k: 0 for k in trial_id_mapping}
collected_mu, collected_label, collected_pose = [], [], []

for batch_idx, (data, label_, seq_len, _) in enumerate(tqdm(dataloader)):
    labels_np = label_['trialname'].numpy().astype(int).squeeze(-1) if label_['trialname'].ndim > 1 else label_['trialname'].numpy().astype(int)
    pos_np = data['pos'][:, 0, :].numpy()  # (B, 23), native AddBiomechanics convention

    data_cat = torch.concat([data[key] for key in STATE_KEYS], dim=-1).to(device)
    data_pre = vae_model._preprocess_torch(data_cat)
    mu_, _ = vae_model.model.encode(data_pre)
    mu_np = mu_.detach().cpu().numpy()
    if mu_np.ndim == 3:
        mu_np = mu_np[:, 0, :]

    for i in range(len(labels_np)):
        lab = inv_map[int(labels_np[i])]
        if counts[lab] >= CAP_PER_LABEL:
            continue
        counts[lab] += 1
        collected_mu.append(mu_np[i])
        collected_label.append(lab)
        collected_pose.append(pos_np[i, POSE_DOF_IDX])

    if len(collected_mu) >= TARGET_TOTAL or batch_idx >= MAX_BATCHES:
        break

print("Collected counts:", counts, "total:", len(collected_mu))

z = np.array(collected_mu)
reducer = TSNE(n_components=2, init='pca', random_state=0, perplexity=min(30, max(5, len(z) // 4)))
z2 = reducer.fit_transform(z)

points = []
for i in range(len(collected_label)):
    points.append({
        "x": float(z2[i, 0]),
        "y": float(z2[i, 1]),
        "label": collected_label[i],
        "pose": {name: float(val) for name, val in zip(POSE_DOFS, collected_pose[i])},
    })

out_path = "result/latent_thumbnail_samples.json"
with open(out_path, "w") as f:
    json.dump(points, f)
print(f"Wrote {len(points)} points to {out_path}")
