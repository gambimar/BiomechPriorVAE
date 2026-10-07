"""Select ~1500 stratified samples (by trial_id_mapping category) from the
AddBiomechanics dataset for the website's latent-space hover thumbnails.

v2: unlike sample_stratified_thumbnails.py, this captures the FULL raw pos
vector (all 37 dof_names -- pelvis through wrist) per sampled frame, not just
the 13-DOF subset used for posing before. The VAE's 50-D encoder input
(q23+qdot23+F4) only ever used a curated subset (target_dof_names) that
omits pelvis, subtalar, mtp, and wrist entirely -- fine for training, but it
means "sample from AddB" was silently dropping real DOFs the render could
otherwise use. Reads frames directly (bypassing AddBiomechanicsDataset's
DataLoader path, which only exposes the target-dof subset and no
window_start), so both the full pos and the VAE's reduced input come from
the exact same frame.
"""
import json
import os
import random
import sys

import numpy as np
import torch
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
TARGET_TOTAL = 1500
CAP_PER_LABEL = 350
MAX_WINDOWS_SCANNED = 400000

dataset = AddBiomechanicsDataset(
    "/home/public/data/AddBiomechanicsDataset/train/With_Arm/",
    1,
    '',
    testing_with_short_dataset=False,
)

full_dof_names = None
for subj in dataset.subjects:
    if subj is not None:
        skel = subj.readSkel(0, '')
        full_dof_names = [skel.getDofByIndex(j).getName() for j in range(skel.getNumDofs())]
        break
print(f"Full DOF set ({len(full_dof_names)}): {full_dof_names}")
target_dof_indices = [full_dof_names.index(n) for n in target_dof_names]

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
collected_mu, collected_label, collected_full_pos = [], [], []

window_order = list(range(len(dataset.windows)))
random.seed(0)
random.shuffle(window_order)

scanned = 0
for w_idx in tqdm(window_order):
    if len(collected_mu) >= TARGET_TOTAL or scanned >= MAX_WINDOWS_SCANNED:
        break
    scanned += 1
    subject_idx, trial, window_start = dataset.windows[w_idx]
    subject = dataset.subjects[subject_idx]
    if subject is None:
        continue

    trialname = subject.getTrialName(trial)
    tl = trialname.lower()
    if tl.find('static') != -1:
        lab = 'static'
    elif tl.find('walk') != -1:
        lab = 'walk'
    elif tl.find('run') != -1:
        lab = 'run'
    elif tl.find('gait') != -1:
        lab = 'gait_any'
    elif tl.find('sit') != -1 and tl.find('stand') != -1:
        lab = 'sit to stand'
    elif tl.find('stair') != -1:
        lab = 'stair'
    elif tl.startswith('t'):
        lab = 'no_name'
    else:
        lab = 'other'

    if counts[lab] >= CAP_PER_LABEL:
        continue

    frames = subject.readFrames(trial, window_start, 1, stride=1,
                                 includeSensorData=False, includeProcessingPasses=True)
    if len(frames) != 1:
        continue
    fp = frames[0].processingPasses[-1]
    full_pos = np.asarray(fp.pos, dtype=np.float32)
    full_vel = np.asarray(fp.vel, dtype=np.float32)
    force_raw = np.asarray(fp.groundContactForce, dtype=np.float32)
    mass = subject.getMassKg()
    f = force_raw / mass / 9.81
    force4 = np.array([f[1], np.sqrt(f[0] ** 2 + f[2] ** 2), f[4], np.sqrt(f[3] ** 2 + f[5] ** 2)], dtype=np.float32)

    q = full_pos[target_dof_indices]
    qd = full_vel[target_dof_indices]
    x50 = np.concatenate([q, qd, force4])[None, None, :]  # (1,1,50)

    x50_t = torch.tensor(x50, dtype=torch.float32, device=device)
    x50_pre = vae_model._preprocess_torch(x50_t)
    mu_, _ = vae_model.model.encode(x50_pre)
    mu_np = mu_.detach().cpu().numpy().reshape(-1)

    counts[lab] += 1
    collected_mu.append(mu_np)
    collected_label.append(lab)
    collected_full_pos.append({name: float(full_pos[i]) for i, name in enumerate(full_dof_names)})

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
        "full_pos": collected_full_pos[i],
    })

out_path = "result/latent_thumbnail_samples_full.json"
with open(out_path, "w") as f:
    json.dump(points, f)
print(f"Wrote {len(points)} points to {out_path}")
