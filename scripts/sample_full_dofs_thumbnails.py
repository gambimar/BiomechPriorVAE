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
from pathlib import Path
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_utils import parse_export_args, embed_latents

NUM_DOFS = 50
LATENT_DIM = 24
STATE_KEYS = ["pos", "vel", "force"]
TARGET_TOTAL = 1500
CAP_PER_LABEL = 350
MAX_WINDOWS_SCANNED = 400000

@torch.inference_mode()
def main() -> None:
    """Export samples using paths relative to the repository."""
    args = parse_export_args(__doc__)
    torch.manual_seed(0)
    from src.data.addBiomechanicsDataset import (
        AddBiomechanicsDataset, classify_trial, ground_force_features,
        trial_id_mapping, target_dof_names,
    )
    from src.vaemodel import VAEModelWrapper

    dataset = AddBiomechanicsDataset(
        args.data_path,
        1,
        '',
        testing_with_short_dataset=False,
        skip_loading_skeletons=True,
    )

    full_dof_names = None
    for subj in dataset.subjects:
        if subj is not None:
            skel = subj.readSkel(0, '')
            full_dof_names = [skel.getDofByIndex(j).getName() for j in range(skel.getNumDofs())]
            break
    if full_dof_names is None:
        raise ValueError("No valid subjects were available for export")
    print(f"Full DOF set ({len(full_dof_names)}): {full_dof_names}")
    target_dof_indices = [full_dof_names.index(n) for n in target_dof_names]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = VAEModelWrapper(
        REPO_ROOT / f"result/model/BiomechPriorVAE_best_{NUM_DOFS}.pth",
        REPO_ROOT / f"result/model/scaler_{NUM_DOFS}.pkl",
        num_dofs=NUM_DOFS,
        latent_dim=LATENT_DIM,
        device=device,
    )

    counts = {k: 0 for k in trial_id_mapping}
    collected_inputs, collected_label, collected_full_pos = [], [], []

    window_order = list(range(len(dataset.windows)))
    random.Random(0).shuffle(window_order)

    scanned = 0
    for w_idx in tqdm(window_order):
        if len(collected_inputs) >= TARGET_TOTAL or scanned >= MAX_WINDOWS_SCANNED:
            break
        scanned += 1
        subject_idx, trial, window_start = dataset.windows[w_idx]
        subject = dataset.subjects[subject_idx]
        if subject is None:
            continue

        lab = classify_trial(subject.getTrialName(trial))

        if counts[lab] >= CAP_PER_LABEL:
            continue

        frames = subject.readFrames(trial, window_start, 1, stride=1,
                                     includeSensorData=False, includeProcessingPasses=True)
        if len(frames) != 1:
            continue
        fp = frames[0].processingPasses[-1]
        full_pos = np.asarray(fp.pos, dtype=np.float32)
        full_vel = np.asarray(fp.vel, dtype=np.float32)
        force4 = ground_force_features(
            torch.as_tensor(fp.groundContactForce, dtype=torch.float32),
            subject.getGroundForceBodies(),
            subject.getMassKg(),
        ).numpy()

        q = full_pos[target_dof_indices]
        qd = full_vel[target_dof_indices]
        counts[lab] += 1
        collected_inputs.append(np.concatenate([q, qd, force4]))
        collected_label.append(lab)
        collected_full_pos.append({name: float(full_pos[i]) for i, name in enumerate(full_dof_names)})

    print("Collected counts:", counts, "total:", len(collected_inputs))

    if len(collected_inputs) < 2:
        raise ValueError("At least two latent vectors are required for t-SNE")
    inputs = torch.as_tensor(np.stack(collected_inputs), dtype=torch.float32, device=device)
    mu, _ = vae_model.model.encode(vae_model._preprocess_torch(inputs))
    z = mu.cpu().numpy()
    z2 = embed_latents(z)

    points = []
    for i in range(len(collected_label)):
        points.append({
            "x": float(z2[i, 0]),
            "y": float(z2[i, 1]),
            "label": collected_label[i],
            "full_pos": collected_full_pos[i],
        })

    out_path = REPO_ROOT / "result/latent_thumbnail_samples_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(points, f)
    print(f"Wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()
