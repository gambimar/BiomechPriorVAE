# BiomechPriorVAE
Using priors with Predictive Movement Simulations. Official method implementation for "Closing the realism gap in physics-based gait simulations with a learned state prior" (preprint: [arXiv:2610.08506](https://doi.org/10.48550/arXiv.2610.08506)).

**Authors:** Markus Gambietz, Zhihao Zhao, Theodoros Balougias, Xiang Wang, Anne D. Koelewijn

## Setup
Developed on Linux with MATLAB and Python 3.12. The expected folder layout is:
```
├── BiomechPriorVAE/            # this repository (folder name must stay "BiomechPriorVAE")
│   └── BioMAC-Sim-Toolbox/     # cloned inside this repository
└── mexIPOPT/                   # next to this repository
```

### 1. Python environment
Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`):
```bash
cd BiomechPriorVAE
uv sync            # creates .venv/ with Python 3.12, torch, nimblephysics, ...
```
MATLAB starts this interpreter out-of-process: `vaeReconstructionTerm.m` hard-codes `.../BiomechPriorVAE/.venv/bin/python3.12`. Adapt that path if your checkout is elsewhere, or if using different Python versions. MATLAB is quite restrictive when it comes to Python versions, so the easiest way is to use the same versions as we did (MATLAB R2024b, Python 3.12).

### 2. BioMAC-Sim-Toolbox
The simulations use the `opensimContactModel` branch of the [BioMAC-Sim-Toolbox](https://github.com/mad-lab-fau/BioMAC-Sim-Toolbox), which provides the smooth-sphere contact model (`Gait3d_smoothsphere`) and a logarithmic `cotTerm` with selectable metabolic models.
```bash
cd BiomechPriorVAE
git clone -b opensimContactModel https://github.com/mad-lab-fau/BioMAC-Sim-Toolbox.git
```
In MATLAB, add the toolbox to the path (see the toolbox README for details):
```matlab
cd BioMAC-Sim-Toolbox
addpath(genpath('src'));
savepath;
```
Models are compiled automatically the first time they are used; a C compiler for MATLAB `mex` is required. OpenSim is optional: the `.mat` files next to each `.osim` model (in `data/model/` and the toolbox's `osim_files/`) contain the precomputed model data and moment arms.

### 3. IPOPT
Install [mexIPOPT](https://github.com/ebertolazzi/mexIPOPT) next to this repository. On Linux with gcc > 9 use the fork [gambimar/mexIPOPT](https://github.com/gambimar/mexIPOPT/).
```bash
cd ..   # parent folder of BiomechPriorVAE
git clone https://github.com/gambimar/mexIPOPT.git
```

### 5. Trained VAE weights
We have added the trained weights for the kinematics + velocities + ground reaction forces condition (`nDimVae = 50`) to the repository. If you want to train your own VAE, use the `src/vaetrainer.py` script with the AddBiomechanics dataset.
Run it as a module from the repository root (it imports `src.*`, so `python src/vaetrainer.py` fails with `ModuleNotFoundError: No module named 'src'`):
```bash
uv run python -m src.vaetrainer --addbiomechanics_path /path/to/AddBiomechanicsDataset/train/With_Arm/
```
The state space is selected in the `__main__` block via `model = ...`. The default, `"q_qdot"`, trains a 46-dimensional prior; set `model = "q_dot_F"` to train the 50-dimensional one used in the simulations. The weights and scaler are written to `result/model/BiomechPriorVAE_best_<n>.pth` and `result/model/scaler_<n>.pkl`, where `<n>` is the input dimension, which are the file names `runSim.m` looks up via `nDimVae`. The latent size is fixed to 24 in the `train_model(...)` call.

### 6. MATLAB paths
`runGaitSingleStep.m` adds the repository (including the nested toolbox) and `mexIPOPT` with `addpath(genpath(...))`, using hard-coded absolute paths. Adapt them, together with `base_result_path`, before running simulations.

## Overview
The project focuses on generate more reasonable human gait simulations by incorporating a learned state prior (VAE) into the optimal control problem. 

## Structure
### 1. src
Contains the main python scripts for data convertion and model training, as well as the interface for Matlab use.

- **`src/vaetrainer.py`** - Train VAE model based on the addBiomechanics dataset. (This is actually the training script).
- **`src/data/addBiomechanicsDataset.py`** - Adapted from nimblephysics' example, used to load the AddBiomechanics dataset - provides joint angles, joint velocities, ground reaction forces and / or torques.
- **`src/vaemodel.py`** - Interface for using VAE model in Matlab. This script includes a reconstruction term to be used in the optimal control problem in BioMAC-Sim-Toolbox. It also accounts for modelling differences (e.g. locked joints / sign conventions) between the dataset and the musculoskeletal model used in BioMAC-Sim-Toolbox.

### 2. scripts
- **`scripts/script3D.m`** - Adapted example script from the BioMAC-Sim-Toolbox, which shows how to use the VAE prior in an optimal control problem for 3D gait generation.
- **`scripts/func/running3D.m`** and **`scripts/func/standing3D.m`** - Set up optimal control problems for gait (walking/running) and standing tasks, respectively.
- **`scripts/func/runSim.m`** - Runs one predictive simulation (standing initial guess, then gait at a target speed with the VAE prior); called by the `runGait` interface below. `runSimNoVae.m` is the same without the VAE prior.

### 3. utility
- **`vaeReconstructionTerm.m`** (repo root) - The reconstruction objective for the BioMAC-Sim-Toolbox, which calls the VAE model (`src/vaemodel.py`) through MATLAB's Python interface to compute the reconstruction error and its gradient. This function is here for reference, the actual implementation used in the simulations is copied to the toolbox's `Collocation` class. 

## Running gait simulations (`runGait` interface)
Batches of predictive gait simulations are started from the shell with **`runGait.sh`** (repo root). For each speed in the requested range it repeatedly calls MATLAB (`scripts/func/runGaitSingleStep.m` → `scripts/func/runSim.m`) until **10 converged solutions** exist for that speed.

### Usage
Run from the repository root (the script adds `scripts/func` via a relative path):
```bash
./runGait.sh --model=<model.osim> [--metmodel=<name|0>] [--min=SPEED] [--max=SPEED]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model=` | yes | – | OpenSim model file, e.g. `gait3d_pelvis213.osim` or any `.osim` in `data/model/`. Models whose name contains `smoothsphere` (and its contact variants `softdamp2/4`, `stiffer2/4`, `nodamp`, `stiff10x`) automatically use the matching smooth-sphere contact model class. |
| `--metmodel=` | no | `0` | Effort / energy term. `0`: muscle activation + torque effort (VAE weight 3). A metabolic model name (`umberger`, `bhargava`, `lichtwark`, `margaria`, `houdijk`, `minetti`): cost of transport with that model (VAE weight 1). `none`: no effort term, only the VAE prior. |
| `--min=` | no | `0.73` | Lowest speed (m/s) to simulate. |
| `--max=` | no | `1.63` | Highest speed (m/s) to simulate. |

Only speeds on a fixed grid are run; `--min`/`--max` select the grid points inside the range:
- `0.00` – free speed (speed is an optimisation variable bounded to 0.2–1.8 m/s)
- `0.73` – `3.53` in steps of 0.1
- `3.73` – `5.63` in steps of 0.2
- `6.00`, `7.00`

### Examples
```bash
# Default effort objective, generic model, walking speeds 0.73–1.63 m/s
./runGait.sh --model=gait3d_pelvis213.osim

# Houdijk metabolic cost, SIPP model, walking speeds
./runGait.sh --model=sipp_generic_runmad.osim --metmodel=houdijk --min=0.73 --max=1.63

# Umberger metabolic cost, smooth-sphere contact, running speeds 2.53–5.63 m/s
./runGait.sh --model=sipp_generic_runmad_smoothsphere.osim --metmodel=umberger --min=2.53 --max=5.63

# A single speed: set min == max
./runGait.sh --model=gait3d_pelvis213.osim --metmodel=minetti --min=1.33 --max=1.33

# Free-speed simulation only
./runGait.sh --model=gait3d_pelvis213.osim --metmodel=houdijk --min=0 --max=0

# Long batches: keep running after logout and log the output
nohup ./runGait.sh --model=gait3d_pelvis213.osim --metmodel=houdijk --min=0.73 --max=3.53 > runGait_houdijk.log 2>&1 &
```

### Output
Results are written to `result/simulations/` as
```
<prefix><iter>_<speed>.mat
```
where `<prefix>` is the model name without `.osim`, plus `_<metmodel>` if one is given, and `<iter>` is the run index, also used as random seed (`rng(iter)`). For example, `--model=gait3d_pelvis213.osim --metmodel=houdijk` at 1.33 m/s produces `gait3d_pelvis213_houdijk1_1.33.mat`, `gait3d_pelvis213_houdijk2_1.33.mat`, … Each file holds a `result` object; `result.converged` tells whether IPOPT converged.

### How it works / resuming
- Each MATLAB call runs **one** simulation (the next missing `<iter>`) and exits, so MATLAB (and the Python VAE process) is restarted between runs to free memory.
- Exit code protocol of `runGaitSingleStep`: `0` = one simulation done, call again; `1` = 10 converged runs reached for this speed, move on to the next speed; anything else = error, the script aborts.
- Existing result files are skipped, so an interrupted batch can be restarted with the same command and continues where it stopped.

### (Deprectated) Variants
- **`runGait_vaeOff.sh`** – same arguments, but calls `runGaitSingleStepVaeOff.m` / `runSimNoVae.m` (no VAE prior). Files are named `<prefix>noVae<iter>_<speed>.mat`, seeds are `iter+30`, and the `0.00`, `6.00` and `7.00` speed bins are not included.
- **`runGait_houdijk.sh`**, **`runGait_umberger.sh`** – older wrappers with a fixed metabolic model (`--min`, `--max`, `--sipp`); superseded by `runGait.sh --metmodel=...`.
- **`scripts/runGait.m`** – old MATLAB-only batch script (all sections disabled with `break`).

### Before running
- `runGaitSingleStep.m` contains hard-coded absolute paths (repository, `mexIPOPT`, result folder); adapt them to your machine.
- `matlab` must be on your `PATH`, and the Python environment used by MATLAB's `pyenv` must be able to import the VAE (see `src/vaemodel.py`).

## Further analysis of the results

All further analysis code has been moved to the [BiomechPriorVAE-analysis](https://github.com/gambimar/BiomechPriorVAE-visuals) repository. It contains scripts to generate the figures in the preprint, as well as additional visualizations and analyses of the simulation results.
