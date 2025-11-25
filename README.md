# BiomechPriorVAE
Using priors with BioMAC-Sim-Toolbox. This readme is still under construction. 

## Overview
The project focuses on generate more reasonable human lower body posture with pretrained VAE model based on open-source musculoskeletal motion dataset, as a biomechanics prior, provides guidance for optimal control for gait generation.

## Structure
### 1. src
Contains the main python scripts for data convertion and model training, as well as the interface for Matlab use.

- **`src/vaetrainer.py`** - Train VAE model based on the addBiomechanics dataset. (This is actually the training script).
- **`src/data/addBiomechanicsDataset.py`** - Adapted from nimblephysics' example, used to load the AddBiomechanics dataset - provides joint angles, joint velocities, ground reaction forces and / or torques.
- **`src/vaemodel.py`** - Interface for using VAE model in Matlab. This script includes a reconstruction term to be used in the optimal control problem in BioMAC-Sim-Toolbox. It also accounts for modelling differences (e.g. locked joints / sign conventions) between the dataset and the musculoskeletal model used in BioMAC-Sim-Toolbox.

### 2. scripts
- **`scripts/script3D.m`** - Adapted example script from the BioMAC-Sim-Toolbox, which shows how to use the VAE prior in an optimal control problem for 3D gait generation.
- **`scripts/running3D.m`** and **`scripts/standing3D.m`** - Set up optimal control problems for running and standing tasks, respectively, used by `script3D.m`.

### 3. utility
- **`BioMAC-Sim-Toolbox/src/problem/@Collocation/vaeReconstructionTerm.m`** - The reconstruction objective in the BioMAC-Sim-Toolbox, which calls the VAE model to compute the reconstruction error. This file should be placed in the `BioMAC-Sim-Toolbox/src/problem/@Collocation/` directory.


## Notes
The dataset used in the project comes from AddBiomechanics dataset, this uses the default **'Rajagopal' model (37 DOFs)**, but for the optimal control, the model we used is **'gait3d_pelvis213' (33 DOFs)**, there are four wrist dofs are missing, thus the input dimension of VAE model is 33 instead of 37, and four wrist DOFs would be automatically filled by 0.0 when visualization is needed. **So please note whether the DOFs of musculoskeletal model match the VAE model**.


### 3. Things to try
VAE state spaces:
- q
- qdot
- Fy
- ||Fx,Fz||/Fy

or any of their combinations.

For mixed tasks, scaling between the categories may be helpful