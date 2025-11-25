# BiomechPriorVAE
A biomechanics prior based VAE project.

## Overview
The project focuses on generate more reasonable human lower body posture with pretrained VAE model based on open-source musculoskeletal motion dataset, as a biomechanics prior, provides guidance for optimal control for gait generation.

## Structure
### 1.scripts
Contains the main python scripts for data convertion and model training, as well as the interface for Matlab use.

- **`b3dconverter.py`** - Convert the .b3d files into python-friendly numpy array.
- **`datasetvisualize.py`** - Visualize the musculoskeletal model with specific posture using nimblephysics default 'Rajagopal' model. 
- **`vaetrainer.py`** - VAE model trainer.
- **`vaemodel.py`** - Interface for using VAE model in Matlab

### 2.result
Contains the VAE model trained using PyTorch and saved StandardScaler.

## Notes
The dataset used in the project comes from AddBiomechanics dataset, which uses the default **'Rajagopal' model (37 DOFs)**, but for the optimal control, the model we used is **'gait3d_pelvis213' (33 DOFs)**, there are four wrist dofs are missing, thus the input dimension of VAE model is 33 instead of 37, and four wrist DOFs would be automatically filled by 0.0 when visualization is needed. **So please note whether the DOFs of musculoskeletal model match the VAE model**.


### 3. Things to try
VAE state spaces:
- q
- qdot
- Fy
- ||Fx,Fz||/Fy

or any of their combinations.

For mixed tasks, scaling between the categories may be helpful