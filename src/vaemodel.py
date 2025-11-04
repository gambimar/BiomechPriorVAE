import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
from src.vaetrainer import BiomechPriorVAE



class VAEModelWrapper:
    def __init__(self, model_path, scaler_path, num_dofs=27, latent_dim=20, hidden_dim=512, device='cpu'):
        self.device = device
        self.num_dofs = num_dofs
        self.mask = torch.zeros(102, dtype=torch.bool, device=device)
        if num_dofs == 27:
            self.mask[:27] = True
        elif num_dofs == 54:
            self.mask[:54] = True
        else:
            raise ValueError(f"Unsupported num_dofs: {num_dofs}, only 27 or 54 are supported.")

        self.model = BiomechPriorVAE(
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded VAE model from: {model_path}")

        # Load scaler dictionary with mean_ and scale_ tensors
        self.scaler = None
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Successfully loaded scaler from: {scaler_path}")

        # Check if scaler is a dict with mean_ and scale_
        if isinstance(self.scaler, dict):
            if 'mean_' in self.scaler and 'scale_' in self.scaler:
                # Convert to device if needed
                if isinstance(self.scaler['mean_'], torch.Tensor):
                    self.scaler['mean_'] = self.scaler['mean_'].to(device)
                    self.scaler['scale_'] = self.scaler['scale_'].to(device)
                else:
                    # Convert numpy arrays to tensors
                    self.scaler['mean_'] = torch.tensor(self.scaler['mean_'], dtype=torch.float32, device=device)
                    self.scaler['scale_'] = torch.tensor(self.scaler['scale_'], dtype=torch.float32, device=device)
                print(f"Scaler expects {len(self.scaler['mean_'])} features")
                if len(self.scaler['mean_']) != num_dofs:
                    raise ValueError(f"Scaler expects {len(self.scaler['mean_'])} features, but VAE is configured for {num_dofs}")
            else:
                raise ValueError("Scaler dict must contain 'mean_' and 'scale_' keys")
        else:
            raise ValueError("Scaler must be a dictionary with 'mean_' and 'scale_' keys")

    #Process data using scaler
    def _preprocess(self, joint_angles_subset):
        if len(joint_angles_subset) != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} joint angles, got {len(joint_angles_subset)}")
        if self.scaler is not None:
            joint_angles_tensor = torch.tensor(joint_angles_subset, dtype=torch.float32, device=self.device)
            joint_angles_scaled = (joint_angles_tensor - self.scaler['mean_']) / self.scaler['scale_']
            return joint_angles_scaled.cpu().numpy()
        else:
            raise ValueError("Scaler is not loaded")
        
    def _preprocess_torch(self, joint_angles_subset_tensor):
        if self.scaler is not None:
            joint_angles_scaled = (joint_angles_subset_tensor - self.scaler['mean_']) / self.scaler['scale_']
            return joint_angles_scaled
        else:
            raise ValueError("Scaler is not loaded")

    def _postprocess(self, joint_angles_subset):
        if self.scaler is not None:
            joint_angles_tensor = torch.tensor(joint_angles_subset, dtype=torch.float32, device=self.device)
            joint_angles_unscaled = joint_angles_tensor * self.scaler['scale_'] + self.scaler['mean_']
            return joint_angles_unscaled.cpu().numpy()
        else:
            raise ValueError("Scaler is not loaded")

    def _postprocess_torch(self, joint_angles_subset_tensor):
        if self.scaler is not None:
            joint_angles_unscaled = joint_angles_subset_tensor * self.scaler['scale_'] + self.scaler['mean_']
            return joint_angles_unscaled
        else:
            raise ValueError("Scaler is not loaded")

    def _extract_joint_subset(self, full_joints):
        if len(full_joints) == 33:
            return full_joints[6:]
        elif len(full_joints) == 27:
            return full_joints
        else:
            raise ValueError(f"Unexpected joint dimension: {len(full_joints)}")

    def _reconstruct_full_joints(self, subset_joints, original_pelvis):
        full_joints = np.zeros(33)
        full_joints[:6] = original_pelvis
        full_joints[6:] = subset_joints
        return full_joints

    #Reconstruct the joint angle (without gradient), return the reconstruction error
    def reconstruct(self, joint_angles, getgrad=False):
        ndim = len(joint_angles.shape) 
        # Convert to tensor if numpy array
        if getgrad:
            joint_angles = torch.from_numpy(joint_angles).to(dtype=torch.float32).to(self.device)
            joint_angles.requires_grad_(True)
        else:
            joint_angles = torch.from_numpy(joint_angles).to(dtype=torch.float32).to(self.device)

        if ndim == 1:
            subset_joints = joint_angles.unsqueeze(0)  # Add batch dimension
        else:
            subset_joints = joint_angles
        # Set mtp and subtalar to zero for foot joints
        scaling = torch.ones_like(subset_joints)
        scaling[:,[5,6,12,13]] = 0.0
        scaling[:,[3,10]] = -1 # Knee inverted
        if len(joint_angles) > 50:
            scaling[:,[5+27,6+27,12+27,13+27]] = 0.0
            scaling[:,[3+27,10+27]] = -1

        subset_joints = subset_joints * scaling
        subset_joints = subset_joints[:,self.mask]
        processed_angles = self._preprocess_torch(subset_joints)

        x = processed_angles
        mu, logvar = self.model.encode(x)
        mu = self.model.decode(mu)
        rec_x_np = mu.squeeze(0)
        recon_subset = self._postprocess_torch(rec_x_np)
        # output = self._reconstruct_full_joints(recon_subset, original_pelvis)
        error = torch.norm(subset_joints - recon_subset)**2
        if not getgrad:
            return error.detach().cpu().numpy()
        else: 
            error.backward(retain_graph=False)
            return joint_angles.grad.detach().cpu().numpy()

    #Reconstruct the joint angle (with gradient w.r.t original scale), return the gradient
    def reconstruct_withgrad(self, joint_angles):
        return self.reconstruct(joint_angles, getgrad=True).flatten()


#Initialize a instance to get the result
_vae_instance = None

def initialize_vae(model_path, scaler_path, num_dofs=27, latent_dim=20, hidden_dim=512, device='cpu'):
    global _vae_instance
    try:
        _vae_instance = VAEModelWrapper(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        return True
    except Exception as e:
        print(f"Failed to initialize VAE: {e}")
        return False

def reconstruct(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = joint_angles_np.copy().T
    error = _vae_instance.reconstruct(joint_angles)

    return error

def reconstruct_withgrad(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = joint_angles_np.copy().T
    gradient = _vae_instance.reconstruct_withgrad(joint_angles)

    return gradient
