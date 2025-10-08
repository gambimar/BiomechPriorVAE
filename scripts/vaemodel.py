import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

#VAE network configuration    
class BiomechPriorVAE(nn.Module):
    def __init__(self, num_dofs, latent_dim=32, hidden_dim=512):
        super(BiomechPriorVAE, self).__init__()

        self.num_dofs = num_dofs
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_dofs),
            nn.Linear(num_dofs, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_dofs)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, x):

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)

        # The wrapper is only used for inference, thus we make it deterministic instead of random.
        z = mu
        rec_x = self.decode(z)

        return rec_x, mu, logvar

class VAEModelWrapper:
    def __init__(self, model_path, scaler_path, num_dofs=27, latent_dim=20, hidden_dim=512, device='cpu'):
        self.device = device
        self.num_dofs = num_dofs

        self.model = BiomechPriorVAE(
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Successfully loaded VAE model from: {model_path}")

        self.scaler = None
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Successfully loaded scaler from: {scaler_path}")

        if hasattr(self.scaler, 'n_features_in_'):
            print(f"Scaler expects {self.scaler.n_features_in_} features")
            if self.scaler.n_features_in_ != num_dofs:
                raise ValueError(f"Scaler expects {self.scaler.n_features_in_} features, but VAE is configured for {num_dofs}")

    #Process data using scaler
    def _preprocess(self, joint_angles_subset):
        if len(joint_angles_subset) != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} joint angles, got {len(joint_angles_subset)}")
        if self.scaler is not None:
            joint_angles_scaled = self.scaler.transform(joint_angles_subset.reshape(1, -1)).flatten()
            return joint_angles_scaled
        else:
            raise ValueError("Scaler is not loaded")
        
    def _preprocess_torch(self, joint_angles_subset_tensor):
        if self.scaler is not None:
            mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
            scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)
            
            joint_angles_scaled = (joint_angles_subset_tensor - mean) / scale
            return joint_angles_scaled
        else:
            raise ValueError("Scaler is not loaded")

    def _postprocess(self, joint_angles_subset):
        if self.scaler is not None:
            joint_angles_unscaled = self.scaler.inverse_transform(joint_angles_subset.reshape(1, -1)).flatten()
            return joint_angles_unscaled
        else:
            raise ValueError("Scaler is not loaded")

    def _postprocess_torch(self, joint_angles_subset_tensor):
        if self.scaler is not None:
            mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
            scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)
            
            joint_angles_unscaled = joint_angles_subset_tensor * scale + mean
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
    def reconstruct(self, joint_angles):
        joint_angles = torch.tensor(joint_angles, dtype=torch.float32, device=self.device)

        if len(joint_angles) == 33:
            original_pelvis = joint_angles[:6]
            subset_joints = joint_angles[6:]
        elif len(joint_angles) == 27:
            subset_joints = joint_angles
        else:
            raise ValueError("Input should be 33-or-27-dimensional")
        processed_angles = self._preprocess_torch(subset_joints)

        x = torch.tensor(processed_angles, dtype=torch.float32, device=self.device).unsqueeze(0)
        rec_x, mu, logvar = self.model.forward(x)
        rec_x_np = rec_x.squeeze(0)
        recon_subset = self._postprocess_torch(rec_x_np)
        # output = self._reconstruct_full_joints(recon_subset, original_pelvis)
        error = torch.norm(recon_subset - subset_joints)**2
        return error.cpu().detach().numpy()

    #Reconstruct the joint angle (with gradient w.r.t original scale), return the gradient
    def reconstruct_withgrad(self, joint_angles):
        joint_angles = torch.tensor(joint_angles, dtype=torch.float32, device=self.device, requires_grad=True)
        if joint_angles.grad is not None:
            joint_angles.grad.zero_()
        if len(joint_angles) == 33:
            original_pelvis = joint_angles[:6]
            subset_joints = joint_angles[6:]
        elif len(joint_angles) == 27:
            subset_joints = joint_angles
        else:
            raise ValueError("Input should be 33-or-27-dimensional")
        
        #===Adjust===
        processed_angles = self._preprocess_torch(subset_joints)
        x_batch = processed_angles.unsqueeze(0)
        rec_x_batch, mu, logvar = self.model.forward(x_batch)
        rec_x = rec_x_batch.squeeze(0)
        recon_subset_tensor = self._postprocess_torch(rec_x)
        error = torch.norm(recon_subset_tensor - subset_joints)**2
        error.backward()
        #===Adjust===


        # subset_joints_np = subset_joints.detach().cpu().numpy()
        # processed_angles = self._preprocess(subset_joints_np)
        # x = torch.tensor(processed_angles, dtype=torch.float32, device=self.device, requires_grad=True)
            
        # x_batch = x.unsqueeze(0)
        # rec_x_batch, mu, logvar = self.model.forward(x_batch)
        # rec_x = rec_x_batch.squeeze(0)
        # recon_subset_np = self._postprocess(rec_x.detach().cpu().numpy())
        # recon_subset_tensor = torch.tensor(recon_subset_np, dtype=torch.float32, device=self.device)
        
        # error = torch.norm(recon_subset_tensor - subset_joints)**2
        # error.backward()
        return joint_angles.grad.detach().cpu().numpy()

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
    
    joint_angles = joint_angles_np.flatten()
    error = _vae_instance.reconstruct(joint_angles)

    return error

def reconstruct_withgrad(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = joint_angles_np.flatten()
    gradient = _vae_instance.reconstruct_withgrad(joint_angles)

    return gradient
