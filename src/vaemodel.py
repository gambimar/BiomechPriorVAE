import torch
import numpy as np
from src.scaler import load_scaler
from src.vaetrainer import BiomechPriorVAE



class VAEModelWrapper:
    def __init__(self, model_path, scaler_path, num_dofs=27, latent_dim=20, hidden_dim=512, device='cpu', state_keys=None):
        self.device = device
        self.num_dofs = num_dofs
        self.mask = torch.zeros(91, dtype=torch.bool, device=device)
        if state_keys is None and num_dofs in (23, 25, 46, 48, 69):
            state_keys = {
                23: ('pos',), 25: ('pos', 'M_ankle'),
                46: ('pos', 'vel'), 48: ('pos', 'vel', 'M_ankle'),
                69: ('pos', 'vel', 'tau'),
            }[num_dofs]
        self.feature_indices = None
        if state_keys is not None:
            pos = [i for i in range(27) if i not in (5, 6, 12, 13)]
            blocks = {
                'pos': pos,
                'vel': [27 + i for i in pos],
                'force': list(range(54, 58)),
                'M_ankle': [68, 75],
                'tau': [64 + i for i in pos],
            }
            if not state_keys or len(set(state_keys)) != len(state_keys) or any(key not in blocks for key in state_keys):
                raise ValueError('state_keys must be unique supported feature groups')
            indices = [i for key in state_keys for i in blocks[key]]
            if len(indices) != num_dofs:
                raise ValueError(f'state_keys select {len(indices)} features, expected {num_dofs}')
            self.feature_indices = torch.tensor(indices, device=device)
            self.mask[self.feature_indices] = True
            self.result_scale = torch.ones(num_dofs, dtype=torch.float32, device=device)
        elif num_dofs == 27:
            self.mask[:27] = True
            self.result_scale = torch.ones(27, dtype=torch.float32, device=device)
        elif num_dofs == 54: #Here q and M
            self.mask[:27] = True
            self.mask[-27:] = True
            self.result_scale = torch.ones(54, dtype=torch.float32, device=device)
            self.result_scale[-27:] = 10 # scaling for torque dimensions
        elif num_dofs == 31:
            self.mask[:27] = True
            self.mask[54:58] = True
            self.result_scale = torch.ones(31, dtype=torch.float32, device=device)
            self.result_scale[-4:] = 27/4 # Scale force dimensions so that they weigh equally
        elif num_dofs == 58:
            self.mask[:54] = True
            self.mask[54:58] = True
            self.result_scale = torch.ones(58, dtype=torch.float32, device=device)
            self.result_scale[-4:] = 27/4 # Scale force dimensions so that they weigh equally
        elif num_dofs == 50: # Q, Qdot and F
            self.mask[:27] = True
            self.mask[[5,6,12,13]] = False # zero out mtp and subtalar for foot joints
            self.mask[27:54] = True
            self.mask[[5+27,6+27,12+27,13+27]] = False # zero out mtp and subtalar for foot joints
            self.mask[54:54+4] = True # forces
            self.result_scale = torch.ones(50, dtype=torch.float32, device=device)
            self.result_scale[-4:] = 27/4 # Scale force dimensions so that they weigh equally
        elif num_dofs == 52: # Q, Qdot, F and M_ankle
            self.mask[:27] = True
            self.mask[[5,6,12,13]] = False # zero out mtp and subtalar for foot joints
            self.mask[27:54] = True
            self.mask[[5+27,6+27,12+27,13+27]] = False # zero out mtp and subtalar for foot joints
            self.mask[54:54+4] = True # forces
            self.mask[54+4+10] = True # right ankle
            self.mask[54+4+17] = True # left ankle
            self.result_scale = torch.ones(52, dtype=torch.float32, device=device)
            #self.result_scale[:23] = 1/0.299 # scaling for velocity dimensions
            #self.result_scale[23:46] = 1/2.3127 # scaling for position dimensions
            #self.result_scale[-6:-2] = 1/0.4248 # Scale force dimensions so that they weigh equally
            #self.result_scale[-2:] = 1/0.0884 # Scale ankle moment dimensions so that they weigh equally
            self.result_scale[-6:-2] = 27/4 # Scale force dimensions so that they weigh equally
            self.result_scale[-2:] = 27/2 # Scale ankle moment dimensions so that
        elif num_dofs == 29: # Q and M_ankle
            self.mask[:27] = True
            self.mask[58+10] = True # right ankle
            self.mask[58+17] = True # left ankle
            self.result_scale = torch.ones(29, dtype=torch.float32, device=device)  
            self.result_scale[-2:] = 27/2 # Scale force dimensions so that they weigh equally
        elif num_dofs == 81: # Q, Qdot and M
            self.mask[:27] = True
            self.mask[27:54] = True
            self.mask[-27:] = True
            self.result_scale = torch.ones(81, dtype=torch.float32, device=device)
            self.result_scale[-27:] = 1 # scaling for torque dimensions
            self.result_scale[27:54] = 1 # scaling for velocity dimensions
        else:
            raise ValueError(f"Unsupported num_dofs: {num_dofs}; supply explicit state_keys for a trained feature layout")

        self.input_scale = torch.ones(91, dtype=torch.float32, device=device)
        self.input_scale[[5, 6, 12, 13, 32, 33, 39, 40]] = 0
        self.input_scale[[3, 10, 30, 37]] = -1

        self.model = BiomechPriorVAE(
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.requires_grad_(False)
        print(f"Successfully loaded VAE model from: {model_path}")

        self.scaler = load_scaler(scaler_path, device=device)
        print(f"Successfully loaded scaler from: {scaler_path}")
        if self.scaler['mean_'].numel() != num_dofs:
            raise ValueError(f"Scaler expects {self.scaler['mean_'].numel()} features, but VAE is configured for {num_dofs}")

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

    def _postprocess_torch(self, joint_angles_subset_tensor, mode="mean"):
        if self.scaler is not None:
            if mode == "mean":
                joint_angles_unscaled = joint_angles_subset_tensor * self.scaler['scale_'] + self.scaler['mean_']
            elif mode == "var":
                joint_angles_unscaled = joint_angles_subset_tensor * self.scaler['scale_'] ** 2
            else: 
                raise ValueError(f"Unsupported mode: {mode}, only 'mean' and 'var' are supported.")
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
        """Score a vector or row batch of 123/135 BioMAC state features.

        Return the scalar mean Gaussian NLL, or its gradient in the input shape.
        For ambiguous model widths, state_keys specifies the training feature order.
        """
        joint_angles = np.asarray(joint_angles)
        if joint_angles.ndim not in (1, 2) or joint_angles.shape[-1] not in (123, 135) or joint_angles.size == 0:
            raise ValueError(f'Expected a nonempty vector or row batch with 123 or 135 features, got {joint_angles.shape}')
        if not np.isfinite(joint_angles).all():
            raise ValueError('Input features must be finite')
        with torch.set_grad_enabled(getgrad):
            inputs = torch.as_tensor(joint_angles.copy(), dtype=torch.float32, device=self.device)
            inputs.requires_grad_(getgrad)
            error = self._reconstruction_error(inputs)
            if getgrad:
                return torch.autograd.grad(error, inputs)[0].detach().cpu().numpy()
            return error.detach().cpu().numpy()

    def _reconstruction_error(self, joint_angles):
        subset_joints = joint_angles.unsqueeze(0) if joint_angles.ndim == 1 else joint_angles
        
        # Accumulate the forces - 8 dim for each Fx, Fy, Fz for each foot
        if joint_angles.shape[-1] == 135:  
            Fxr, Fyr, Fzr, Fxl, Fyl, Fzl = subset_joints[:, 54:102].split(8, dim=1)
        elif joint_angles.shape[-1] == 123:
            Fxr, Fyr, Fzr, Fxl, Fyl, Fzl = subset_joints[:, 54:90].split(6, dim=1)
        else:
            raise ValueError(f'Number of joint angles {joint_angles.shape[-1]} is not supported. Input shape is {joint_angles.shape}')
        F_vec = torch.concatenate([
            Fyr.sum(dim=1,keepdim=True),
            torch.sqrt(Fxr.sum(dim=1,keepdim=True)**2 + Fzr.sum(dim=1,keepdim=True)**2 + 1e-12),
            Fyl.sum(dim=1,keepdim=True),
            torch.sqrt(Fxl.sum(dim=1,keepdim=True)**2 + Fzl.sum(dim=1,keepdim=True)**2 + 1e-12)
        ],dim=1)

        if joint_angles.shape[-1] == 135:
            subset_joints = torch.cat([subset_joints[:, :54], F_vec, subset_joints[:, 102:]], dim=1)
        elif joint_angles.shape[-1] == 123:
            subset_joints = torch.cat([subset_joints[:, :54], F_vec, subset_joints[:, 90:]], dim=1)
        subset_joints = subset_joints * self.input_scale
        subset_joints = subset_joints[:, self.mask] if self.feature_indices is None else subset_joints[:, self.feature_indices]
        processed_angles = self._preprocess_torch(subset_joints)

        mu_latent, _ = self.model.encode(processed_angles)
        mu_out, var_out = self.model.decode(mu_latent)
        recon_subset = self._postprocess_torch(mu_out)
        var_subset = self._postprocess_torch(var_out, mode="var")
        error = torch.nn.functional.gaussian_nll_loss(recon_subset, subset_joints, var_subset.clamp_min(1e-6), reduction='mean')
        return error

    #Reconstruct the joint angle (with gradient w.r.t original scale), return the gradient
    def reconstruct_withgrad(self, joint_angles):
        return self.reconstruct(joint_angles, getgrad=True)


#Initialize a instance to get the result
_vae_instance = None

def initialize_vae(model_path, scaler_path, num_dofs=27, latent_dim=20, hidden_dim=512, device='cpu', state_keys=None):
    global _vae_instance
    _vae_instance = None
    try:
        _vae_instance = VAEModelWrapper(
            model_path=model_path,
            scaler_path=scaler_path,
            num_dofs=num_dofs,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
            state_keys=state_keys,
        )
        return True
    except Exception as e:
        print(f"Failed to initialize VAE: {e}")
        return False

def reconstruct(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = np.asarray(joint_angles_np).T
    error = _vae_instance.reconstruct(joint_angles)

    return error

def reconstruct_withgrad(joint_angles_np):
    global _vae_instance
    if _vae_instance is None:
        raise RuntimeError("VAE instances is not initialized, call initialize_vae first!")
    
    joint_angles = np.asarray(joint_angles_np).T
    gradient = _vae_instance.reconstruct_withgrad(joint_angles)

    return gradient
