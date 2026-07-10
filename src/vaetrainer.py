import argparse

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import time
from tqdm import tqdm
import pickle
import random

from src.datavisualize import PoseVisualizer
from src.data.addBiomechanicsDataset import AddBiomechanicsDataset, target_dof_names

# use argparse to set the path of dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--addbiomechanics_path", type=str, default="/home/public/data/AddBiomechanicsDataset/train/With_Arm/")
    args = parser.parse_args()
    addbiomechanics_path = args.addbiomechanics_path
else:
    addbiomechanics_path = "/home/public/data/AddBiomechanicsDataset/train/With_Arm/"

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(data_root_path, geometry_path, batch_size=64, train_split=0.8):
    dataset = AddBiomechanicsDataset(
        addbiomechanics_path,
        geometry_folder='',
        window_size=1,
        device='gpu' if torch.cuda.is_available() else 'cpu',
        testing_with_short_dataset=False
    )


    #Initialize data converter
    train_size = int(train_split * len(dataset))
    val_size = int(len(dataset) - train_size)
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    mean, std, n_batches = [{key: 0 for key in state_keys} for _ in range(2)] + [0]
    print('Getting scaler')
    for (data, _, _, _) in tqdm(train_loader):
        for key in state_keys:
            batch_data = data[key].numpy()
            # clamp extreme values to prevent skewing mean/std
            if key == "M_ankle":
                batch_data = np.clip(batch_data, -5, 5)
            mean[key] += np.mean(batch_data, axis=0)
            std[key] += np.var(batch_data, axis=0)
        n_batches += 1

            

    mean = np.array(np.hstack([mean[key] for key in state_keys])).flatten()
    std = np.array(np.hstack([std[key] for key in state_keys])).flatten()
    mean /= n_batches
    std = np.sqrt(std / n_batches)
    # Check for null in std
    std[std == 0] = 1
    # print bounds
    for i, key in enumerate(target_dof_names):
        print(f"{key}: mean={mean[i]:.4f}, std={std[i]:.4f}")
        if 'vel' in state_keys:
            print(f"{key}_vel: mean={mean[i+23]:.4f}, std={std[i+23]:.4f}")
    if "force" in state_keys:
        for i, key in enumerate(['Fy_r', 'Flat_r', 'Fy_l', 'Flat_l']):
            print(f"{key}: mean={mean[-6+i]:.4f}, std={std[-6+i]:.4f}")
    if "M_ankle" in state_keys:
        for i, key in enumerate(['M_ankle_r', 'M_ankle_l']):
            print(f"{key}: mean={mean[-2+i]:.4f}, std={std[-2+i]:.4f}")
            

    return train_loader, val_loader, {'mean_': torch.tensor(mean, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'),
                                       'scale_': torch.tensor(std, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')}


#VAE network configuration    
class BiomechPriorVAE(nn.Module):
    def __init__(self, num_dofs, latent_dim=20, hidden_dim=512):
        super(BiomechPriorVAE, self).__init__()

        self.num_dofs = num_dofs
        self.latent_dim = latent_dim
        print(self.num_dofs, self.latent_dim)

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_dofs),
            nn.Linear(num_dofs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2*num_dofs)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp encoder logvar to prevent extreme values
        logvar = logvar.clamp(min=-10, max=10)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z
    
    def decode(self, x):
        x = self.decoder(x)
        mu_out = x[..., :self.num_dofs]

        logvar_out = torch.tanh(x[..., self.num_dofs:]) * 8  # Scale tanh output to range [-8, 8]
        var_out = torch.exp(logvar_out)
        return mu_out, var_out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec_x = self.decode(z)
        return rec_x, mu, logvar

@torch.compile
def vae_loss(x, recon_x, mu, logvar, beta):
    recon_x, recon_x_var = recon_x
    batch_size = x.size(0)
    
    # Reconstruction loss: sum over features, mean over batch
    recon_loss = F.gaussian_nll_loss(recon_x, x, recon_x_var, reduction='sum') / batch_size
    
    # KL divergence loss: sum over latent dims, mean over batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = 1e-1 * recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class BiomechPriorVAETrainer:
    def __init__(self,
                 model: BiomechPriorVAE,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    def train_epoch(self,
                    train_loader,
                    optimizer: torch.optim.Optimizer,
                    beta=1.0,
                    scaler=None):
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_idx, (data, _, _, _) in enumerate(tqdm(train_loader, desc="Training")):
            data = torch.concat([data[key] for key in state_keys], dim=-1)
            data = data.to(self.device)
            if scaler is not None:
                data = (data - scaler['mean_']) / scaler['scale_']
            optimizer.zero_grad()

            recon_data, mu, logvar = self.model(data)
            loss, recon_loss, kl_loss = vae_loss(x=data, recon_x=recon_data, mu=mu, logvar=logvar, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Add gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            if batch_idx == 1000:
                continue

        num_batches = batch_idx + 1

        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'kl_loss': epoch_kl_loss / num_batches
        }
        
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, beta=1.0, save_path=None, scaler=None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            #Train
            train_metrics = self.train_epoch(train_loader=train_loader, optimizer=optimizer, beta=beta, scaler=scaler)

            #Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader=val_loader, beta=beta, scaler=scaler)
                scheduler.step(val_metrics['loss'])

                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f}")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print("Best model saved!")
                
            else:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}")

            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['recon_loss'].append(train_metrics['recon_loss'])
            self.training_history['kl_loss'].append(train_metrics['kl_loss'])

    def validate(self, val_loader, beta=1.0, scaler=None):
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            for i, (data, _, _, _) in enumerate(val_loader):
                data = torch.concat([data[key] for key in state_keys], dim=-1)
                data = data.to(self.device)
                if scaler is not None:
                    data = (data - scaler['mean_']) / scaler['scale_']

                recon_data, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(x=data, recon_x=recon_data, mu=mu, logvar=logvar, beta=beta)

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

                if i == 500:
                    continue

            num_batches = i + 1

            return{
                'loss': val_loss / num_batches,
                'recon_loss': val_recon_loss / num_batches,
                'kl_loss': val_kl_loss / num_batches
            }
    
    def test(self, val_loader, scaler):
        self.model.eval()
        result = []

        with torch.no_grad():
            for i, (data, _, _, _) in enumerate(val_loader):
                data = torch.concat([data[key] for key in state_keys], dim=-1)
                data = data.to(self.device)
                if scaler is not None:
                    data = (data - scaler['mean_']) / scaler['scale_']
                recon_data, mu, logvar = self.model(data)
                
                ori_data = data.cpu().numpy()
                rec_data = recon_data.cpu().numpy()


                result.append({
                    'original': ori_data,
                    'recon': rec_data
                })
        
        return result

    def visualize_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(self.training_history['loss'])
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        axes[1].plot(self.training_history['recon_loss'])
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        
        axes[2].plot(self.training_history['kl_loss'])
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        
        plt.tight_layout()
        #plt.show()



def train_model(
        data_root_path,
        geometry_path,
        output_path,
        latent_dim=20,
        batch_size=64,
        num_dofs=27,
        num_epochs=100,
        learning_rate=1e-3,
        beta=1.0,
        train_split=0.8,
        scaler=None
):
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"BiomechPriorVAE_best_{num_dofs}.pth")
    scaler_path = os.path.join(output_path, f"scaler_{num_dofs}.pkl")

    train_loader, val_loader, scaler = load_data(
        data_root_path=data_root_path,
        geometry_path=geometry_path,
        batch_size=batch_size,
        train_split=train_split
    )
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    model = BiomechPriorVAE(num_dofs=num_dofs, latent_dim=latent_dim)
    model.compile()
    trainer = BiomechPriorVAETrainer(model=model)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta=beta,
        save_path=save_path,
        scaler=scaler
    )
    trainer.visualize_history()

    print("Training complete!")
    print("\n" + "="*50)
    print("Starting visualize test sample...")
    print("="*50)

    #visualize_result(
    #    model=model,
    #    trainer=trainer,
    #    val_loader=val_loader,
    #    scaler=scaler
    #)

    return model, trainer

def test_model(
        data_root_path,
        geometry_path,
        model_path,
        scaler_path,
        latent_dim=20,
        batch_size=64,
        num_dofs=27,
        train_split=0.8,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    train_loader, val_loader, scaler = load_data(
        data_root_path=data_root_path,
        geometry_path=geometry_path,
        batch_size=batch_size,
        train_split=train_split
    )
    print("Data loaded successfully!")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
    
    model = BiomechPriorVAE(num_dofs=num_dofs, latent_dim=latent_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Model loaded successfully on {device}!")

    trainer = BiomechPriorVAETrainer(model=model, device=device)
    print("\n" + "="*50)
    print("Starting visualize test sample...")
    print("="*50)

    #visualize_result(
    #    model=model,
    #    trainer=trainer,
    #    val_loader=val_loader,
    #    scaler=scaler
    #)

    return model, trainer, scaler



if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root_path = os.path.join(script_dir, "../data/Dataset")
    output_path = os.path.join(script_dir, "../result/model/")
    geometry_path = os.path.join(script_dir, "../data/Geometry/")
    analysis_path = os.path.join(script_dir, "../result/")

    mode = "train" # "train" / "test" / "analyze"
    model = "q_qdot"    # "q", "q_qdot", "F", "q_dot_F"
    
    num_dofs_dict = {
        "q": 23,
        "q_qdot": 23*2,
        "q_qdot_M_ankle": 23*2+2,
        "F": 23+4,
        "q_dot_F": 23*2+4,
        "q_qdot_F_M_ankle": 23*2+4+2,
        "q_M_ankle": 23+2,
        "q_M": 23+23,
        "q_qdot_M": 23*2+23,
    }

    latent_size = {
        "q": 24,
        "q_qdot": 12,
        "q_qdot_M_ankle": 24,
        "F": 24,
        "q_dot_F": 24,
        "q_M_ankle": 24,
        "q_M": 24,
        "q_qdot_M": 24,
    }
    state_keys_ = {
        "q": ["pos"],
        "q_qdot": ["pos", "vel"],
        "q_qdot_M_ankle": ["pos", "vel", "M_ankle"],
        "F": ["pos", "force"],
        "q_dot_F": ["pos", "vel", "force"],
        "q_qdot_F_M_ankle": ["pos", "vel", "force", "M_ankle"],
        "q_M_ankle": ["pos", "M_ankle"],
        "q_M": ["pos", "tau"],
        "q_qdot_M": ["pos", "vel", "tau"],
    }

    state_keys = state_keys_[model]
    num_dofs = num_dofs_dict[model]
    if mode == "train":
        print("Starting training...")
        model, trainer = train_model(
            data_root_path=data_root_path,
            geometry_path=geometry_path,
            output_path=output_path,
            latent_dim=24,
            batch_size=256,
            num_dofs=num_dofs,
            num_epochs=100,
            learning_rate=1e-3,
            train_split=0.8,
        )
        
    elif mode == "test":
        print("Starting testing...")
        model_path = os.path.join(output_path, "BiomechPriorVAE_best.pth")
        scaler_path = os.path.join(output_path, "scaler.pkl")
        model, trainer, scaler = test_model(
            data_root_path=data_root_path,
            geometry_path=geometry_path,
            model_path=model_path,
            scaler_path=scaler_path,
            latent_dim=24,
            batch_size=256,
            num_dofs=num_dofs,
            train_split=0.8,
        )


    else:
        print("Invalid mode! Please select 'train', 'test' or 'analyze' mode.")

    plt.show()