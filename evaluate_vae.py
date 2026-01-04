'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import label
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Configuration
# -----------------------------
SAVE_DIR = "dataset/dataset_cantilever_sym6_mmc"
os.makedirs(SAVE_DIR, exist_ok=True)

# VAE config
LATENT_DIM = 20
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3

# Constraints (from paper)
GRID = 64
MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50
MIN_AREA = int(MIN_AREA_FRAC * GRID * GRID)
MAX_AREA = int(MAX_AREA_FRAC * GRID * GRID)
LARGE_COMPONENT_RATIO = 0.90
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2
LOAD_POINT_WINDOW = 3

# -----------------------------
# VAE Model
# -----------------------------
class VAE3Channel(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 3 input channels
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16→8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder: 1 output channel
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU()
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 32→64
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -----------------------------
# Feasibility Validator 
# -----------------------------
def is_feasible(rho_recon):
    geom = (rho_recon > 0.5).astype(np.uint8)
    area = geom.sum()
    if area < MIN_AREA or area > MAX_AREA:
        return False

    labeled, num = label(geom, structure=np.ones((3,3)))
    if num == 0:
        return False
    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    largest = max(sizes)
    if largest < LARGE_COMPONENT_RATIO * area:
        return False
    main_mask = (labeled == (1 + sizes.index(largest)))

    if main_mask[:, 0].sum() == 0:
        return False

    y_low = max(0, LOAD_POINT_Y - LOAD_POINT_WINDOW)
    y_high = min(GRID, LOAD_POINT_Y + LOAD_POINT_WINDOW + 1)
    if main_mask[y_low:y_high, LOAD_POINT_X].sum() == 0:
        return False

    return True

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    phi = np.load(os.path.join(SAVE_DIR, "phi.npy"))
    rho_smooth = np.load(os.path.join(SAVE_DIR, "rho_smooth.npy"))
    geometry = np.load(os.path.join(SAVE_DIR, "geometry.npy"))

    # Normalize phi to [0,1]
    phi_clipped = np.clip(phi, -5.0, 1.0)
    phi_norm = (phi_clipped - (-5.0)) / (1.0 - (-5.0))

    # Stack input channels
    input_channels = np.stack([phi_norm, rho_smooth, geometry], axis=1).astype(np.float32)
    target = rho_smooth.astype(np.float32)

    # Create dataset
    X = torch.from_numpy(input_channels)
    y = torch.from_numpy(target).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = VAE3Channel(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Loss
    def vae_loss(recon, target, mu, logvar):
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    # Training
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss = vae_loss(recon, batch_y, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_model.pth"))
    print("✅ Model saved.")

    # Evaluation
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(X.to(device))
        recon_np = recon.cpu().numpy().squeeze(1)

    # Save reconstructions
    np.save(os.path.join(SAVE_DIR, "recon_rho.npy"), recon_np)

    # Feasibility check
    feasible_count = sum(is_feasible(recon_np[i]) for i in range(recon_np.shape[0]))
    feasibility_rate = feasible_count / recon_np.shape[0]
    print(f"\n✅ Feasibility rate: {feasible_count}/{recon_np.shape[0]} = {feasibility_rate:.1%}")

    # Visualization
    n_show = min(5, recon_np.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.5 * n_show))
    fig.suptitle("VAE Reconstruction: [Input-phi, Input-rho, Target, Recon]", fontsize=14)

    for i in range(n_show):
        axes[i, 0].imshow(phi[i], cmap='RdBu', origin='lower')
        axes[i, 0].set_title("Input: phi")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(rho_smooth[i], cmap='gray', origin='lower')
        axes[i, 1].set_title("Input: rho")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(rho_smooth[i], cmap='gray', origin='lower')
        axes[i, 2].set_title("Target: rho")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(recon_np[i], cmap='gray', origin='lower')
        axes[i, 3].set_title("Recon: rho")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "vae_reconstructions.png"), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ Done.")

if __name__ == "__main__":
    main()

'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import label
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASETS = [
    "dataset/dataset_cantilever_sym6_mmc",
    "dataset/dataset_cantilever_sym6_mmc1"
]
OUTPUT_DIR = "dataset/merged_vae_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VAE config
LATENT_DIM = 16
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Constraints (from paper)
GRID = 64
MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50
MIN_AREA = int(MIN_AREA_FRAC * GRID * GRID)
MAX_AREA = int(MAX_AREA_FRAC * GRID * GRID)
LARGE_COMPONENT_RATIO = 0.90
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2
LOAD_POINT_WINDOW = 3

# -----------------------------
# VAE Model (Simplified: 1-channel input)
# -----------------------------
class TopologyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16→8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8→4
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 32→64
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -----------------------------
# Feasibility Validator 
# -----------------------------
def is_feasible(rho_recon):
    geom = (rho_recon > 0.5).astype(np.uint8)
    area = geom.sum()
    if area < MIN_AREA or area > MAX_AREA:
        return False

    labeled, num = label(geom, structure=np.ones((3,3)))
    if num == 0:
        return False
    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    largest = max(sizes)
    if largest < LARGE_COMPONENT_RATIO * area:
        return False
    main_mask = (labeled == (1 + sizes.index(largest)))

    if main_mask[:, 0].sum() == 0:
        return False

    y_low = max(0, LOAD_POINT_Y - LOAD_POINT_WINDOW)
    y_high = min(GRID, LOAD_POINT_Y + LOAD_POINT_WINDOW + 1)
    if main_mask[y_low:y_high, LOAD_POINT_X].sum() == 0:
        return False

    return True

# -----------------------------
# Load and merge datasets
# -----------------------------
def load_and_merge_datasets():
    all_rho = []
    for d in DATASETS:
        rho = np.load(os.path.join(d, "rho_smooth.npy"))
        all_rho.append(rho)
        print(f"Loaded {rho.shape[0]} samples from {d}")
    
    merged_rho = np.concatenate(all_rho, axis=0)
    print(f"✅ Total merged samples: {merged_rho.shape[0]}")
    return merged_rho

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and merge
    rho_data = load_and_merge_datasets()
    np.save(os.path.join(OUTPUT_DIR, "rho_merged.npy"), rho_data)

    # Shuffle and split
    N = len(rho_data)
    indices = np.random.permutation(N)
    rho_data = rho_data[indices]

    n_train = int(TRAIN_RATIO * N)
    n_val = int(VAL_RATIO * N)
    n_test = N - n_train - n_val

    train_data = rho_data[:n_train]
    val_data = rho_data[n_train:n_train+n_val]
    test_data = rho_data[n_train+n_val:]

    print(f"Split: Train={n_train}, Val={n_val}, Test={n_test}")

    # Create datasets
    X_train = torch.from_numpy(train_data).unsqueeze(1).float()
    X_val = torch.from_numpy(val_data).unsqueeze(1).float()
    X_test = torch.from_numpy(test_data).unsqueeze(1).float()

    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = TopologyVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Loss with β-annealing
    def vae_loss(recon, target, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld

    # Training loop with validation
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # β-annealing: start at 0.1, reach 1.0 by epoch 100
        beta = min(1.0, 0.1 + 0.9 * epoch / 100)

        # Train
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar, beta=1.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | β={beta:.2f} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_best.pth"))

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "vae_best.pth")))
    model.eval()
    test_recon = []
    test_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch = X_test[i:i+BATCH_SIZE].to(device)
            recon, _, _ = model(batch)
            test_recon.append(recon.cpu())
            test_loss += F.mse_loss(recon, batch, reduction='sum').item()
    test_recon = torch.cat(test_recon, dim=0)
    test_mse = test_loss / (len(X_test) * GRID * GRID)
    print(f"\n✅ Final Test MSE: {test_mse:.6f}")

    # Feasibility rate on test reconstructions
    recon_np = test_recon.squeeze(1).numpy()
    feasible_count = sum(is_feasible(recon_np[i]) for i in range(len(recon_np)))
    feasibility_rate = feasible_count / len(recon_np)
    print(f"✅ Test Feasibility Rate: {feasible_count}/{len(recon_np)} = {feasibility_rate:.1%}")

    # Save results
    np.save(os.path.join(OUTPUT_DIR, "test_recon.npy"), recon_np)
    print(f"\n✅ All results saved to {OUTPUT_DIR}")

    # Plot reconstructions
    n_show = min(5, len(recon_np))
    fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show))
    fig.suptitle("Test Set: [Ground Truth, Reconstruction]", fontsize=14)

    for i in range(n_show):
        axes[i, 0].imshow(X_test[i].squeeze(), cmap='gray', origin='lower')
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon_np[i], cmap='gray', origin='lower')
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_reconstructions.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()