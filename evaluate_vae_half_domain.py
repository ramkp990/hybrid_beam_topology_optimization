
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
    "dataset_half"
]

OUTPUT_DIR = "dataset_half/merged_vae_train_half_domain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_HALF = 32
GRID_FULL = 64
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3

MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helper Functions (FIXED)
# -----------------------------
def reconstruct_full_from_half(rho_half_batch):
    if isinstance(rho_half_batch, torch.Tensor):
        rho_half_batch = rho_half_batch.detach().cpu().numpy()
    
    full_batch = []
    for rho_half in rho_half_batch:
        if rho_half.ndim == 3:
            rho_half = rho_half[0]

        rho_full = np.vstack([rho_half, np.flipud(rho_half)])
        full_batch.append(rho_full)

    return np.array(full_batch)


def reconstruct_full_from_half_torch(rho_half):
    # rho_half: [B,1,32,64]
    flipped = torch.flip(rho_half, dims=[2])  # vertical flip
    full = torch.cat([rho_half, flipped], dim=2)  # [B,1,64,64]
    return full




def is_feasible_full(rho_full):
    geom = (rho_full > 0.5).astype(np.uint8)

    H, W = geom.shape
    area = geom.sum()

    if area < MIN_AREA_FRAC * H * W:
        return False
    if area > MAX_AREA_FRAC * H * W:
        return False

    labeled, num = label(geom)
    if num == 0:
        return False

    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    if max(sizes) < 0.9 * area:
        return False

    # support (left edge)
    if geom[:, 0].sum() == 0:
        return False

    # load point (center right)
    if geom[H//2, W-1] == 0:
        return False

    return True


# -----------------------------
# VAE MODEL
# -----------------------------
class HalfTopologyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 4 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 8, latent_dim)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 4, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_half = self.decode(z)
        return recon_half, mu, logvar

# -----------------------------
# LOSS FUNCTIONS (FIXED)
# -----------------------------
def compute_full_domain_losses(recon_half, target_half):
    """Compute all losses on the full symmetric domain"""
    # Reconstruct full domains (this now works with gradient tensors)
    #recon_full_np = reconstruct_full_from_half(recon_half)
    #target_full_np = reconstruct_full_from_half(target_half)
    
    # Convert back to tensors on the same device
    #recon_full = torch.from_numpy(recon_full_np).unsqueeze(1).float().to(recon_half.device)
    #target_full = torch.from_numpy(target_full_np).unsqueeze(1).float().to(target_half.device)

    recon_full = reconstruct_full_from_half_torch(recon_half)
    target_full = reconstruct_full_from_half_torch(target_half)

    
    # Reconstruction loss (full domain)
    recon_loss = F.binary_cross_entropy(recon_full, target_full, reduction="mean")
    
    # Volume loss (full domain)
    vol_recon = recon_full.mean(dim=[1,2,3])
    vol_target = target_full.mean(dim=[1,2,3])
    vol_loss = F.mse_loss(vol_recon, vol_target)
    
    # Binary loss (full domain)
    binary_loss_val = torch.mean(recon_full * (1 - recon_full))
    
    # Thinness loss (full domain)
    laplace = (
        -4 * recon_full +
        torch.roll(recon_full, 1, 2) +
        torch.roll(recon_full, -1, 2) +
        torch.roll(recon_full, 1, 3) +
        torch.roll(recon_full, -1, 3)
    )
    thinness_loss_val = torch.mean(torch.abs(laplace))
    
    return recon_loss, vol_loss, binary_loss_val, thinness_loss_val

def vae_loss_full_domain(recon_half, target_half, mu, logvar, beta):
    """Full VAE loss computed on symmetric full domain"""
    recon_loss, vol_loss, binary_loss_val, thinness_loss_val = compute_full_domain_losses(recon_half, target_half)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = (
        recon_loss +
        beta * kld +
        0.5 * vol_loss +
        2.0 * binary_loss_val +
        0.05 * thinness_loss_val
    )
    
    return total_loss, {
        'recon': recon_loss.item(),
        'kld': kld.item(),
        'vol': vol_loss.item(),
        'binary': binary_loss_val.item(),
        'thinness': thinness_loss_val.item()
    }

# -----------------------------
# DATA LOADING
# -----------------------------
def load_and_merge():
    data = []
    for d in DATASETS:
        rho = np.load(os.path.join(d, "half_rho_smooth.npy"))
        data.append(rho)
        print(f"Loaded {rho.shape[0]} from {d}")
    return np.concatenate(data, axis=0)

# -----------------------------
# Visualization Function
# -----------------------------
def plot_reconstruction_results(test_half, recon_half, test_full, recon_full, output_dir, num_samples=4):
    """Create side-by-side plots comparing original vs reconstructed designs"""
    import matplotlib.pyplot as plt
    
    indices = np.random.choice(len(test_half), min(num_samples, len(test_half)), replace=False)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(3*num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        # Original half
        axes[0, i].imshow(test_half[idx].squeeze(), cmap='gray_r', origin='lower')
        axes[0, i].set_title(f'Original Half\n(Sample {idx})')
        axes[0, i].axis('off')
        
        # Reconstructed half  
        axes[1, i].imshow(recon_half[idx].squeeze(), cmap='gray_r', origin='lower')
        axes[1, i].set_title('Reconstructed Half')
        axes[1, i].axis('off')
        
        # Original full
        axes[2, i].imshow(test_full[idx].T, cmap='gray_r', origin='lower', extent=[0, 20, 0, 10])
        axes[2, i].set_title('Original Full')
        axes[2, i].set_xlabel('X (mm)')
        axes[2, i].set_ylabel('Y (mm)')
        
        # Reconstructed full
        axes[3, i].imshow(recon_full[idx].T, cmap='gray_r', origin='lower', extent=[0, 20, 0, 10])
        axes[3, i].set_title('Reconstructed Full')
        axes[3, i].set_xlabel('X (mm)')
        axes[3, i].set_ylabel('Y (mm)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "reconstruction_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Reconstruction comparison saved to: {plot_path}")

# -----------------------------
# TRAINING
# -----------------------------
def main():
    data = load_and_merge()
    np.random.shuffle(data)

    N = len(data)
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)

    train = torch.tensor(data[:n_train]).unsqueeze(1).float()
    val = torch.tensor(data[n_train:n_train+n_val]).unsqueeze(1).float()
    test = torch.tensor(data[n_train+n_val:]).unsqueeze(1).float()

    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, BATCH_SIZE)

    model = HalfTopologyVAE(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        beta = min(1.0, epoch / 50)

        model.train()
        train_loss = 0
        train_metrics = {'recon': 0, 'kld': 0, 'vol': 0, 'binary': 0, 'thinness': 0}
        
        for x in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            recon_half, mu, logvar = model(x)
            loss, metrics = vae_loss_full_domain(recon_half, x, mu, logvar, beta)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics[k] += v

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                recon_half, mu, logvar = model(x)
                loss, _ = vae_loss_full_domain(recon_half, x, mu, logvar, beta)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        print(f"Epoch {epoch+1:03d} | β={beta:.2f} | Train={train_loss:.4f} | Val={val_loss:.4f}")
        print(f"  Components: recon={train_metrics['recon']:.4f}, kld={train_metrics['kld']:.4f}, "
              f"vol={train_metrics['vol']:.4f}")

        if beta == 1.0 and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_best.pth"))

    # -----------------------------
    # TEST
    # -----------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "vae_best.pth")))
    model.eval()

    with torch.no_grad():
        recon_half = model(test.to(DEVICE))[0].cpu()
        test_half = test.cpu()

    recon_full = reconstruct_full_from_half(recon_half)
    test_full = reconstruct_full_from_half(test_half)
    
    feasible = sum(is_feasible_full(r) for r in recon_full)
    print(f"Feasible reconstructions: {feasible}/{len(recon_full)} ({100*feasible/len(recon_full):.1f}%)")

    np.save(os.path.join(OUTPUT_DIR, "test_recon_half.npy"), recon_half.numpy())
    np.save(os.path.join(OUTPUT_DIR, "test_recon_full.npy"), recon_full)
    np.save(os.path.join(OUTPUT_DIR, "test_target_half.npy"), test_half.numpy())
    np.save(os.path.join(OUTPUT_DIR, "test_target_full.npy"), test_full)

    # -----------------------------
    # PLOT RESULTS
    # -----------------------------
    print("\n📊 Generating reconstruction comparison plot...")
    plot_reconstruction_results(
        test_half.numpy(), 
        recon_half.numpy(), 
        test_full, 
        recon_full, 
        OUTPUT_DIR,
        num_samples=4
    )

if __name__ == "__main__":
    main()

'''
# plot_half_designs.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_random_half_designs(dataset_path="dataset_half", num_samples=10):
    """
    Plot 10 random half-domain designs to verify orientation and quality
    """
    # Load half-domain data
    try:
        half_rho = np.load(os.path.join(dataset_path, "half_rho_smooth.npy"))
        print(f"Loaded half-domain data: {half_rho.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}/half_rho_smooth.npy")
        print("Please run preprocess_half_dataset.py first")
        return
    
    # Verify shape: should be [N, 32, 64]
    if half_rho.ndim != 3 or half_rho.shape[1] != 32 or half_rho.shape[2] != 64:
        print(f"⚠️ Unexpected shape: {half_rho.shape}")
        print("Expected: [N, 32, 64] (N samples, 32 rows, 64 columns)")
        return

    # Select random samples
    N = len(half_rho)
    indices = np.random.choice(N, min(num_samples, N), replace=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        rho_half = half_rho[idx]
        
        # Plot with proper orientation
        # In FEM: y=0 is bottom, y=31 is top of half-domain
        # So we use origin='lower' for correct physical orientation
        im = axes[i].imshow(rho_half, cmap='gray_r', origin='lower',
                        extent=[0, 20, 0, 10])

        
        axes[i].set_title(f'Sample {idx}', fontsize=10)
        axes[i].set_xlabel('X (mm)', fontsize=8)
        axes[i].set_ylabel('Y (mm)', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        
        # Add load point indicator (right edge, center height)
        axes[i].plot([20, 20], [5, 5], 'ro', markersize=4, label='Load point')
        # Add support indicator (left edge)
        axes[i].plot([0, 0], [5, 5], 'go', markersize=4, label='Support')
        
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=6)

    plt.suptitle('Random Half-Domain Designs (Top 32 rows of 64×64)', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(dataset_path, "half_domain_samples.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plotted {min(num_samples, N)} random half-domain designs")
    print(f"Saved to: {output_path}")

def analyze_half_domain_statistics(half_rho):
    """Basic statistics analysis"""
    print("\n📊 Half-domain Statistics:")
    print(f"Total samples: {len(half_rho)}")
    print(f"Shape: {half_rho.shape}")
    
    # Material distribution
    material_fraction = half_rho.mean(axis=(1,2))  # [N]
    print(f"Material fraction: mean={material_fraction.mean():.3f}, std={material_fraction.std():.3f}")
    print(f"Range: {material_fraction.min():.3f} - {material_fraction.max():.3f}")
    
    # Check if designs are beam-like
    # Beam-like designs should have higher density in diagonal regions
    # Let's check correlation between X and Y positions
    sample_idx = 0
    rho_sample = half_rho[sample_idx]
    
    # Simple check: is there material on left edge?
    left_edge_material = rho_sample[:, 0].mean()
    right_edge_material = rho_sample[:, -1].mean()
    print(f"Sample {sample_idx}: Left edge material={left_edge_material:.3f}, Right edge={right_edge_material:.3f}")

if __name__ == "__main__":
    # Plot random half designs
    plot_random_half_designs()
    
    # Optional: Analyze statistics
    try:
        half_rho = np.load("dataset_half/half_rho_smooth.npy")
        analyze_half_domain_statistics(half_rho)
    except Exception as e:
        print(f"📊 Analysis failed: {e}")

'''