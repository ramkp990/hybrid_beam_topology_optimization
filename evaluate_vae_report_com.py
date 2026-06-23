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
    "dataset/dataset_cantilever_sym6_mmc1",
    "dataset/dataset_cantilever_sym6_mmc2",
    "dataset/dataset_cantilever_sym6_mmc3",
    "dataset/dataset_cantilever_sym6_mmc4",
    "dataset/dataset_cantilever_sym6_mmc5",
    "dataset/dataset_cantilever_sym6_mmc6",
    "dataset/dataset_cantilever_sym6_mmc7",
    "dataset/dataset_cantilever_sym6_mmc8",
    "dataset/dataset_cantilever_sym6_mmc9"
]

OUTPUT_DIR = "dataset/merged_vae_train_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR_kld = "dataset/merged_vae_train_report_kld"
os.makedirs(OUTPUT_DIR_kld, exist_ok=True)

GRID = 64
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# VAE MODEL
# -----------------------------
class TopologyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder — unchanged
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder — unchanged
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # compliance prediction head

        self.compliance_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)     # outputs raw scalar, no activation
        )


    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.dec(h)

    def predict_compliance(self, mu):
        return self.compliance_head(mu).squeeze(-1)   

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        c_pred     = self.predict_compliance(mu)
        return recon, mu, logvar, c_pred

# -----------------------------
# LOSS FUNCTIONS
# -----------------------------
def projection_loss(rho):
    rho_bin = (rho > 0.5).float()
    return F.mse_loss(rho, rho_bin.detach())

def volume_loss(rho):
    vol = rho.mean(dim=[1,2,3])
    target = 0.5 * (MIN_AREA_FRAC + MAX_AREA_FRAC)
    #target = torch.empty_like(vol).uniform_(MIN_AREA_FRAC, MAX_AREA_FRAC)
    return F.mse_loss(vol, torch.full_like(vol, target))

def entropy_loss(rho):
    eps = 1e-6
    return -(rho * torch.log(rho + eps) + (1 - rho) * torch.log(1 - rho + eps)).mean()

def thinness_loss(rho):
    laplace = (
        -4 * rho +
        torch.roll(rho, 1, 2) +
        torch.roll(rho, -1, 2) +
        torch.roll(rho, 1, 3) +
        torch.roll(rho, -1, 3)
    )
    return torch.mean(torch.abs(laplace))

def binary_loss(rho):
    return torch.mean(rho * (1 - rho))

def focal_binary_loss(rho, gamma=2.0):
    # Punishes values near 0.5 quadratically
    return torch.mean((rho * (1 - rho)) ** gamma)

def dice_loss(recon, target):
    # Better than BCE for sparse binary fields
    intersection = (recon * target).sum(dim=[1,2,3])
    union = recon.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
    return 1.0 - (2.0 * intersection / (union + 1e-6)).mean()



def vae_loss(recon, target, mu, logvar, beta,
             c_pred=None, c_true=None, w_compliance=0.0):
    
    recon_loss = F.binary_cross_entropy(recon, target, reduction="mean")
    dice       = dice_loss(recon, target)
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    vol        = volume_loss(recon)
    bce_bin    = focal_binary_loss(recon, gamma=2.0)


    comp_loss = torch.tensor(0.0, device=recon.device)
    if w_compliance > 0.0 and c_pred is not None and c_true is not None:
        # normalise compliance to ~[0,1] 
        # mean ~54, std ~15 from dataset
        c_norm  = (c_true - 54.2) / 15.0
        cp_norm = (c_pred - 54.2) / 15.0
        comp_loss = F.mse_loss(cp_norm, c_norm)


    loss = (recon_loss
          + beta    * kld
          + 0.5     * vol
          + dice
          + 4.0     * bce_bin
          + w_compliance * comp_loss)

    return loss, {
        "recon":      recon_loss.item(),
        "kld":        kld.item(),
        "vol":        vol.item(),
        "thin":       0.0,
        "binary":     bce_bin.item(),
        "compliance": comp_loss.item(),
    }

def diagnose_infeasibility(rho, min_area_frac=0.10, max_area_frac=0.50, grid=64):
    """
    Diagnose why a reconstruction is infeasible.
    Returns: dict with violation details
    """
    geom = (rho > 0.5).astype(np.uint8)
    area = geom.sum()
    total_pixels = grid * grid
    area_frac = area / total_pixels
    
    result = {
        'area_fraction': area_frac,
        'volume_violation': False,
        'connectivity_violation': False,
        'support_violation': False,
        'load_violation': False,
        'violation_type': 'feasible'
    }
    
    # Check volume constraints
    if area_frac < min_area_frac:
        result['volume_violation'] = True
        result['violation_type'] = 'volume_too_low'
        return result
    elif area_frac > max_area_frac:
        result['volume_violation'] = True
        result['violation_type'] = 'volume_too_high'
        return result
    
    # Check connectivity
    labeled, num = label(geom)
    if num == 0:
        result['connectivity_violation'] = True
        result['violation_type'] = 'no_material'
        return result
        
    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    largest = max(sizes)
    largest_component_frac = largest / area if area > 0 else 0
    
    if largest_component_frac < 0.9:
        result['connectivity_violation'] = True
        result['violation_type'] = 'disconnected_components'
        return result
    
    # Check support and load connections
    main_idx = 1 + sizes.index(largest)
    main_mask = (labeled == main_idx)
    
    LOAD_POINT_X = grid - 1
    LOAD_POINT_Y = grid // 2
    
    if main_mask[:, 0].sum() == 0:
        result['support_violation'] = True
        result['violation_type'] = 'no_support_connection'
        return result
        
    if main_mask[LOAD_POINT_Y, LOAD_POINT_X] == 0:
        result['load_violation'] = True
        result['violation_type'] = 'no_load_connection'
        return result
    
    return result



# -----------------------------
# DATA LOADING
# -----------------------------
def load_and_merge():
    data = []
    for d in DATASETS:
        rho = np.load(os.path.join(d, "rho_smooth.npy"))
        data.append(rho)
        print(f"Loaded {rho.shape[0]} from {d}")
    return np.concatenate(data, axis=0)

# -----------------------------
# FEASIBILITY CHECK (unchanged)
# -----------------------------
def is_feasible(rho):
    geom = (rho > 0.5).astype(np.uint8)
    area = geom.sum()

    if area < MIN_AREA_FRAC * GRID * GRID:
        return False
    if area > MAX_AREA_FRAC * GRID * GRID:
        return False

    labeled, num = label(geom)
    if num == 0:
        return False

    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    if max(sizes) < 0.9 * area:
        return False

    return True


def plot_reconstruction_samples(original, reconstructed, epoch, output_dir):
    """Plot 6 random reconstruction samples"""
    indices = np.random.choice(len(original), min(6, len(original)), replace=False)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(original[idx].squeeze(), cmap='gray_r', origin='lower')
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[idx].squeeze(), cmap='gray_r', origin='lower')
        axes[1, i].set_title(f'Reconstructed {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Handle both integer epochs and string labels like 'final'
    if isinstance(epoch, int):
        filename = f'reconstruction_epoch_{epoch:03d}.png'
    else:
        filename = f'reconstruction_{epoch}.png'
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_to_file(metrics, output_dir):
    """Save metrics to a text file"""
    with open(os.path.join(output_dir, 'training_metrics.txt'), 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Beta\n")
        for metric in metrics:
            f.write(f"{metric['epoch']},{metric['train_loss']:.6f},{metric['val_loss']:.6f},{metric['beta']:.6f}\n")

# -----------------------------
# TRAINING
# -----------------------------
def main():
    data = load_and_merge()
    

    compliance_all = np.load("dataset/compliance_all.npy")
    assert len(compliance_all) == len(data), \
        f"Mismatch: {len(data)} designs but {len(compliance_all)} compliance values"
    valid = compliance_all > 0   # filter FEM failures
    data           = data[valid]
    compliance_all = compliance_all[valid]
    print(f"After filtering FEM failures: {len(data)} designs")

    np.random.seed(42)
    idx = np.random.permutation(len(data))
    data           = data[idx]
    compliance_all = compliance_all[idx]

    N       = len(data)
    n_train = int(0.7  * N)
    n_val   = int(0.15 * N)

    train    = torch.tensor(data[:n_train]).unsqueeze(1).float()
    val      = torch.tensor(data[n_train:n_train+n_val]).unsqueeze(1).float()
    test     = torch.tensor(data[n_train+n_val:]).unsqueeze(1).float()


    c_train = torch.tensor(compliance_all[:n_train]).float()
    c_val   = torch.tensor(compliance_all[n_train:n_train+n_val]).float()
    c_test  = torch.tensor(compliance_all[n_train+n_val:]).float()

    train_loader = DataLoader(
        TensorDataset(train, c_train), BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val, c_val), BATCH_SIZE
    )
    model = TopologyVAE(LATENT_DIM).to(DEVICE)
    #opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val = float("inf")
    best_val_kld = float("inf")
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    metrics_history = []
    # compliance weight: start at 0, ramp up from epoch 50 to epoch 150
    # gives reconstruction time to stabilize before adding performance signal
    W_COMPLIANCE_MAX = 2.0

    for epoch in range(EPOCHS):
        # your existing beta schedule unchanged
        cycle_len = 75
        cycle_pos = epoch % cycle_len
        warmup    = 3
        beta = 0.0 if cycle_pos < warmup else \
               min(0.04, ((cycle_pos - warmup) / (cycle_len - warmup)) * 0.04)


        if epoch < 50:
            w_compliance = 0.0
        else:
            w_compliance = W_COMPLIANCE_MAX * min(1.0, (epoch - 50) / 100.0)

        model.train()
        train_loss = 0
        breakdown_train = {"recon":0,"kld":0,"vol":0,"thin":0,
                           "binary":0,"compliance":0}

        for x, c_true in train_loader:          # ← unpack compliance
            x      = x.to(DEVICE)
            c_true = c_true.to(DEVICE)
            opt.zero_grad()

            recon, mu, logvar, c_pred = model(x)   # ← 4 outputs now

            loss, bd = vae_loss(
                recon, x, mu, logvar, beta,
                c_pred=c_pred, c_true=c_true,
                w_compliance=w_compliance
            )
            loss.backward()
            opt.step()
            train_loss += loss.item()
            for k in breakdown_train:
                breakdown_train[k] += bd[k]

        # validation loop — same structure, w_compliance=0 for clean metric
        model.eval()
        val_loss = 0
        breakdown_val = {"recon":0,"kld":0,"vol":0,"thin":0,
                         "binary":0,"compliance":0}

        with torch.no_grad():
            for x, c_true in val_loader:
                x      = x.to(DEVICE)
                c_true = c_true.to(DEVICE)
                recon, mu, logvar, c_pred = model(x)
                loss_v, bd = vae_loss(
                    recon, x, mu, logvar, beta,
                    c_pred=c_pred, c_true=c_true,
                    w_compliance=0.0    # don't penalize in val
                )
                val_loss += loss_v.item()
                for k in breakdown_val:
                    breakdown_val[k] += bd[k]

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        # Normalize breakdowns per batch
        for k in breakdown_train:
            breakdown_train[k] /= len(train_loader)
        for k in breakdown_val:
            breakdown_val[k] /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'beta': beta
        })

        print(f"Epoch {epoch+1:03d} | β={beta:.4f} | "
            f"Recon={breakdown_train['recon']:.4f} | KLD={breakdown_train['kld']:.4f} | "
            f"Vol={breakdown_train['vol']:.4f} | Bin={breakdown_train['binary']:.4f} | "
            f"Total={train_loss:.4f}")
        print(f"         VAL    | "
            f"Recon={breakdown_val['recon']:.4f} | KLD={breakdown_val['kld']:.4f} | "
            f"Vol={breakdown_val['vol']:.4f} | Bin={breakdown_val['binary']:.4f} | "
            f"Total={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_best.pth"))
                
        if epoch == 25:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_epoch_25.pth"))

        if epoch == 50:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_epoch_50.pth"))

        if epoch == 75:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_epoch_75.pth"))

        if epoch == 99:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vae_epoch_100.pth"))    
        
        # Save reconstruction samples at best validation
        if epoch % 10 == 0 or epoch == EPOCHS - 1:  # Every 10 epochs + final
            with torch.no_grad():
                sample_batch = val[:6].to(DEVICE)
                #recon_sample, _, _ = model(sample_batch)
                recon_sample, mu, logvar, c_pred = model(sample_batch)
                plot_reconstruction_samples(
                    sample_batch.cpu().numpy(),
                    recon_sample.cpu().numpy(),
                    epoch + 1,
                    OUTPUT_DIR
                )

    # Save final metrics
    plot_training_curves(train_losses, val_losses, OUTPUT_DIR)
    save_metrics_to_file(metrics_history, OUTPUT_DIR)

    # -----------------------------
    # TEST
    # -----------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "vae_best.pth")))
    model.eval()

    with torch.no_grad():
        recon = model(test.to(DEVICE))[0].cpu().numpy()

    # Diagnose all test samples
    diagnostics = []
    feasible_count = 0
    for i, r in enumerate(recon):
        diag = diagnose_infeasibility(r.squeeze())
        diagnostics.append(diag)
        if diag['violation_type'] == 'feasible':
            feasible_count += 1

    print(f"Feasible reconstructions: {feasible_count}/{len(recon)} ({100*feasible_count/len(recon):.1f}%)")
    
    # Count violation types
    violation_counts = {}
    for diag in diagnostics:
        vtype = diag['violation_type']
        violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
    
    print("\nViolation breakdown:")
    for vtype, count in violation_counts.items():
        print(f"  {vtype}: {count}")
    
    # Save diagnostics
    np.save(os.path.join(OUTPUT_DIR, "test_recon.npy"), recon)
    np.save(os.path.join(OUTPUT_DIR, "test_diagnostics.npy"), diagnostics)
    
    # Plot examples of each violation type
    plot_violation_examples(recon, diagnostics, OUTPUT_DIR)

# Add this visualization function
def plot_violation_examples(recon, diagnostics, output_dir, max_examples=3):
    """Plot examples of each violation type"""
    violation_types = set(d['violation_type'] for d in diagnostics)
    
    for vtype in violation_types:
        if vtype == 'feasible':
            continue
            
        # Find examples of this violation
        indices = [i for i, d in enumerate(diagnostics) if d['violation_type'] == vtype]
        if not indices:
            continue
            
        # Take up to max_examples
        plot_indices = indices[:max_examples]
        n_plots = len(plot_indices)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
            
        for idx, recon_idx in enumerate(plot_indices):
            rho = recon[recon_idx].squeeze()
            axes[idx].imshow(rho.T, cmap='gray_r', origin='lower')
            axes[idx].set_title(f'{vtype}\nArea: {diagnostics[recon_idx]["area_fraction"]:.2f}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violation_{vtype}.png'), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()