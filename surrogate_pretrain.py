# surrogate_pretrain.py
import os
import numpy as np
import torch
from vae_model import TopologyVAE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
import joblib

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

# Load VAE encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TopologyVAE(latent_dim=32).to(device)
model.load_state_dict(torch.load("dataset/merged_vae_train/vae_best.pth"))
model.eval()

all_z = []
all_compliance = []

with torch.no_grad():
    for d in DATASETS:
        rho = np.load(os.path.join(d, "rho_smooth.npy"))
        meta = np.load(os.path.join(d, "metadata_with_compliance.npy"), allow_pickle=True)
        compliance = meta['compliance']
        
        rho_tensor = torch.from_numpy(rho).unsqueeze(1).float().to(device)
        mu, _ = model.encode(rho_tensor)
        z = mu.cpu().numpy()
        
        all_z.append(z)
        all_compliance.append(compliance)
        print(f"Loaded {len(rho)} samples from {d}")

# Merge datasets
z_train = np.concatenate(all_z, axis=0)
compliance_train = np.concatenate(all_compliance, axis=0)
print(f"\n✅ Total samples: {len(z_train)}")



# --- CRITICAL: Use subset for GP training ---
SUBSET_SIZE = 4000
if len(z_train) > SUBSET_SIZE:
    indices = np.random.choice(len(z_train), size=SUBSET_SIZE, replace=False)
    z_train = z_train[indices]
    compliance_train = compliance_train[indices]
    print(f"   Using subset of {SUBSET_SIZE} samples for GP training")

# Scale latent vectors
scaler = StandardScaler()
z_scaled = scaler.fit_transform(z_train)

# Train GP with increased iterations
kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(0.1, 10.0))
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-3,           # Accounts for VAE reconstruction noise
    n_restarts_optimizer=15,  # More restarts for better optimum
    normalize_y=True,
    #max_iter=20000,       # Prevent convergence warnings
    random_state=42
)

gp.fit(z_scaled, compliance_train)

# Save everything
os.makedirs("surrogate", exist_ok=True)
joblib.dump(gp, "surrogate/gp_surrogate.pkl")
joblib.dump(scaler, "surrogate/z_scaler.pkl")

print(f"\n✅ GP surrogate trained and saved!")
print(f"   R² score: {gp.score(z_scaled, compliance_train):.4f}")
print(f"   Kernel: {gp.kernel_}")
print(f"   Samples used: {len(z_train)}")