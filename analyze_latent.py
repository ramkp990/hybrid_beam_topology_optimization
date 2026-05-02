"""
#precompute_compliance.py
#Run FEM on every design in the dataset and save compliance values.
#Run this once — takes a while but only needs to happen once.


import numpy as np
import os
from tqdm import tqdm
from fem_code import fem_physical_compliance

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
    "dataset/dataset_cantilever_sym6_mmc9",
]

LOAD_VALUE = -100.0   # adjust if yours is different
SAVE_PATH  = "dataset/compliance_all.npy"

all_rho = []
for d in DATASETS:
    p = os.path.join(d, "rho_smooth.npy")
    if os.path.exists(p):
        arr = np.load(p)
        all_rho.append(arr)
        print(f"  Loaded {arr.shape[0]} from {d}")
    else:
        print(f"  WARNING: not found — {p}")

rho_all = np.concatenate(all_rho, axis=0)
print(f"\nTotal designs: {len(rho_all)}")
print("Computing compliance via FEM...")

compliance_all = np.zeros(len(rho_all), dtype=np.float32)
failed = 0

for i in tqdm(range(len(rho_all))):
    try:
        c, _ = fem_physical_compliance(rho_all[i], load_value=LOAD_VALUE)
        compliance_all[i] = float(c)
    except Exception as e:
        # mark failed designs with -1 so you can filter them later
        compliance_all[i] = -1.0
        failed += 1
        if failed <= 5:
            print(f"  FEM failed at index {i}: {e}")

print(f"\nDone. Failed: {failed}/{len(rho_all)}")
print(f"Compliance range (excluding failures): "
      f"[{compliance_all[compliance_all > 0].min():.2f}, "
      f"{compliance_all[compliance_all > 0].max():.2f}]")

np.save(SAVE_PATH, compliance_all)
print(f"Saved to {SAVE_PATH}")


"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import os

# pip install umap-learn
import umap

from evaluate_vae_report_com import TopologyVAE 

# ── config ────────────────────────────────────────────────────────────────
VAE_PATH   = "dataset/merged_vae_train_report/vae_best.pth"
RHO_PATHS  = [f"dataset/dataset_cantilever_sym6_mmc{i if i>0 else ''}/rho_smooth.npy"
               for i in range(10)]
PARAM_PATHS = [f"dataset/dataset_cantilever_sym6_mmc{i if i>0 else ''}/params.npy"
               for i in range(10)]
META_PATHS  = [f"dataset/dataset_cantilever_sym6_mmc{i if i>0 else ''}/metadata.npy"
               for i in range(10)]
OUT_DIR    = "analysis/latent_space"
LATENT_DIM = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────
def try_load(paths, key=None):
    arrays = []
    for p in paths:
        p = p.replace("mmc0", "mmc")  # first dataset has no number
        if os.path.exists(p):
            arr = np.load(p, allow_pickle=True)
            if key is not None and arr.dtype == object:
                arr = np.array([a[key] for a in arr])
            arrays.append(arr)
    return np.concatenate(arrays, axis=0)

print("Loading data...")
rho    = try_load(RHO_PATHS)           # [N, 64, 64]
params = try_load(PARAM_PATHS)         # [N, 15]
meta   = try_load(META_PATHS)          # [N] — compliance values or structured

# metadata may be structured array or plain float array
# adjust depending on how your metadata is stored:
compliance = np.load("dataset/compliance_all.npy")

volume_frac = rho.reshape(len(rho), -1).mean(axis=1)

print(f"  Loaded {len(rho)} designs")
print(f"  Compliance: [{compliance.min():.2f}, {compliance.max():.2f}]")
print(f"  Volume:     [{volume_frac.min():.3f}, {volume_frac.max():.3f}]")

# ── encode all designs ────────────────────────────────────────────────────
model = TopologyVAE(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
model.eval()

rho_tensor = torch.tensor(rho).unsqueeze(1).float()
all_mu = []

with torch.no_grad():
    for i in range(0, len(rho_tensor), BATCH_SIZE):
        batch = rho_tensor[i:i+BATCH_SIZE].to(DEVICE)
        mu, _ = model.encode(batch)
        all_mu.append(mu.cpu().numpy())

mu_all = np.concatenate(all_mu, axis=0)   # [N, 32]
np.save(os.path.join(OUT_DIR, "mu_all.npy"), mu_all)
print(f"Encoded {len(mu_all)} designs → μ shape {mu_all.shape}")


valid_mask = compliance > 0
print(f"Valid designs (FEM succeeded): {valid_mask.sum()}/{len(valid_mask)}")

# apply mask to everything
rho        = rho[valid_mask]
compliance = compliance[valid_mask]
params     = params[valid_mask]
volume_frac = rho.reshape(len(rho), -1).mean(axis=1)
mu_all     = mu_all[valid_mask]

# ── PCA ───────────────────────────────────────────────────────────────────
print("Running PCA...")
pca    = PCA(n_components=2)
pca_2d = pca.fit_transform(mu_all)
print(f"  Explained variance: {pca.explained_variance_ratio_}")

# ── UMAP ──────────────────────────────────────────────────────────────────
print("Running UMAP (this takes ~1 min for 10k points)...")
reducer = umap.UMAP(n_components=2, n_neighbors=15,
                    min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(mu_all)

# ── plotting helper ───────────────────────────────────────────────────────
def scatter(coords, color_vals, title, fname, cmap="plasma", log_scale=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    vals = np.log1p(color_vals) if log_scale else color_vals
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=vals, cmap=cmap, s=2, alpha=0.6, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar_label = f"log(1 + {title})" if log_scale else title
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

# ── colour by compliance and volume ───────────────────────────────────────
for prefix, coords in [("pca", pca_2d), ("umap", umap_2d)]:
    scatter(coords, compliance,  "Compliance (N·mm)", f"{prefix}_compliance.png",
            cmap="plasma_r")
    scatter(coords, volume_frac, "Volume fraction",   f"{prefix}_volume.png",
            cmap="viridis")

# ── colour by each beam geometric parameter ───────────────────────────────
# params shape: [N, 15] = 3 beams × 5 params (xc, yc, theta, L, t)
param_names = []
for beam_idx in range(3):
    for p in ["x_c", "y_c", "theta", "L", "t"]:
        param_names.append(f"beam{beam_idx+1}_{p}")

print("\nPlotting by beam parameters...")
for i, name in enumerate(param_names):
    for prefix, coords in [("umap", umap_2d)]:
        scatter(coords, params[:, i], name,
                f"{prefix}_{name}.png", cmap="coolwarm")

# ── PCA: show top loadings per PC ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for pc_idx, ax in enumerate(axes):
    loadings = pca.components_[pc_idx]   # [32]
    ax.bar(range(LATENT_DIM), loadings)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(f"PC{pc_idx+1} loading")
    ax.set_title(f"PC{pc_idx+1} ({100*pca.explained_variance_ratio_[pc_idx]:.1f}% var)")
    ax.axhline(0, color="k", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_loadings.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Done. All plots saved to", OUT_DIR)



from scipy.stats import pearsonr

# ── config ────────────────────────────────────────────────────────────────
OUT_DIR    = "analysis/latent_space"
LATENT_DIM = 32
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

mu_all     = np.load(os.path.join(OUT_DIR, "mu_all.npy"))   # [N, 32]


param_names = []
for beam_idx in range(3):
    for p in ["x_c", "y_c", "theta", "L", "t"]:
        param_names.append(f"beam{beam_idx+1}_{p}")

# stack all target variables together
targets      = np.column_stack([params, compliance, volume_frac])
target_names = param_names + ["compliance", "volume_frac"]

N_targets = len(target_names)
N_latent  = LATENT_DIM

# ── compute correlation matrix ────────────────────────────────────────────
print("Computing Pearson correlations...")
corr_matrix = np.zeros((N_latent, N_targets))
pval_matrix = np.zeros((N_latent, N_targets))

for z in range(N_latent):
    for t in range(N_targets):
        r, p = pearsonr(mu_all[:, z], targets[:, t])
        corr_matrix[z, t] = r
        pval_matrix[z, t] = p

np.save(os.path.join(OUT_DIR, "corr_matrix.npy"), corr_matrix)

# ── heatmap ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, N_targets * 0.6), 10))
im = ax.imshow(corr_matrix, cmap="RdBu_r",
               vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(N_targets))
ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(N_latent))
ax.set_yticklabels([f"z{i}" for i in range(N_latent)], fontsize=7)
ax.set_xlabel("Design parameter / objective")
ax.set_ylabel("Latent dimension")
ax.set_title("Pearson correlation: latent dims vs design parameters")

# mark statistically significant cells (p < 0.01) with a dot
sig = pval_matrix < 0.01
for zi in range(N_latent):
    for ti in range(N_targets):
        if sig[zi, ti]:
            ax.plot(ti, zi, "k.", markersize=2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "disentanglement_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved disentanglement_heatmap.png")

# ── top correlated pairs ──────────────────────────────────────────────────
abs_corr = np.abs(corr_matrix)
flat_idx = np.argsort(abs_corr.ravel())[::-1][:30]  # top 30

print("\nTop 30 (latent dim, parameter) correlations:")
print(f"{'Rank':<5} {'z_dim':<8} {'parameter':<25} {'r':>8} {'p':>12}")
print("-" * 60)
for rank, idx in enumerate(flat_idx):
    zi, ti = divmod(idx, N_targets)
    print(f"{rank+1:<5} z{zi:<7} {target_names[ti]:<25} "
          f"{corr_matrix[zi,ti]:>8.4f} {pval_matrix[zi,ti]:>12.2e}")

# ── per-parameter: which z_dim correlates most ───────────────────────────
print("\nBest latent dimension per parameter:")
for ti, name in enumerate(target_names):
    best_z = int(np.argmax(np.abs(corr_matrix[:, ti])))
    r_val  = corr_matrix[best_z, ti]
    print(f"  {name:<25} → z{best_z:<3}  r={r_val:+.4f}")

# ── latent traversal for top-correlated dims ─────────────────────────────
# pick the z_dim most correlated with compliance and with theta of beam 1
compliance_col = target_names.index("compliance")
theta1_col     = target_names.index("beam1_theta")

top_compliance_z = int(np.argmax(np.abs(corr_matrix[:, compliance_col])))
top_theta_z      = int(np.argmax(np.abs(corr_matrix[:, theta1_col])))

print(f"\nTop compliance z: z{top_compliance_z}  "
      f"r={corr_matrix[top_compliance_z, compliance_col]:+.4f}")
print(f"Top theta z:      z{top_theta_z}  "
      f"r={corr_matrix[top_theta_z, theta1_col]:+.4f}")

# save these indices for script 3
np.save(os.path.join(OUT_DIR, "top_z_indices.npy"),
        np.array([top_compliance_z, top_theta_z]))

import torch.nn.functional as F
import glob

# ── config ────────────────────────────────────────────────────────────────
VAE_PATH         = "dataset/merged_vae_train_report/vae_best.pth"
OUT_DIR          = "analysis/latent_space"
CMAES_RESULT_DIR = "results"   # folder with per-gen .npz files
LATENT_DIM       = 32
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 256
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# reload encoded means and raw data
mu_all     = np.load(os.path.join(OUT_DIR, "mu_all.npy"))

#from latent_space_analysis import rho, compliance, volume_frac

# ── per-sample reconstruction loss ───────────────────────────────────────
model = TopologyVAE(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
model.eval()

rho_tensor  = torch.tensor(rho).unsqueeze(1).float()
recon_losses = []

print("Computing per-sample reconstruction loss...")
with torch.no_grad():
    for i in range(0, len(rho_tensor), BATCH_SIZE):
        batch = rho_tensor[i:i+BATCH_SIZE].to(DEVICE)
        recon, mu, logvar, c_pred = model(batch)
        # per-sample BCE: mean over pixels, keep batch dimension
        loss = F.binary_cross_entropy(
            recon, batch, reduction="none"
        ).mean(dim=[1, 2, 3])
        recon_losses.append(loss.cpu().numpy())

recon_losses = np.concatenate(recon_losses)   # [N]
np.save(os.path.join(OUT_DIR, "recon_losses.npy"), recon_losses)

# ── plot 1: reconstruction loss vs compliance ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(compliance, recon_losses,
                c=volume_frac, cmap="viridis",
                s=3, alpha=0.5, rasterized=True)
plt.colorbar(sc, ax=ax, label="Volume fraction")
ax.set_xlabel("Compliance (N·mm)")
ax.set_ylabel("Reconstruction loss (BCE)")
ax.set_title("Reconstruction difficulty vs compliance\n"
             "(colour = volume fraction)")

# add a running median line
sort_idx = np.argsort(compliance)
comp_s   = compliance[sort_idx]
loss_s   = recon_losses[sort_idx]
window   = max(1, len(comp_s) // 50)
running_med = np.array([
    np.median(loss_s[max(0, i-window):i+window])
    for i in range(len(loss_s))
])
ax.plot(comp_s, running_med, color="red", linewidth=1.5,
        label="Running median")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "recon_vs_compliance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved recon_vs_compliance.png")

# ── plot 2: where are low/high compliance designs in UMAP ────────────────
# re-fit UMAP (or load if you saved it — better to save in script 1)
print("Re-fitting UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15,
                    min_dist=0.1, random_state=42)
umap_2d = reducer.fit_transform(mu_all)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: coloured by compliance
sc1 = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1],
                       c=compliance, cmap="plasma_r",
                       s=2, alpha=0.5, rasterized=True)
plt.colorbar(sc1, ax=axes[0], label="Compliance (N·mm)")
axes[0].set_title("UMAP — coloured by compliance")
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

# right: coloured by reconstruction loss
sc2 = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1],
                       c=recon_losses, cmap="hot_r",
                       s=2, alpha=0.5, rasterized=True)
plt.colorbar(sc2, ax=axes[1], label="Reconstruction loss")
axes[1].set_title("UMAP — coloured by reconstruction loss")
axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_compliance_vs_recon.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved umap_compliance_vs_recon.png")

# ── plot 3: CMA-ES trajectory on UMAP ────────────────────────────────────
# looks for any .npz in CMAES_RESULT_DIR with a 'best_z' key
npz_files = sorted(glob.glob(os.path.join(CMAES_RESULT_DIR, "**/*.npz"),
                              recursive=True))
if not npz_files:
    # also try flat dir
    npz_files = sorted(glob.glob("cmaes_result_*.npz"))

if not npz_files:
    print("No CMA-ES .npz files found — skipping trajectory plot.")
    print("  Expected: npz files with a 'best_z' key in", CMAES_RESULT_DIR)
else:
    traj_zs = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if "best_z" in d:
            traj_zs.append(d["best_z"])

    if traj_zs:
        traj_zs  = np.stack(traj_zs)          # [n_snapshots, 32]
        traj_2d  = reducer.transform(traj_zs)  # project into same UMAP space

        fig, ax = plt.subplots(figsize=(8, 7))

        # background: all training designs coloured by compliance
        sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                        c=compliance, cmap="plasma_r",
                        s=2, alpha=0.3, rasterized=True,
                        zorder=1)
        plt.colorbar(sc, ax=ax, label="Compliance (N·mm)")

        # trajectory: line + coloured dots (early=blue, late=red)
        n_steps = len(traj_2d)
        colors  = cm.cool(np.linspace(0, 1, n_steps))
        ax.plot(traj_2d[:, 0], traj_2d[:, 1],
                color="white", linewidth=1.5, zorder=2, alpha=0.8)
        ax.scatter(traj_2d[:, 0], traj_2d[:, 1],
                   c=colors, s=40, edgecolors="k", linewidths=0.5,
                   zorder=3, label="CMA-ES steps")

        # mark start and end
        ax.scatter(*traj_2d[0],  s=120, marker="*", color="cyan",
                   edgecolors="k", linewidths=0.8, zorder=4, label="Start")
        ax.scatter(*traj_2d[-1], s=120, marker="D", color="lime",
                   edgecolors="k", linewidths=0.8, zorder=4, label="Best found")

        ax.set_title("CMA-ES trajectory in UMAP latent space\n"
                     "(background = training designs, colour = compliance)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "umap_cmaes_trajectory.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved umap_cmaes_trajectory.png  ({n_steps} trajectory points)")
    else:
        print("No 'best_z' keys found in npz files.")

print("\nAll done.")


