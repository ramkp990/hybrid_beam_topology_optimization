"""
dataset_analysis.py
====================
Comprehensive dataset analysis for the cantilever topology optimisation dataset.
Produces figures for the "Dataset" section of the report.

Sections
--------
1.  Basic statistics
2.  Parameter distributions (per-beam and aggregate)
3.  Compliance distribution and outlier inspection
4.  Volume-fraction distribution and area-fraction check
5.  Compliance vs volume-fraction (the key design trade-off)
6.  Inter-parameter correlation within a design
7.  Per-parameter vs compliance (15 scatter plots)
8.  Spatial density map  (mean / std topology across all designs)
9.  Cross-dataset consistency
10. Acceptance-rate proxy (compliance CDF + threshold line)
11. Design diversity (pairwise-distance histogram)
12. Representative design gallery (low / mid / high compliance)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, ks_2samp, gaussian_kde
from sklearn.metrics import pairwise_distances_chunked
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
DATASET_DIRS = [
    f"dataset/dataset_cantilever_sym6_mmc{'' if i == 0 else i}"
    for i in range(10)
]
COMPLIANCE_PATH = "dataset/compliance_all.npy"
OUT_DIR = "analysis/dataset"
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_NAMES = []
for b in range(1, 4):
    for p in ["x_c", "y_c", "θ", "L", "t"]:
        PARAM_NAMES.append(f"beam{b}_{p}")

COMPLIANCE_THRESHOLD = 80.0   # filter used during generation
GRID = 64

# ── helpers ────────────────────────────────────────────────────────────────
def savefig(name, fig=None, dpi=150):
    path = os.path.join(OUT_DIR, name)
    (fig or plt).savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓  {name}")

def load_all():
    rho_list, params_list, ds_labels = [], [], []
    for idx, d in enumerate(DATASET_DIRS):
        rp = os.path.join(d, "rho_smooth.npy")
        pp = os.path.join(d, "params.npy")
        if not (os.path.exists(rp) and os.path.exists(pp)):
            print(f"  SKIP (missing) {d}")
            continue
        r = np.load(rp);  params_list.append(np.load(pp))
        rho_list.append(r)
        ds_labels.extend([idx] * len(r))
        print(f"  Loaded {len(r):>5} designs from {os.path.basename(d)}")
    rho    = np.concatenate(rho_list,    axis=0)
    params = np.concatenate(params_list, axis=0)
    labels = np.array(ds_labels, dtype=np.int32)
    return rho, params, labels

# ══════════════════════════════════════════════════════════════════════════
print("\n── Loading data ─────────────────────────────────────────────────")
rho, params, ds_labels = load_all()
compliance = np.load(COMPLIANCE_PATH)
assert len(rho) == len(compliance), "Size mismatch — re-run precompute_compliance.py"

# mask failed FEM runs
valid = compliance > 0
rho        = rho[valid]
params     = params[valid]
compliance = compliance[valid]
ds_labels  = ds_labels[valid]

volume_frac = rho.reshape(len(rho), -1).mean(axis=1)
N = len(rho)
print(f"\nTotal valid designs : {N:,}")
print(f"Compliance          : [{compliance.min():.2f}, {compliance.max():.2f}]  "
      f"mean={compliance.mean():.2f}  std={compliance.std():.2f}")
print(f"Volume fraction     : [{volume_frac.min():.3f}, {volume_frac.max():.3f}]  "
      f"mean={volume_frac.mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 1. Summary statistics table (printed + saved as txt)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 1. Summary statistics ────────────────────────────────────────")
lines = []
lines.append(f"{'Variable':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Median':>8}")
lines.append("-" * 66)
for i, name in enumerate(PARAM_NAMES):
    v = params[:, i]
    lines.append(f"{name:<22} {v.min():>8.4f} {v.max():>8.4f} "
                 f"{v.mean():>8.4f} {v.std():>8.4f} {np.median(v):>8.4f}")
lines.append("-" * 66)
for name, v in [("compliance", compliance), ("volume_frac", volume_frac)]:
    lines.append(f"{name:<22} {v.min():>8.4f} {v.max():>8.4f} "
                 f"{v.mean():>8.4f} {v.std():>8.4f} {np.median(v):>8.4f}")
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT_DIR, "summary_stats.txt"), "w") as f:
    f.write(summary)

# ══════════════════════════════════════════════════════════════════════════
# 2. Parameter distributions — 3×5 grid (one subplot per param)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 2. Parameter distributions ───────────────────────────────────")
fig, axes = plt.subplots(3, 5, figsize=(18, 9))
axes = axes.flatten()
beam_colors = ["#4C72B0", "#DD8452", "#55A868"]   # blue / orange / green

for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
    beam_idx = i // 5
    ax.hist(params[:, i], bins=40, color=beam_colors[beam_idx],
            alpha=0.80, edgecolor="white", linewidth=0.4)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("value", fontsize=8)
    ax.set_ylabel("count",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle("Distribution of all 15 beam parameters across the dataset",
             fontsize=12, y=1.01)
plt.tight_layout()
savefig("param_distributions.png")

# ── Aggregate per-parameter-type comparison (theta / L / t across beams) ─
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
param_types = {"θ (deg)": [2, 7, 12], "L (half-length)": [3, 8, 13], "t (thickness)": [4, 9, 14]}
for ax, (label, indices) in zip(axes, param_types.items()):
    for j, idx in enumerate(indices):
        kde_x = np.linspace(params[:, idx].min(), params[:, idx].max(), 300)
        kde = gaussian_kde(params[:, idx], bw_method=0.15)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.35, color=beam_colors[j],
                        label=f"Beam {j+1}")
        ax.plot(kde_x, kde(kde_x), color=beam_colors[j], linewidth=1.5)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle("KDE of shared parameter types across all three beams", fontsize=11)
plt.tight_layout()
savefig("param_kde_by_beam.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. Compliance distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n── 3. Compliance distribution ───────────────────────────────────")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histogram
axes[0].hist(compliance, bins=60, color="#4C72B0", edgecolor="white",
             linewidth=0.4, alpha=0.85)
axes[0].axvline(COMPLIANCE_THRESHOLD, color="crimson", linewidth=1.5,
                linestyle="--", label=f"Filter threshold ({COMPLIANCE_THRESHOLD})")
axes[0].set_xlabel("Compliance (N·mm)")
axes[0].set_ylabel("Count")
axes[0].set_title("Compliance distribution")
axes[0].legend(fontsize=8)
axes[0].grid(True, linewidth=0.3, alpha=0.5)

# Log-scale histogram (reveals tail structure)
axes[1].hist(compliance, bins=60, color="#4C72B0", edgecolor="white",
             linewidth=0.4, alpha=0.85, log=True)
axes[1].set_xlabel("Compliance (N·mm)")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title("Compliance distribution (log-scale count)")
axes[1].grid(True, linewidth=0.3, alpha=0.5)

# Empirical CDF
x_sorted = np.sort(compliance)
cdf = np.arange(1, N + 1) / N
axes[2].plot(x_sorted, cdf, color="#4C72B0", linewidth=1.5)
axes[2].axvline(COMPLIANCE_THRESHOLD, color="crimson", linewidth=1.5,
                linestyle="--", label=f"Threshold = {COMPLIANCE_THRESHOLD}")
frac_below = (compliance < COMPLIANCE_THRESHOLD).mean()
axes[2].axhline(frac_below, color="gray", linewidth=1, linestyle=":",
                label=f"{100*frac_below:.1f}% below threshold")
axes[2].set_xlabel("Compliance (N·mm)")
axes[2].set_ylabel("Cumulative fraction")
axes[2].set_title("Empirical CDF — effective acceptance rate")
axes[2].legend(fontsize=8)
axes[2].grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle("Compliance analysis", fontsize=12)
plt.tight_layout()
savefig("compliance_distribution.png")

pct = np.percentile(compliance, [10, 25, 50, 75, 90])
print(f"  Percentiles  [10,25,50,75,90]: {pct}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Volume-fraction distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n── 4. Volume-fraction distribution ─────────────────────────────")
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(volume_frac, bins=50, color="#55A868", edgecolor="white",
        linewidth=0.4, alpha=0.85)
ax.axvline(0.10, color="gray", linestyle="--", linewidth=1.2, label="Min 10%")
ax.axvline(0.50, color="gray", linestyle="--", linewidth=1.2, label="Max 50%")
ax.set_xlabel("Volume fraction")
ax.set_ylabel("Count")
ax.set_title("Volume fraction distribution across all valid designs")
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.3, alpha=0.5)
plt.tight_layout()
savefig("volume_fraction_dist.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. Compliance vs volume fraction — the fundamental design trade-off
# ══════════════════════════════════════════════════════════════════════════
print("\n── 5. Compliance vs volume fraction ─────────────────────────────")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter (density-coloured via 2D histogram)
h, xe, ye = np.histogram2d(volume_frac, compliance, bins=60)
xc = 0.5 * (xe[:-1] + xe[1:])
yc = 0.5 * (ye[:-1] + ye[1:])
axes[0].pcolormesh(xc, yc, h.T, cmap="Blues", norm=LogNorm())
axes[0].set_xlabel("Volume fraction")
axes[0].set_ylabel("Compliance (N·mm)")
axes[0].set_title("Compliance vs volume fraction\n(2D density, log-colour)")

# Pearson r
r, p = pearsonr(volume_frac, compliance)
axes[0].text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=axes[0].transAxes,
             fontsize=9, va="top")

# Running median in 20 percentile bins
vf_bins = np.percentile(volume_frac, np.linspace(0, 100, 21))
bin_idx = np.digitize(volume_frac, vf_bins) - 1
bin_idx = np.clip(bin_idx, 0, 19)
bin_med = [np.median(compliance[bin_idx == b]) for b in range(20)]
bin_ctr = 0.5 * (vf_bins[:-1] + vf_bins[1:])
axes[0].plot(bin_ctr, bin_med, color="crimson", linewidth=2,
             label="Running median", zorder=5)
axes[0].legend(fontsize=8)

# Box-plot by quartile of volume fraction
quartile = np.digitize(volume_frac,
                       np.percentile(volume_frac, [25, 50, 75])).astype(int)
data_bp = [compliance[quartile == q] for q in range(4)]
axes[1].boxplot(data_bp, labels=["Q1\n(sparse)", "Q2", "Q3", "Q4\n(dense)"],
                patch_artist=True,
                boxprops=dict(facecolor="#AED6F1", color="#2471A3"),
                medianprops=dict(color="crimson", linewidth=2),
                whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
axes[1].set_xlabel("Volume-fraction quartile")
axes[1].set_ylabel("Compliance (N·mm)")
axes[1].set_title("Compliance per volume-fraction quartile")
axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.5)

plt.suptitle("Compliance vs volume fraction trade-off", fontsize=12)
plt.tight_layout()
savefig("compliance_vs_volume.png")
print(f"  Pearson r(volume, compliance) = {r:.4f}  p={p:.2e}")

# ══════════════════════════════════════════════════════════════════════════
# 6. Inter-parameter correlation heatmap (15×15)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 6. Inter-parameter correlation heatmap ───────────────────────")
corr = np.corrcoef(params.T)   # [15, 15]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.04)
ticks = np.arange(15)
ax.set_xticks(ticks); ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=7)
ax.set_yticks(ticks); ax.set_yticklabels(PARAM_NAMES, fontsize=7)
ax.set_title("Inter-parameter Pearson correlation\n(independent sampling → near-zero off-diagonal)",
             fontsize=10)
# Annotate cells
for i in range(15):
    for j in range(15):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                fontsize=5, color="black" if abs(corr[i,j]) < 0.6 else "white")
plt.tight_layout()
savefig("param_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. Per-parameter vs compliance scatter (show top-5 by |r|)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 7. Param–compliance correlations ─────────────────────────────")
r_vals = []
for i in range(15):
    r_c, _ = pearsonr(params[:, i], compliance)
    r_vals.append(r_c)
r_vals = np.array(r_vals)
top5 = np.argsort(np.abs(r_vals))[::-1][:5]
print("  Top-5 params by |r| with compliance:")
for rank, idx in enumerate(top5):
    print(f"    {rank+1}. {PARAM_NAMES[idx]:<20}  r={r_vals[idx]:+.4f}")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, idx in zip(axes, top5):
    x = params[:, idx]
    # 2D density for dense scatter
    h2, xe2, ye2 = np.histogram2d(x, compliance, bins=50)
    xc2 = 0.5 * (xe2[:-1] + xe2[1:])
    yc2 = 0.5 * (ye2[:-1] + ye2[1:])
    ax.pcolormesh(xc2, yc2, h2.T, cmap="Blues", norm=LogNorm())
    ax.set_xlabel(PARAM_NAMES[idx], fontsize=9)
    ax.set_ylabel("Compliance", fontsize=8)
    ax.set_title(f"r = {r_vals[idx]:+.3f}", fontsize=9, color="crimson")
    ax.grid(True, linewidth=0.3, alpha=0.4)

plt.suptitle("Top-5 parameters most correlated with compliance\n(2D density)", fontsize=11)
plt.tight_layout()
savefig("top5_params_vs_compliance.png")

# Full 15-panel version
fig, axes = plt.subplots(3, 5, figsize=(20, 11))
for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes.flatten())):
    h2, xe2, ye2 = np.histogram2d(params[:, i], compliance, bins=40)
    xc2 = 0.5 * (xe2[:-1] + xe2[1:])
    yc2 = 0.5 * (ye2[:-1] + ye2[1:])
    ax.pcolormesh(xc2, yc2, h2.T, cmap="Blues", norm=LogNorm())
    ax.set_title(f"{name}\nr={r_vals[i]:+.3f}", fontsize=8,
                 color="crimson" if abs(r_vals[i]) > 0.15 else "black")
    ax.set_xlabel("param value", fontsize=7)
    ax.set_ylabel("compliance", fontsize=7)
    ax.tick_params(labelsize=6)
plt.suptitle("All parameters vs compliance (2D density, log-colour)", fontsize=12)
plt.tight_layout()
savefig("all_params_vs_compliance.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. Spatial density maps
# ══════════════════════════════════════════════════════════════════════════
print("\n── 8. Spatial density maps ──────────────────────────────────────")
mean_rho = rho.mean(axis=0)
std_rho  = rho.std(axis=0)

# Split by compliance tertile
t33, t66 = np.percentile(compliance, [33, 66])
mask_lo  = compliance <= t33
mask_mid = (compliance > t33) & (compliance <= t66)
mask_hi  = compliance > t66

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
maps = [
    (mean_rho,              "Mean density (all designs)",       "viridis"),
    (std_rho,               "Std of density (uncertainty)",     "plasma"),
    (rho[mask_lo].mean(0),  f"Mean density — low compliance\n(≤{t33:.1f})",  "RdYlGn"),
    (rho[mask_mid].mean(0), f"Mean density — mid compliance",               "RdYlGn"),
    (rho[mask_hi].mean(0),  f"Mean density — high compliance\n(>{t66:.1f})", "RdYlGn"),
    (rho[mask_hi].mean(0) - rho[mask_lo].mean(0),
                             "Δ mean density (high − low comp.)",           "RdBu_r"),
]
for ax, (m, title, cmap) in zip(axes.flatten(), maps):
    vmin = -abs(m).max() if "RdBu" in cmap else 0
    im = ax.imshow(m, cmap=cmap, origin="upper",
                   vmin=vmin, vmax=abs(m).max() if "RdBu" in cmap else m.max())
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
    # Mark load point and BCs
    ax.plot(GRID - 1, GRID // 2, "r*", markersize=10, label="Load")
    ax.axvline(0, color="cyan", linewidth=1.5, linestyle="--", label="Fixed BC")
    if ax == axes.flatten()[0]:
        ax.legend(fontsize=7, loc="upper right")

plt.suptitle("Spatial material density maps", fontsize=13, y=1.01)
plt.tight_layout()
savefig("spatial_density_maps.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. Cross-dataset consistency (KS-test on compliance per dataset pair)
# ══════════════════════════════════════════════════════════════════════════
print("\n── 9. Cross-dataset consistency ─────────────────────────────────")
unique_ds = np.unique(ds_labels)
n_ds = len(unique_ds)

# Compliance per dataset + KDE plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, n_ds))
ds_compliance = {}
for ds_idx, color in zip(unique_ds, colors):
    c = compliance[ds_labels == ds_idx]
    ds_compliance[ds_idx] = c
    x = np.linspace(c.min(), c.max(), 300)
    kde = gaussian_kde(c, bw_method=0.2)
    label = f"mmc{'9' if ds_idx == 0 else ds_idx}"
    axes[0].plot(x, kde(x), color=color, linewidth=1.5, label=label)
    axes[0].axvline(np.median(c), color=color, linewidth=0.8, linestyle=":")

axes[0].set_xlabel("Compliance (N·mm)")
axes[0].set_ylabel("Density")
axes[0].set_title("Compliance KDE per sub-dataset")
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, linewidth=0.3, alpha=0.5)

# KS-test matrix
ks_matrix = np.zeros((n_ds, n_ds))
for i, di in enumerate(unique_ds):
    for j, dj in enumerate(unique_ds):
        if i != j:
            stat, _ = ks_2samp(ds_compliance[di], ds_compliance[dj])
            ks_matrix[i, j] = stat

im = axes[1].imshow(ks_matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=axes[1], label="KS statistic")
tick_labels = [f"mmc{'' if k == 0 else k}" for k in unique_ds]
axes[1].set_xticks(range(n_ds)); axes[1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(n_ds)); axes[1].set_yticklabels(tick_labels, fontsize=8)
axes[1].set_title("Pairwise KS statistic on compliance\n(0 = identical, 1 = maximally different)")
for i in range(n_ds):
    for j in range(n_ds):
        axes[1].text(j, i, f"{ks_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if ks_matrix[i,j] < 0.5 else "white")

plt.suptitle("Cross-dataset consistency", fontsize=12)
plt.tight_layout()
savefig("cross_dataset_consistency.png")

# Print volume fractions per dataset
print("  Volume fraction median per dataset:")
for ds_idx in unique_ds:
    vf = volume_frac[ds_labels == ds_idx]
    n  = (ds_labels == ds_idx).sum()
    print(f"    mmc{ds_idx if ds_idx>0 else ''}  n={n}  "
          f"vf_med={np.median(vf):.3f}  comp_med={np.median(ds_compliance[ds_idx]):.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 10. Acceptance rate proxy — compliance CDF with threshold
# ══════════════════════════════════════════════════════════════════════════
print("\n── 10. Acceptance rate proxy ────────────────────────────────────")
# Compliance values are post-filter; we can only show the accepted distribution.
# Use the CDF to show how tight the threshold is.
fig, ax = plt.subplots(figsize=(8, 4))
x_sorted = np.sort(compliance)
cdf = np.arange(1, N + 1) / N
ax.plot(x_sorted, cdf * 100, color="#4C72B0", linewidth=1.8)
ax.axvline(COMPLIANCE_THRESHOLD, color="crimson", linewidth=1.5,
           linestyle="--", label=f"Threshold = {COMPLIANCE_THRESHOLD}")
pct_below = 100 * (compliance < COMPLIANCE_THRESHOLD).mean()
ax.axhline(pct_below, color="gray", linewidth=1, linestyle=":",
           label=f"{pct_below:.1f}% below threshold")
ax.set_xlabel("Compliance (N·mm)")
ax.set_ylabel("Cumulative % of accepted designs")
ax.set_title("Accepted-design compliance CDF\n(all designs already passed the threshold filter)")
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.3, alpha=0.5)
plt.tight_layout()
savefig("acceptance_cdf.png")

# ══════════════════════════════════════════════════════════════════════════
# 11. Design diversity — pairwise distance distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n── 11. Design diversity ─────────────────────────────────────────")
# Use param space (cheap) for the full dataset
# and rho space (pixel-level) for a sample
SAMPLE = min(2000, N)
idx_s = np.random.RandomState(42).choice(N, SAMPLE, replace=False)

print(f"  Computing pairwise distances in param space (sample={SAMPLE})...")
param_sample = params[idx_s]
# Normalise each column to [0,1] first
param_norm = (param_sample - params.min(0)) / (params.max(0) - params.min(0) + 1e-9)
dists_param = []
for chunk in pairwise_distances_chunked(param_norm, metric="euclidean", n_jobs=-1, working_memory=512):
    for row in chunk:
        dists_param.extend(row[row > 0].tolist())
dists_param = np.array(dists_param, dtype=np.float32)

print(f"  Computing pairwise distances in pixel space (sample={SAMPLE})...")
rho_sample = rho[idx_s].reshape(SAMPLE, -1)
dists_rho = []
for chunk in pairwise_distances_chunked(rho_sample, metric="euclidean", n_jobs=-1, working_memory=512):
    for row in chunk:
        dists_rho.extend(row[row > 0].tolist())
dists_rho = np.array(dists_rho, dtype=np.float32)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(dists_param, bins=60, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("L2 distance (normalised param space)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Pairwise distances — parameter space\n(sample n={SAMPLE:,})")
axes[0].grid(True, linewidth=0.3, alpha=0.5)

axes[1].hist(dists_rho, bins=60, color="#DD8452", alpha=0.8, edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("L2 distance (pixel space)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Pairwise distances — pixel/density space\n(sample n={SAMPLE:,})")
axes[1].grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle("Dataset diversity (pairwise L2 distances)", fontsize=12)
plt.tight_layout()
savefig("design_diversity.png")
print(f"  Param-space   distances: mean={dists_param.mean():.3f}  std={dists_param.std():.3f}")
print(f"  Pixel-space   distances: mean={dists_rho.mean():.3f}  std={dists_rho.std():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 12. Representative design gallery
# ══════════════════════════════════════════════════════════════════════════
print("\n── 12. Representative design gallery ────────────────────────────")
n_show = 5
pct_vals = [5, 25, 50, 75, 95]   # compliance percentiles to showcase
compliance_pcts = np.percentile(compliance, pct_vals)

fig, axes = plt.subplots(2, n_show, figsize=(18, 6))
for col, (pv, cv) in enumerate(zip(pct_vals, compliance_pcts)):
    closest = int(np.argmin(np.abs(compliance - cv)))
    ax_rho = axes[0, col]
    ax_phi = axes[1, col]

    ax_rho.imshow(rho[closest], cmap="gray", origin="upper", vmin=0, vmax=1)
    ax_rho.set_title(f"P{pv}  C={compliance[closest]:.1f}", fontsize=9)
    ax_rho.axis("off")
    # Mark load point
    ax_rho.plot(GRID - 1, GRID // 2, "r*", markersize=8)

    # Binary version for visual clarity
    ax_phi.imshow(rho[closest] > 0.5, cmap="binary", origin="upper")
    ax_phi.set_title(f"V={volume_frac[closest]:.3f}", fontsize=9)
    ax_phi.axis("off")
    ax_phi.plot(GRID - 1, GRID // 2, "r*", markersize=8)

axes[0, 0].set_ylabel("Density (rho)", fontsize=9)
axes[1, 0].set_ylabel("Binary (rho>0.5)", fontsize=9)
plt.suptitle("Representative designs at compliance percentiles P5 → P95\n"
             "(top: smooth density, bottom: binarised,  ★ = load point)", fontsize=11)
plt.tight_layout()
savefig("design_gallery.png")

print(f"\n{'─'*55}")
print(f" All figures saved to:  {OUT_DIR}/")
print(f"{'─'*55}")