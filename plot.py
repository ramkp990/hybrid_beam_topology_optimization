import os
import numpy as np
import matplotlib.pyplot as plt

REPORT_DIR = "dataset/merged_vae_train_report"

recon = np.load(os.path.join(REPORT_DIR, "test_recon.npy"), allow_pickle=True)
diagnostics = np.load(os.path.join(REPORT_DIR, "test_diagnostics.npy"), allow_pickle=True)

os.makedirs(os.path.join(REPORT_DIR, "post_plots"), exist_ok=True)
OUT = os.path.join(REPORT_DIR, "post_plots")


# -------------------------------------------------
# helper: add black border around subplot
# -------------------------------------------------
def add_subplot_border(ax, lw=2):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor("black")


# -------------------------------------------------
# 1) RANDOM RECONSTRUCTION GRID
# -------------------------------------------------
print("Creating reconstruction grid...")

N_SHOW = 12
indices = np.random.choice(len(recon), N_SHOW, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.ravel()

for i, idx in enumerate(indices):
    rho = recon[idx].squeeze()

    # FIX ROTATION HERE  ✅
    rho = np.rot90(rho, k=1)

    axes[i].imshow(rho, cmap="gray_r", origin="lower")
    axes[i].set_title(f"Sample {idx}", fontweight="bold")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    add_subplot_border(axes[i])

plt.suptitle("Random VAE Reconstructions (Test Set)", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT, "reconstruction_grid.png"), dpi=300)
plt.close()



# -------------------------------------------------
# 2) VIOLATION EXAMPLES (FIXED ROTATION)
# -------------------------------------------------
# -------------------------------------------------
# 2) VIOLATION EXAMPLES (ROTATE 90° ANTICLOCKWISE)
# -------------------------------------------------
print("Creating violation plots...")

violation_types = {}
for i, d in enumerate(diagnostics):
    v = d["violation_type"]
    violation_types.setdefault(v, []).append(i)

for vtype, idxs in violation_types.items():
    if vtype == "feasible":
        continue

    show = idxs[:6]
    n = len(show)

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, show):
        rho = recon[idx].squeeze()

        # 🔁 FIX ROTATION → 90° ANTICLOCKWISE
        rho = np.flipud(rho)   # flip vertically to match FEM orientation


        area = diagnostics[idx]["area_fraction"]

        ax.imshow(rho, cmap="gray_r", origin="lower")
        ax.set_title(f"{vtype}\nArea={area:.2f}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Border around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    plt.suptitle(f"Violation examples: {vtype}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(os.path.join(OUT, f"violations_{vtype}.png"), dpi=300)
    plt.close()