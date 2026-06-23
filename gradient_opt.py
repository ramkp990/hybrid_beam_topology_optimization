import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from fem_code import fem_physical_compliance, is_feasible, element_stiffness_plane_stress


# Config

GRID          = 64
LATENT_DIM    = 32
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50
MAX_AREA_FRAC_P2 = 0.20  

TOTAL_STEPS   = 400         # steps 67–199 → Phase 2
PHASE1_STEPS  = 133 #TOTAL_STEPS // 3  # first 1/3 steps → Phase 1

E_SOLID = 2.1e5
E_VOID  = 1e-3 * E_SOLID
NU      = 0.3
P_SIMP  = 3.0


# Adjoint sensitivity  dC/dρ

def compute_compliance_and_sensitivity(rho_np: np.ndarray):
    """
    Run FEM, then compute adjoint sensitivity dC/dρ analytically.

    For SIMP:  C = f^T u
    dC/dρ_e  = -p * ρ_e^(p-1) * (E_solid - E_void) * u_e^T Ke_unit u_e
    Each nodal value is the sum of contributions from its 4 surrounding elements.

    Returns
    -------
    compliance : float
    dC_drho    : np.ndarray shape (64, 64)
    """
    compliance, u = fem_physical_compliance(rho_np)

    ny, nx     = rho_np.shape
    nel_y      = ny - 1
    nel_x      = nx - 1
    xe         = 20.0 / nel_x
    ye         = 10.0 / nel_y

    Ke_unit    = element_stiffness_plane_stress(1.0, NU, xe, ye)
    dC_drho    = np.zeros((ny, nx), dtype=np.float32)

    for ey in range(nel_y):
        for ex in range(nel_x):
            # average nodal densities → elemental density
            rho_e = (rho_np[ey,   ex]   + rho_np[ey,   ex+1] +
                     rho_np[ey+1, ex+1] + rho_np[ey+1, ex  ]) / 4.0

            # element DOFs
            nids = [ey*nx+ex, ey*nx+ex+1, (ey+1)*nx+ex+1, (ey+1)*nx+ex]
            dofs = []
            for n in nids:
                dofs.extend([2*n, 2*n+1])

            u_e = u[dofs]

            # dC/dE_e  (adjoint)
            dC_dE  = -float(u_e @ Ke_unit @ u_e)
            # dE_e/dρ_e  (SIMP derivative)
            dE_drho = P_SIMP * max(rho_e, 1e-6)**(P_SIMP-1) * (E_SOLID - E_VOID)

            # distribute equally to 4 corner nodes
            node_sens = dC_dE * dE_drho / 4.0
            for (iy, ix) in [(ey, ex), (ey, ex+1), (ey+1, ex+1), (ey+1, ex)]:
                dC_drho[iy, ix] += node_sens

    return compliance, dC_drho



# Differentiable penalty terms

def volume_penalties1(rho):
    """Lower-bound and upper-bound volume fraction penalties."""
    vol  = rho.mean()
    low  = F.relu(MIN_AREA_FRAC - vol)   # > 0 if below minimum
    high = F.relu(vol - MAX_AREA_FRAC)   # > 0 if above maximum
    return low, high

def volume_penalties(rho, max_frac):
    vol  = rho.mean()
    low  = F.relu(MIN_AREA_FRAC - vol)
    high = F.relu(vol - max_frac)
    return low, high

def connectivity_penalty(rho):
    """
    Soft proxy for structural connectivity:
      - material must exist on the left edge (support)
      - material must exist at the right-centre node (load point)
      - binary term discourages intermediate densities (gray)
    """
    support  = F.relu(0.5 - rho[:, 0].mean())
    load_pt  = F.relu(0.5 - rho[GRID//2, GRID-1])
    binary   = (rho * (1.0 - rho)).mean()
    return support + load_pt + 0.5 * binary

def plot_design(design, output_dir, step, filename):
    """Plot and save the optimized design."""
    plt.figure(figsize=(6, 6))
    plt.imshow(design.squeeze(), cmap='gray_r', origin='lower')
    plt.title(f'Optimized Design - Step {step}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()



# Main optimisation loop

def optimize_latent(model,
                    z_init=None,
                    n_steps=TOTAL_STEPS,
                    lr=0.05,
                    save_dir="latent_opt_results"):


    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # ── initialise z ──
    if z_init is None:
        z = torch.randn(1, model.latent_dim, device=DEVICE) * 0.5
    else:
        z = z_init.clone().to(DEVICE)
    z = z.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.1
    )

    history = {"compliance": [], "vol": [], "feasible": [], "loss": []}
    best_compliance = float("inf")
    best_rho        = None
    best_step       = -1
    COMPLIANCE_SCALE = 100.0


    for step in range(n_steps):
        optimizer.zero_grad()
        #print("z BEFORE step:")
        #print(z.detach().cpu().numpy())
        # ── decode ──
        rho = model.decode(z).squeeze()          # (64,64)  grad enabled

        # ── FEM + adjoint (numpy, no grad) ──
        rho_np = rho.detach().cpu().numpy().astype(np.float64)
        compliance, dC_drho = compute_compliance_and_sensitivity(rho_np)


        dC_drho_t = torch.tensor(dC_drho, dtype=torch.float32, device=DEVICE)

        # ── linearised compliance (exact gradient w.r.t. z via chain rule) ──
        compliance_loss = (dC_drho_t.detach() * rho).sum()
        #print(f"Step {step:03d} | Compliance = {compliance:.2f}")
        # ── phase-dependent penalties ──
        #vol_low, vol_high = volume_penalties(rho)


        # if step < PHASE1_STEPS:
        #     # Phase 1 — push volume up gently, minimise compliance
        #     loss = compliance_loss + 10.0 * vol_low

        # else:
        #     global MAX_AREA_FRAC
        #     MAX_AREA_FRAC = 0.2
        #     # Phase 2 — enforce full feasibility + minimise compliance
        #     conn = connectivity_penalty(rho)
        #     loss = (  compliance_loss
        #             + 30.0 * vol_low
        #             + 50.0 * vol_high
        #             + 20.0 * conn    )


        if step < PHASE1_STEPS:
            vol_low, vol_high = volume_penalties(rho, MAX_AREA_FRAC)      # 0.50
            loss = compliance_loss + 10.0 * vol_low
        else:
            vol_low, vol_high = volume_penalties(rho, MAX_AREA_FRAC_P2)   # 0.20
            conn = connectivity_penalty(rho)
            loss = (compliance_loss
                    + 30.0 * vol_low
                    + 50.0 * vol_high
                    + 20.0 * conn)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        #print("z AFTER step:")
        #print(z.detach().cpu().numpy())

        # ── bookkeeping ──
        vol_val  = float(rho_np.mean())
        feasible = is_feasible(rho_np)

        history["compliance"].append(float(compliance))
        history["vol"].append(vol_val)
        history["feasible"].append(feasible)
        history["loss"].append(float(loss.item()))

        # save best *feasible* design
        if step > PHASE1_STEPS and feasible and compliance < best_compliance:
            best_compliance = compliance
            best_rho        = rho_np.copy()
            best_step       = step

        if step % 20 == 0 or step == n_steps - 1:
            tag = "P1" if step < PHASE1_STEPS else "P2"
            print(f"[{tag}] step {step:03d} | "
                  f"C={compliance:8.2f} | vol={vol_val:.3f} | "
                  f"feasible={feasible} | "
                  f"best C={best_compliance:.2f} @ {best_step}")

        plot_design(rho_np, save_dir, step, f"design_step_{step:03d}.png")

    _save_results(history, best_rho, best_compliance, best_step, save_dir)
    return best_rho, best_compliance, history



# Visualisation helpers

def _save_results1(history, best_rho, best_compliance, best_step, save_dir):
    steps = np.arange(len(history["compliance"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # compliance curve
    ax = axes[0]
    ax.plot(history["compliance"], lw=1.5, label="compliance")
    ax.axvline(PHASE1_STEPS, color="red", ls="--", label="Phase 2 start")
    ax.set(xlabel="Step", ylabel="Compliance", title="Compliance history")
    ax.legend(); ax.grid(alpha=0.3)

    # volume fraction
    ax = axes[1]
    ax.plot(history["vol"], color="orange", lw=1.5)
    ax.axhline(MIN_AREA_FRAC, color="red",   ls="--", label="min vol")
    ax.axhline(MAX_AREA_FRAC, color="green", ls="--", label="max vol")
    ax.axvline(PHASE1_STEPS, color="gray", ls="--")
    ax.set(xlabel="Step", ylabel="Volume fraction", title="Volume history")
    ax.legend(); ax.grid(alpha=0.3)

    # feasibility scatter
    ax = axes[2]
    feas  = [i for i, f in enumerate(history["feasible"]) if f]
    infeas= [i for i, f in enumerate(history["feasible"]) if not f]
    ax.scatter(infeas, [history["compliance"][i] for i in infeas],
               c="red",   s=6,  alpha=0.4, label="infeasible")
    ax.scatter(feas,   [history["compliance"][i] for i in feas],
               c="green", s=10, alpha=0.8, label="feasible")
    ax.axvline(PHASE1_STEPS, color="gray", ls="--")
    ax.set(xlabel="Step", ylabel="Compliance", title="Feasibility map")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "history.png"), dpi=150)
    plt.close()

    if best_rho is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(best_rho,              cmap="gray_r", origin="lower", vmin=0, vmax=1)
        axes[0].set_title(f"Density  |  C={best_compliance:.2f}  |  step {best_step}")
        axes[0].axis("off")
        axes[1].imshow(best_rho > 0.5, cmap="gray_r", origin="lower")
        axes[1].set_title("Binary (threshold 0.5)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_design.png"), dpi=150)
        plt.close()
        np.save(os.path.join(save_dir, "best_rho.npy"), best_rho)

    print(f"\n✅  Best feasible compliance = {best_compliance:.4f}  (step {best_step})")
    print(f"    Results saved → {save_dir}")

def _save_results(history, best_rho, best_compliance, best_step, save_dir):
    n      = len(history["compliance"])
    steps  = np.arange(n)
    PHASE_COLOR = "purple"   # one colour for the phase boundary in all panels

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ---- compliance ----
    ax = axes[0]
    ax.plot(steps, history["compliance"], lw=1.5, label="compliance")
    ax.axvline(PHASE1_STEPS, color=PHASE_COLOR, ls="--", label="Phase 2 start")
    ax.set(xlabel="Step", ylabel="Compliance", title="Compliance history")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- volume fraction (per-phase bounds) ----
    ax = axes[1]
    ax.plot(steps, history["vol"], color="orange", lw=1.5, label="volume")
    # lower bound: active the whole run
    ax.axhline(MIN_AREA_FRAC, color="red", ls="--", label="min vol")
    # upper bound: 0.50 in Phase 1, 0.20 in Phase 2 — drawn as a step
    ax.hlines(MAX_AREA_FRAC,    0,            PHASE1_STEPS, color="green", ls="--",
              label="max vol (P1)")
    ax.hlines(MAX_AREA_FRAC_P2, PHASE1_STEPS, n - 1,        color="green", ls=":",
              label="max vol (P2)")
    ax.axvline(PHASE1_STEPS, color=PHASE_COLOR, ls="--")
    ax.set(xlabel="Step", ylabel="Volume fraction", title="Volume history")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- feasibility ----
    ax = axes[2]
    feas   = [i for i, f in enumerate(history["feasible"]) if f]
    infeas = [i for i, f in enumerate(history["feasible"]) if not f]
    ax.scatter(infeas, [history["compliance"][i] for i in infeas],
               c="red",   s=12, alpha=0.5, label="infeasible")
    ax.scatter(feas,   [history["compliance"][i] for i in feas],
               c="green", s=12, alpha=0.8, label="feasible")
    ax.axvline(PHASE1_STEPS, color=PHASE_COLOR, ls="--")
    ax.set(xlabel="Step", ylabel="Compliance", title="Feasibility map")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "history.png"), dpi=150)
    plt.close()

    if best_rho is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(best_rho, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        axes[0].set_title(f"Density  |  C={best_compliance:.2f}  |  step {best_step}")
        axes[0].axis("off")
        axes[1].imshow(best_rho > 0.5, cmap="gray_r", origin="lower")
        axes[1].set_title("Binary (threshold 0.5)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_design.png"), dpi=150)
        plt.close()
        np.save(os.path.join(save_dir, "best_rho.npy"), best_rho)

    print(f"\nBest feasible compliance = {best_compliance:.4f}  (step {best_step})")
    print(f"Results saved -> {save_dir}")

# Entry point

if __name__ == "__main__":
    from evaluate_vae_report import TopologyVAE   # your VAE class

    OUTPUT_DIR = "dataset/merged_vae_train_report"
    model = TopologyVAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "vae_best.pth"), map_location=DEVICE), strict=False
    )
    model.eval()

    # run multiple trials and keep the best
    all_results = []
    for trial in range(5):
        print(f"\n{'='*55}\nTrial {trial+1}/5\n{'='*55}")
        rho, comp, hist = optimize_latent(
            model,
            z_init   = None,
            n_steps  = TOTAL_STEPS,
            lr       = 0.05,
            save_dir = f"latent_opt_results/trial_{trial+1}"
        )
        all_results.append((comp, rho, trial+1))

    all_results.sort(key=lambda x: x[0])
    best_c, best_r, best_t = all_results[0]
    print(f"\n🏆  Global best  |  Trial {best_t}  |  Compliance = {best_c:.4f}")