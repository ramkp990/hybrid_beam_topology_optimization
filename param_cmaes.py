# cma_baseline_params.py
# ---------------------------------------------------------------------------
# Parameter-space CMA-ES baseline, matched to the latent (VAE-z) CMA-ES in
# vae_cmaes.py. The ONLY intended difference is the search variable:
#   latent run  : z in R^32  -> decoder -> rho
#   this baseline: 15 MMC params -> document-1 geometry map -> rho
# Everything else is reproduced from the latent run so the comparison is honest:
#   * SAME scorer + SAME gate : is_feasible / fem_physical_compliance from fem_code
#   * SAME objective shape     : infeasible -> 1e6 (no FEM); feasible -> compliance
#   * SAME budget convention   : while es.countevals < 500 (counts ALL asks)
#   * SAME CMA settings        : popsize=12, sigma0=0.2
#   * matched init semantics   : random *feasible* start per seed (mirrors that
#                                randn(32) decodes, via the prior, to a feasible design)
# The geometry map + bounds match document 1 (the data the VAE was trained on):
# L is the HALF-length (denominator L_half), bounds are the GENERATION ranges.
# ---------------------------------------------------------------------------
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label
from cma import CMAEvolutionStrategy

# >>> import the EXACT same functions the latent run uses (do not reimplement) <<<
from fem_code import fem_physical_compliance, is_feasible

# =============================== matched config ============================
GRID         = 64
SIGMOID_K    = 50.0          # same smoothing as data generation
M_MMC        = 6             # paper eq.(4)
LOAD_VALUE   = -100.0        # == vae_cmaes.py
FEM_BUDGET   = 500           # == vae_cmaes.py (counted in total evaluations)
POP_SIZE     = 12            # == vae_cmaes.py
SIGMA0       = 0.2           # == vae_cmaes.py
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2

# ---- bounds == document-1 GENERATION ranges (NOT document-3's values) ----
# per beam: x, y, theta(deg), L(HALF-length), t(full thickness)
LB = np.array([0.01, 0.50, -75.0, 0.25, 0.015,
               0.25, 0.50, -75.0, 0.25, 0.015,
               0.80, 0.50, -75.0, 0.25, 0.015])
UB = np.array([0.20, 0.85,  75.0, 0.55, 0.080,
               0.75, 0.85,  75.0, 0.55, 0.080,
               0.99, 0.85,  75.0, 0.55, 0.080])

# ---- init: how to pick the per-seed starting design --------------------
#   "sample"  : rejection-sample a random FEASIBLE param vector  (default)
#   "dataset" : pick a random row of DATASET_PARAMS (guaranteed feasible, real designs)
#   "uniform" : a single uniform-random vector (strict/harsh; may be infeasible)
INIT_MODE      = "sample"
DATASET_PARAMS = "dataset_cantilever_sym6_mmc9/params.npy"   # used only if INIT_MODE=="dataset"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # run many; the latent run was a SINGLE run

RESULTS_DIR = "results_param_cmaes"
os.makedirs(RESULTS_DIR, exist_ok=True)

xs = np.linspace(0.0, 1.0, GRID)
ys = np.linspace(0.0, 1.0, GRID)
Xg, Yg = np.meshgrid(xs, ys)

# =============================== geometry (document-1 EXACT) ===============
def compute_local_phi(xc, yc, theta_deg, L_half, t_full, m=M_MMC):
    th = math.radians(float(theta_deg)); c, s = math.cos(th), math.sin(th)
    Xr =  c * (Xg - xc) + s * (Yg - yc)
    Yr = -s * (Xg - xc) + c * (Yg - yc)
    term_p = np.abs(Xr / (L_half + 1e-12)) ** m
    term_q = np.abs(Yr / (t_full / 2.0 + 1e-12)) ** m
    return -(term_p + term_q - 1.0)

def build_phi_global(beams_top):
    phis = [compute_local_phi(*b) for b in beams_top]
    phis += [compute_local_phi(b[0], 1.0 - b[1], -b[2], b[3], b[4]) for b in beams_top]
    return np.maximum.reduce(phis), phis

def sigmoid_stable(x, k=SIGMOID_K):
    z = np.clip(k * x, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))

def params_to_beams(p):
    return [(p[5*i], p[5*i+1], p[5*i+2], p[5*i+3], p[5*i+4]) for i in range(3)]

def density_from_params(p):
    phi, _ = build_phi_global(params_to_beams(p))
    return sigmoid_stable(phi, SIGMOID_K).astype(np.float64)

# =============================== objective (matches vae_cmaes.py shape) ====
class Counters:
    def __init__(self):
        self.total = self.fem = self.feas = self.infeas = 0

def make_objective(counters):
    def objective_norm(z):                     # z in [0,1]^15
        counters.total += 1
        p = LB + np.clip(np.asarray(z), 0.0, 1.0) * (UB - LB)
        rho = density_from_params(p)
        if not is_feasible(rho):               # SAME gate as latent run
            counters.infeas += 1
            return 1e6                          # SAME flat penalty, no FEM
        counters.feas += 1; counters.fem += 1
        comp, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)
        return comp
    return objective_norm

# =============================== init helpers =============================
def _to_norm(p):  return (np.clip(p, LB, UB) - LB) / (UB - LB)

def make_init(seed):
    rng = np.random.RandomState(seed)
    if INIT_MODE == "dataset" and os.path.exists(DATASET_PARAMS):
        params = np.load(DATASET_PARAMS)
        return _to_norm(params[rng.randint(len(params))])
    if INIT_MODE == "uniform":
        return _to_norm(rng.uniform(LB, UB))
    # "sample": rejection-sample a random feasible design (mirrors decoder prior)
    for _ in range(5000):
        p = rng.uniform(LB, UB)
        if is_feasible(density_from_params(p)):
            return _to_norm(p)
    return _to_norm(0.5 * (LB + UB))           # fallback

# =============================== one CMA-ES run ===========================
def run_once(seed):
    counters = Counters()
    objective = make_objective(counters)
    z0 = make_init(seed)
    es = CMAEvolutionStrategy(z0, SIGMA0, {
        "popsize": POP_SIZE, "bounds": [0.0, 1.0], "seed": int(seed), "verbose": -9,
    })
    best = float("inf")
    while es.countevals < FEM_BUDGET:          # SAME loop/budget as latent run
        zs = es.ask()
        fs = [objective(z) for z in zs]
        es.tell(zs, fs)
        best = min(best, min(fs))
    best_z = np.asarray(es.result.xbest)
    best_p = LB + best_z * (UB - LB)
    rho = density_from_params(best_p)
    comp, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)
    feasible = bool(is_feasible(rho))
    return {"seed": seed, "compliance": comp if feasible else float("inf"),
            "raw_compliance": comp, "feasible": feasible, "params": best_p, "rho": rho,
            "total": counters.total, "fem": counters.fem,
            "feas": counters.feas, "infeas": counters.infeas}

# =============================== plotting =================================
def plot_design(res, path):
    p, rho = res["params"], res["rho"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for i, (xc, yc, th, L, t) in enumerate(params_to_beams(p)):   # L is HALF-length
        for yy, tt in [(yc, th), (1.0 - yc, -th)]:
            r = math.radians(tt); dx, dy = L * math.cos(r), L * math.sin(r)
            ax1.plot([xc - dx, xc + dx], [yy - dy, yy + dy], color=f"C{i}",
                     lw=10, alpha=0.85, solid_capstyle="round")
    ax1.plot(0.0, 0.5, "bs", ms=12, label="support"); ax1.plot(1.0, 0.5, "r^", ms=12, label="load")
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05); ax1.set_aspect("equal")
    ax1.set_title("optimized MMC beams (3 top + 3 mirrored)"); ax1.legend(loc="upper right")
    im = ax2.imshow(rho, extent=[0, 1, 0, 1], origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax2.contour(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID), rho, levels=[0.5], colors="tab:blue")
    ax2.plot(1.0, 0.5, "r^", ms=10); ax2.axvline(0, color="tab:blue", ls="--", alpha=0.7)
    ax2.set_aspect("equal")
    ax2.set_title(f"compliance {res['raw_compliance']:.2f} | feasible={res['feasible']}")
    fig.colorbar(im, ax=ax2, label=r"$\rho$")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

# =============================== main =====================================
def main():
    print(f"param-space baseline | INIT_MODE={INIT_MODE} | popsize={POP_SIZE} "
          f"| sigma0={SIGMA0} | budget(countevals)={FEM_BUDGET} | seeds={len(SEEDS)}\n")
    runs = [run_once(s) for s in SEEDS]

    print("per-seed: compliance | feasible | fem_calls/total (FEM efficiency)")
    for r in runs:
        eff = 100.0 * r["fem"] / max(1, r["total"])
        print(f"  seed {r['seed']}: {r['raw_compliance']:8.3f}  feasible={r['feasible']}  "
              f"{r['fem']}/{r['total']} ({eff:4.1f}%)")

    feas = [r for r in runs if r["feasible"]]
    comps = np.array([r["compliance"] for r in feas]) if feas else np.array([])
    if comps.size:
        print(f"\nfeasible runs = {len(feas)}/{len(runs)} | "
              f"median compliance = {np.median(comps):.3f} | best = {comps.min():.3f}")
        best = min(feas, key=lambda r: r["compliance"])
        np.save(os.path.join(RESULTS_DIR, "best_params.npy"), best["params"])
        np.save(os.path.join(RESULTS_DIR, "best_rho.npy"), best["rho"].astype(np.float32))
        np.save(os.path.join(RESULTS_DIR, "all_compliances.npy"), comps)
        plot_design(best, os.path.join(RESULTS_DIR, "best_design.png"))
        print(f"saved best (compliance {best['raw_compliance']:.3f}) to {RESULTS_DIR}/")
    else:
        print("\nNo feasible design found in any run. With INIT_MODE='uniform' under the "
              "flat 1e6 gate this can happen; use 'sample' or 'dataset' to start feasible.")
    print("\nCompare against your latent best_rho compliance. Because the latent run was a "
          "SINGLE run, report where it falls in this baseline's distribution (median/best/range), "
          "and compare fem_calls/total (FEM efficiency) between the two.")

if __name__ == "__main__":
    main()
'''

# cmaes_param_optimization.py
import os
import numpy as np
from cma import CMAEvolutionStrategy
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# -----------------------------
# Configuration (same as your data generation)
# -----------------------------
GRID = 64
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2
M_MMC = 6
SIGMOID_K = 50.0
LOAD_VALUE = -100.0
HEAVISIDE_CUTOFF = 0.0
# Paper constants
MIN_AREA = int(0.10 * GRID * GRID)  # 10% min volume
MAX_AREA = int(0.50 * GRID * GRID)  # 50% max volume (Vreq)
VOLUME_FRACTION = 0.50              # Paper constraint
M_MMC = 6                           # Even integer from paper eq(4)


# Parameter bounds (exactly from your data generation)
BOUNDS = {
    "left_x_range": (0.05, 0.25),      # Paper Fig6 reference layout
    "mid_x_range":  (0.40, 0.60), 
    "right_x_range":(0.75, 0.95),
    "y_top_range":  (0.60, 0.90),      # Top half only (bottom mirrored)
    "theta_range":  (-45.0, 45.0),     # Paper uses smaller angle range
    "L_range":      (0.30, 0.80),      # FULL length li (not half!)
    "t_range":      (0.04, 0.12),      # FULL thickness ti
}



xs = np.linspace(0.0, 1.0, GRID)
ys = np.linspace(0.0, 1.0, GRID)
Xg, Yg = np.meshgrid(xs, ys)

# Results directory
RESULTS_DIR = "results_param_cmaes"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Plot the FINAL BEAMS (add this after VTK export)
import matplotlib.pyplot as plt

def plot_final_beams(best_params, save_path=None, compliance_final=None):
    """Plot MMC beams CORRECTLY with rotation"""
    beams_top = params_to_vector(best_params)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: CORRECT beam geometry (endpoints + thickness)
    for i, (xc, yc, theta_deg, L_half, t_full) in enumerate(beams_top):
        theta_rad = np.radians(theta_deg)
        
        # Beam centerline endpoints
        dx = L_half * np.cos(theta_rad)
        dy = L_half * np.sin(theta_rad)
        
        x_start = xc - dx
        y_start = yc - dy
        x_end = xc + dx
        y_end = yc + dy
        
        # Draw centerline (thick line)
        ax1.plot([x_start, x_end], [y_start, y_end], 
                color=f'C{i}', linewidth=12, alpha=0.9,
                label=f'Top {i+1}: θ={theta_deg:.0f}°')
        
        # Draw thickness (perpendicular direction)
        perp_dx = (t_full/2) * np.sin(theta_rad)  # perpendicular vector
        perp_dy = -(t_full/2) * np.cos(theta_rad)
        
        # Top edge
        ax1.plot([x_start-perp_dx, x_end-perp_dx], 
                [y_start-perp_dy, y_end-perp_dy], 
                color=f'C{i}', linewidth=4, alpha=0.6)
        
        # Bottom edge  
        ax1.plot([x_start+perp_dx, x_end+perp_dx], 
                [y_start+perp_dy, y_end+perp_dy], 
                color=f'C{i}', linewidth=4, alpha=0.6)
        
        # Mirrored bottom beam
        yc_m = 1.0 - yc
        theta_m = -theta_deg
        theta_m_rad = np.radians(theta_m)
        
        dx_m = L_half * np.cos(theta_m_rad)
        dy_m = L_half * np.sin(theta_m_rad)
        
        x_start_m = xc - dx_m
        y_start_m = yc_m - dy_m
        x_end_m = xc + dx_m
        y_end_m = yc_m + dy_m
        
        ax1.plot([x_start_m, x_end_m], [y_start_m, y_end_m], 
                color=f'C{i}', linewidth=10, alpha=0.7, linestyle='--')
    
    # Support and load markers
    ax1.plot(0, 0.5, 'bs', markersize=15, label='Fixed support') 
    ax1.plot(1.0, 0.5, 'r^', markersize=15, label='Load point')
    
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Optimized MMC Beams (3 Top + 3 Mirrored Bottom)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Density field (unchanged)
    phi_final, _ = build_phi_global(beams_top)
    rho_final = sigmoid_stable(phi_final, k=SIGMOID_K)
    im = ax2.imshow(rho_final, extent=[0,1,0,1], cmap='gray_r', vmin=0, vmax=1)
    ax2.contour(rho_final, levels=[0.5], colors='blue', alpha=0.7)  # Beam boundaries
    
    ax2.plot(1.0, 0.5, 'r^', markersize=12, label='Load')
    ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='Fixed')
    ax2.set_title(f'Final Density Field\nCompliance: {compliance_final:.2f} N·mm')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im, ax=ax2, label='Density ρ')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()




def sigmoid_stable(x, k=50.0):
    """Numerically stable sigmoid to avoid overflow/underflow"""
    x = k * x
    # Clip to prevent overflow
    x = np.clip(x, -700, 700)  # np.exp(700) is near float64 limit
    return 1 / (1 + np.exp(-x))


def compute_local_phi(xc, yc, theta_deg, L_full, t_full, m=6):
    """EXACT paper equation (4) - m=6 even integer"""
    theta = math.radians(float(theta_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    # Normalized coordinates [0,1]
    Xr = c * (Xg - xc) + s * (Yg - yc)      # Parallel coord
    Yr = -s * (Xg - xc) + c * (Yg - yc)     # Perp coord

    # EXACT paper denominators: li/2, ti/2
    phi = - ( 
        np.abs(Xr / (L_full/2))**m + 
        np.abs(Yr / (t_full/2))**m - 1.0 
    )
    return phi


# -----------------------------
# Build global phi from list of beams (top + mirrored bottom)
# -----------------------------
def build_phi_global(beams_params):
    """
    beams_params: list of tuples (xc,yc,theta,L_half,t_full) for top beams
    Returns:
      phi_global: (GRID, GRID) float
      phi_components: list of phi_i for each of 6 components [top1,top2,top3,bot1,bot2,bot3]
    """
    phis = []
    # top beams
    for (xc, yc, theta, L, t) in beams_params:
        phis.append(compute_local_phi(xc, yc, theta, L, t))
    # mirrored bottom beams (y -> 1 - y, theta -> -theta)
    for (xc, yc, theta, L, t) in beams_params:
        phis.append(compute_local_phi(xc, 1.0 - yc, -theta, L, t))
    # global phi = max across components
    phi_global = np.maximum.reduce(phis)
    return phi_global, phis

# -----------------------------
# Convert phi -> binary grid (Heaviside)
# -----------------------------
def phi_to_binary(phi, cutoff=HEAVISIDE_CUTOFF):
    return (phi >= cutoff).astype(np.uint8)

def element_stiffness_plane_stress(E, nu, xe, ye):
    """
    4-node bilinear quadrilateral element stiffness matrix (plane stress).
    xe, ye: element dimensions in mm (physical units).
    Returns Ke (8x8) in consistent units (N/mm).
    """
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w = np.array([1.0, 1.0])

    coeff = E / (1.0 - nu**2)
    D = coeff * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, (1.0 - nu) / 2.0]])

    Ke = np.zeros((8, 8), dtype=np.float64)

    for xi in gp:
        for eta in gp:
            dN_dxi = 0.25 * np.array([
                [-(1-eta), -(1-xi)],
                [ (1-eta), -(1+xi)],
                [ (1+eta),  (1+xi)],
                [-(1+eta),  (1-xi)]
            ])

            J = np.array([[xe/2.0, 0.0],
                          [0.0,   ye/2.0]])
            detJ = xe * ye / 4.0
            invJ = np.array([[2.0/xe, 0.0],
                             [0.0,   2.0/ye]])

            B = np.zeros((3, 8))
            for i in range(4):
                dN_nat = dN_dxi[i, :]
                dN_phys = invJ @ dN_nat
                dNdx, dNdy = dN_phys[0], dN_phys[1]
                B[0, 2*i]   = dNdx
                B[1, 2*i+1] = dNdy
                B[2, 2*i]   = dNdy
                B[2, 2*i+1] = dNdx

            Ke += (B.T @ D @ B) * detJ

    return Ke


def fem_physical_compliance(geom, load_node=None, load_value=-100.0,
                            save_vtk=False, vtk_filename=None):
    """
    FEM compliance with SIMP material interpolation.
    - geom: (64,64) density field in [0,1]
    Returns: compliance (float), displacement vector u
    """

    ny, nx = geom.shape            # 64 x 64 nodes
    nel_y, nel_x = ny - 1, nx - 1  # 63 x 63 elements
    nnodes = ny * nx
    ndof = 2 * nnodes

    # Physical domain (same as before)
    xe = 20.0 / nel_x
    ye = 10.0 / nel_y

    # Material parameters
    E_solid = 2.1e5     # MPa
    E_void  = 1e-3 * E_solid   # small stiffness to avoid singularity
    nu = 0.3
    p = 3.0             # SIMP exponent (fixed for dataset)

    # Element stiffness for unit modulus
    Ke_unit = element_stiffness_plane_stress(1.0, nu, xe, ye)

    rows, cols, data = [], [], []

    def node_id(iy, ix):
        return iy * nx + ix

    # === Element loop ===
    for ey in range(nel_y):
        for ex in range(nel_x):

            # Average density (nodal -> elemental)
            rho_e = (
                geom[ey,   ex] +
                geom[ey,   ex+1] +
                geom[ey+1, ex+1] +
                geom[ey+1, ex]
            ) / 4.0

            # SIMP interpolation
            E_e = E_void + (rho_e ** p) * (E_solid - E_void)

            Ke = (E_e / 1.0) * Ke_unit  # scale unit stiffness

            nodes = [
                node_id(ey, ex),
                node_id(ey, ex+1),
                node_id(ey+1, ex+1),
                node_id(ey+1, ex)
            ]

            dofs = []
            for n in nodes:
                dofs.extend([2*n, 2*n+1])

            for i in range(8):
                for j in range(8):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    data.append(Ke[i, j])

    # Assemble global stiffness
    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # === Boundary conditions (same as before) ===
    fixed = []
    for iy in range(ny):
        nid = node_id(iy, 0)
        fixed.extend([2*nid, 2*nid+1])

    free_dofs = np.setdiff1d(np.arange(ndof), fixed)

    # === Load ===
    if load_node is None:
        load_node = (ny//2, nx-1)

    f = np.zeros(ndof)
    lnid = node_id(*load_node)
    f[2*lnid + 1] = load_value

    # === Solve ===
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    f_f = f[free_dofs]

    try:
        u_f = spla.spsolve(K_ff, f_f)
    except:
        u_f = spla.spsolve(K_ff + 1e-8 * sp.eye(K_ff.shape[0]), f_f)

    u = np.zeros(ndof)
    u[free_dofs] = u_f

    # === Compliance ===
    compliance = float(f @ u)

    # Optional VTK export
    if save_vtk:
        save_fem_to_vtk(geom, u, vtk_filename, domain_size=(20.0, 10.0))

    return compliance, u



def save_fem_to_vtk(geom, u, filename, domain_size=(20.0, 10.0)):
    try:
        import vtk
    except ImportError:
        print("⚠️ vtk not installed. Skipping VTK export.")
        return

    ny, nx = geom.shape  # e.g., 64x64
    Lx, Ly = domain_size

    # Create image data (regular grid)
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetSpacing(Lx / (nx - 1), Ly / (ny - 1), 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)

    # Density (scalar field)
    density = vtk.vtkFloatArray()
    density.SetName("Density")
    density.SetNumberOfComponents(1)
    for j in range(ny):
        for i in range(nx):
            density.InsertNextValue(float(geom[j, i]))
    image.GetPointData().SetScalars(density)

    # Displacement (vector field)
    displacement = vtk.vtkFloatArray()
    displacement.SetName("Displacement")
    displacement.SetNumberOfComponents(3)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            ux = float(u[2 * idx]) if 2 * idx < len(u) else 0.0
            uy = float(u[2 * idx + 1]) if 2 * idx + 1 < len(u) else 0.0
            displacement.InsertNextTuple3(ux, uy, 0.0)
    image.GetPointData().AddArray(displacement)

    # Write to file
    writer = vtk.vtkXMLImageDataWriter()  # Uses .vti (modern, robust)
    writer.SetFileName(filename.replace(".vtk", ".vti"))
    writer.SetInputData(image)
    writer.Write()
    print(f"✅ Saved FEM results to {filename.replace('.vtk', '.vti')}")

def fem_check_threshold(geom, thresh):
    """Paper-aligned FEM check."""
    comp, _ = fem_physical_compliance(geom)
    return comp, comp <= thresh



# -----------------------------
# Parameter to vector conversion
# -----------------------------
def params_to_vector(params):
    # params[3,8,13] = FULL l, params[4,9,14] = FULL t
    beams = [] 
    for i in range(3):
        beams.append((params[5*i], params[5*i+1], params[5*i+2], 
                     params[5*i+3], params[5*i+4]))  # l_full, t_full
    return beams

def vector_to_params(beams):
    """Convert beam list back to 15D vector"""
    return np.array([beams[0] + beams[1] + beams[2]])

# -----------------------------
# Feasibility check for parameter space
# -----------------------------
def is_feasible_param1(params):
    """Paper constraints: VOLUME + CONNECTIVITY"""
    try:
        beams = params_to_vector(params)
        phi_global, phi_components = build_phi_global(beams)
        rho_smooth = sigmoid_stable(phi_global, k=SIGMOID_K)
        
        # 🔥 PAPER VOLUME CONSTRAINT (not pixel area)
        volume_frac = rho_smooth.mean()  # <-> V(x)/Vdomain
        if volume_frac > VOLUME_FRACTION:  # <= 50%
            return False
            
        geom = (rho_smooth > 0.5).astype(np.uint8)
        
        # Paper CONNECTIVITY (support + load + single component)
        from scipy.ndimage import label
        labeled, num_features = label(geom)
        if num_features == 0:
            print("⚠️ No material present in design.")
            return False
            
        sizes = np.bincount(labeled.ravel())[1:]
        largest = sizes.max()
        if largest < 0.90 * geom.sum():  # Main component dominates
            print("⚠️ Design is too fragmented. Largest component is only ")
            return False
            
        main_mask = (labeled == np.argmax(sizes) + 1)
        
        # Left support connection
        if main_mask[:, 0].sum() == 0:
            print("⚠️ No connection to fixed support on left edge.")
            return False
        # Right load connection  
        if main_mask[LOAD_POINT_Y, LOAD_POINT_X] == 0:
            print("⚠️ No connection to load point on right edge.")
            return False
            
        return True
    except:
        return False

def is_feasible_param(params):
    """RELAXED edge connections - within 3 pixels = OK"""
    try:
        beams = params_to_vector(params)
        phi_global, _ = build_phi_global(beams)
        rho_smooth = sigmoid_stable(phi_global, k=SIGMOID_K)
        
        # Volume constraint only
        if rho_smooth.mean() > 0.50:
            print("⚠️ Volume fraction exceeds 50%")
            return False
            
        geom = (rho_smooth > 0.3).astype(np.uint8)  # Lower threshold!
        if geom.sum() < 200:
            print("⚠️ Too little material (less than 200 pixels)")
            return False
            
        from scipy.ndimage import label
        labeled, num_features = label(geom)
        if num_features == 0:
            print("⚠️ No material present in design.")
            return False
            
        sizes = np.bincount(labeled.ravel())[1:]
        main_idx = np.argmax(sizes) + 1
        main_mask = (labeled == main_idx)
        
        # 🔥 RELAXED: Within 3-pixel buffer (1.5% of domain)
        LEFT_BUFFER = 3   # pixels from left edge
        RIGHT_BUFFER = 3  # pixels from right edge
        
        # Left support: ANY material in first LEFT_BUFFER columns
        if main_mask[:, :LEFT_BUFFER].sum() == 0:
            print("⚠️ No connection to left support (within 3px)")
            return False
            
        # Right load: ANY material near load point (±3px)
        ly, lx = LOAD_POINT_Y, LOAD_POINT_X
        load_region = main_mask[max(0,ly-3):ly+4, max(0,lx-3):lx+4]
        if load_region.sum() == 0:
            print("⚠️ No connection to load point (within 3x3)")
            return False
            
        return True
    except:
        return False



# -----------------------------
# Objective function for parameter optimization
# -----------------------------
def objective_param(params):
    """Objective function for CMA-ES on parameters"""
    # Check feasibility first
    if not is_feasible_param(params):
        return 1e6  # Penalty for infeasible designs
    
    # Build density field
    beams = params_to_vector(params)
    phi_global, _ = build_phi_global(beams)
    #rho_smooth = 1 / (1 + np.exp(-SIGMOID_K * phi_global))
    rho_smooth = sigmoid_stable(phi_global, k=SIGMOID_K)

    # Compute compliance
    try:
        compliance, _ = fem_physical_compliance(rho_smooth, load_value=LOAD_VALUE)
        return compliance
    except:
        return 1e6  # Penalty for FEM failures

# -----------------------------
# Main optimization
# -----------------------------
def main():
    print("🚀 HKG-LSM CMA-ES Phase (15D Cantilever)")
    
    # PAPER REFERENCE → MAKE FEASIBLE FIRST
    x0 = np.array([
        0.15, 0.70,  20.0, 0.55, 0.08,   # Beam 1 LEFT
        0.50, 0.85, -15.0, 0.65, 0.07,   # Beam 2 MIDDLE-UP  
        0.85, 0.75,   5.0, 0.45, 0.06,   # Beam 3 RIGHT
    ])
    
    # ✅ CORRECTED BOUNDS
    lb = np.array([0.05, 0.60, -360, 0.30, 0.04,  0.40, 0.60, -360, 0.30, 0.04,  0.75, 0.60, -360, 0.30, 0.04])
    ub = np.array([0.25, 0.90,  360, 0.80, 0.12,  0.60, 0.90,  360, 0.80, 0.12,  0.95, 0.90,  360, 0.80, 0.12])
    
    es = CMAEvolutionStrategy(
        x0=x0,
        sigma0=0.05,
        inopts={'popsize': 10, 'bounds': [lb, ub]}
    )
    
    FEM_BUDGET = 500
    best_compliance = float("inf")
    best_params = x0.copy()
    generation = 0  # ✅ MANUAL COUNTER
    
    # Test initial feasibility
    init_comp = objective_param(x0)
    print(f"📐 Initial angles: {x0[[2,7,12]]}")
    print(f"Initial compliance: {init_comp:.2f}")
    
    while es.countevals < FEM_BUDGET:
        generation += 1  # ✅ MANUAL INCREMENT
        
        solutions = es.ask()
        fitnesses = [objective_param(s) for s in solutions]
        es.tell(solutions, fitnesses)
        
        gen_best_idx = np.argmin(fitnesses)
        gen_best_val = fitnesses[gen_best_idx]
        
        print(f"Gen {generation:2d} | Evals {es.countevals:3d} | Best: {gen_best_val:6.2f}")
        
        if gen_best_val < best_compliance:
            best_compliance = gen_best_val
            best_params = solutions[gen_best_idx].copy()
    
    # Rest unchanged...
    np.save(os.path.join(RESULTS_DIR, "best_params.npy"), best_params)
    beams_final = params_to_vector(best_params)
    phi_final, _ = build_phi_global(beams_final)
    rho_final = sigmoid_stable(phi_final, k=SIGMOID_K)
    compliance_final, u = fem_physical_compliance(rho_final, load_value=LOAD_VALUE)
    
    os.makedirs(os.path.join(RESULTS_DIR, "vti"), exist_ok=True)
    save_fem_to_vtk(rho_final, u, os.path.join(RESULTS_DIR, "vti", "best_design.vtk"))
    plot_final_beams(best_params, os.path.join(RESULTS_DIR, "final_beams.png"), compliance_final)
    
    print(f"\n✅ HKG-LSM CMA-ES completed!")
    print(f"Final compliance: {compliance_final:.2f} N·mm")


if __name__ == "__main__":
    main()



⚠️ No connection to left support (within 3px)
📐 Initial angles: [ 20. -15.   5.]
Initial compliance: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  1 | Evals  10 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  2 | Evals  20 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  3 | Evals  30 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  4 | Evals  40 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  5 | Evals  50 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  6 | Evals  60 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  7 | Evals  70 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  8 | Evals  80 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  9 | Evals  90 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 10 | Evals 100 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 11 | Evals 110 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 12 | Evals 120 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 13 | Evals 130 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 14 | Evals 140 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 15 | Evals 150 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 16 | Evals 160 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 17 | Evals 170 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 18 | Evals 180 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 19 | Evals 190 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 20 | Evals 200 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 21 | Evals 210 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 22 | Evals 220 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 23 | Evals 230 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 24 | Evals 240 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 25 | Evals 250 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 26 | Evals 260 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 27 | Evals 270 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 28 | Evals 280 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 29 | Evals 290 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 30 | Evals 300 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 31 | Evals 310 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 32 | Evals 320 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 33 | Evals 330 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 34 | Evals 340 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 35 | Evals 350 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 36 | Evals 360 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 37 | Evals 370 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 38 | Evals 380 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 39 | Evals 390 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 40 | Evals 400 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 41 | Evals 410 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 42 | Evals 420 | Best: 321.79
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 43 | Evals 430 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 44 | Evals 440 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 45 | Evals 450 | Best: 196.30
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 46 | Evals 460 | Best: 131.98
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 47 | Evals 470 | Best: 162.05
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 48 | Evals 480 | Best:  51.93
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 49 | Evals 490 | Best:  32.15
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 50 | Evals 500 | Best: 116.33
✅ Saved FEM results to results_param_cmaes/vti/best_design.vti

✅ HKG-LSM CMA-ES completed!
Final compliance: 32.15 N·mm





⚠️ No connection to left support (within 3px)
📐 Initial angles: [ 20. -15.   5.]
Initial compliance: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  1 | Evals  10 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  2 | Evals  20 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  3 | Evals  30 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  4 | Evals  40 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  5 | Evals  50 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  6 | Evals  60 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  7 | Evals  70 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  8 | Evals  80 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  9 | Evals  90 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 10 | Evals 100 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 11 | Evals 110 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 12 | Evals 120 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 13 | Evals 130 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 14 | Evals 140 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 15 | Evals 150 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 16 | Evals 160 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 17 | Evals 170 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 18 | Evals 180 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 19 | Evals 190 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 20 | Evals 200 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 21 | Evals 210 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 22 | Evals 220 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 23 | Evals 230 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 24 | Evals 240 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 25 | Evals 250 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 26 | Evals 260 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 27 | Evals 270 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 28 | Evals 280 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 29 | Evals 290 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 30 | Evals 300 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 31 | Evals 310 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 32 | Evals 320 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 33 | Evals 330 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 34 | Evals 340 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 35 | Evals 350 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 36 | Evals 360 | Best: 106.21
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 37 | Evals 370 | Best: 116.18
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 38 | Evals 380 | Best:  86.03
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 39 | Evals 390 | Best: 111.71
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 40 | Evals 400 | Best: 118.15
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 41 | Evals 410 | Best:  50.22
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 42 | Evals 420 | Best: 161.37
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 43 | Evals 430 | Best: 107.17
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 44 | Evals 440 | Best: 107.49
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 45 | Evals 450 | Best: 118.12
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 46 | Evals 460 | Best:  72.17
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 47 | Evals 470 | Best:  85.71
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 48 | Evals 480 | Best:  90.02
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 49 | Evals 490 | Best:  84.82
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 50 | Evals 500 | Best: 112.33
✅ Saved FEM results to results_param_cmaes/vti/best_design.vti

✅ HKG-LSM CMA-ES completed!
Final compliance: 50.22 N·mm


⚠️ No connection to left support (within 3px)
📐 Initial angles: [ 20. -15.   5.]
Initial compliance: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  1 | Evals  10 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  2 | Evals  20 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  3 | Evals  30 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen  4 | Evals  40 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen  5 | Evals  50 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  6 | Evals  60 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen  7 | Evals  70 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  8 | Evals  80 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen  9 | Evals  90 | Best: 1000000.00
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 10 | Evals 100 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 11 | Evals 110 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 12 | Evals 120 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 13 | Evals 130 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 14 | Evals 140 | Best: 337.04
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 15 | Evals 150 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 16 | Evals 160 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 17 | Evals 170 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 18 | Evals 180 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 19 | Evals 190 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 20 | Evals 200 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 21 | Evals 210 | Best: 1000000.00
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 22 | Evals 220 | Best: 130.35
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 23 | Evals 230 | Best:  57.13
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 24 | Evals 240 | Best:  50.47
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 25 | Evals 250 | Best:  91.33
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
Gen 26 | Evals 260 | Best: 107.84
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 27 | Evals 270 | Best:  50.79
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 28 | Evals 280 | Best:  42.42
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 29 | Evals 290 | Best:  58.05
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 30 | Evals 300 | Best:  71.42
⚠️ No connection to left support (within 3px)
Gen 31 | Evals 310 | Best:  39.82
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 32 | Evals 320 | Best:  37.63
⚠️ No connection to left support (within 3px)
Gen 33 | Evals 330 | Best:  35.53
⚠️ No connection to load point (within 3x3)
Gen 34 | Evals 340 | Best:  31.74
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 35 | Evals 350 | Best:  47.08
Gen 36 | Evals 360 | Best:  43.07
Gen 37 | Evals 370 | Best:  43.11
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 38 | Evals 380 | Best:  93.57
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 39 | Evals 390 | Best:  46.99
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to load point (within 3x3)
Gen 40 | Evals 400 | Best:  39.81
⚠️ No connection to left support (within 3px)
Gen 41 | Evals 410 | Best:  39.63
⚠️ No connection to load point (within 3x3)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
⚠️ No connection to left support (within 3px)
Gen 42 | Evals 420 | Best:  36.47
Gen 43 | Evals 430 | Best:  34.87
⚠️ No connection to load point (within 3x3)
Gen 44 | Evals 440 | Best:  42.19
Gen 45 | Evals 450 | Best:  41.12
Gen 46 | Evals 460 | Best:  93.27
⚠️ No connection to load point (within 3x3)
⚠️ No connection to load point (within 3x3)
Gen 47 | Evals 470 | Best:  58.72
Gen 48 | Evals 480 | Best:  45.72
⚠️ No connection to load point (within 3x3)
Gen 49 | Evals 490 | Best:  60.63
Gen 50 | Evals 500 | Best:  42.57
✅ Saved FEM results to results_param_cmaes/vti/best_design.vti

✅ HKG-LSM CMA-ES completed!
Final compliance: 31.74 N·mm
'''