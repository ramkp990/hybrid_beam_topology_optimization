
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label
from cma import CMAEvolutionStrategy

#  import the EXACT same functions the latent run uses
from fem_code import fem_physical_compliance, is_feasible

#  matched config 
GRID         = 64
SIGMOID_K    = 50.0          # same smoothing as data generation
M_MMC        = 6             # paper eq.(4)
LOAD_VALUE   = -100.0        
FEM_BUDGET   = 500           
POP_SIZE     = 12            
SIGMA0       = 0.2           
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2


# per beam: x, y, theta(deg), L(HALF-length), t(full thickness)
LB = np.array([0.01, 0.50, -75.0, 0.25, 0.015,
               0.25, 0.50, -75.0, 0.25, 0.015,
               0.80, 0.50, -75.0, 0.25, 0.015])
UB = np.array([0.20, 0.85,  75.0, 0.55, 0.080,
               0.75, 0.85,  75.0, 0.55, 0.080,
               0.99, 0.85,  75.0, 0.55, 0.080])

#  init: how to pick the per-seed starting design 
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

#  geometry (document-1 EXACT) 
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

#  objective (matches vae_cmaes.py shape) 
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

#  init helpers 
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

#  one CMA-ES run 
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

#  plotting 
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

#  main 
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
        print("\nNo feasible design found in any run.")

if __name__ == "__main__":
    main()
