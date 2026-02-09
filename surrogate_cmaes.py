# surrogate_cmaes.py
import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from vae_model import TopologyVAE
from fem_code import fem_physical_compliance, is_feasible, save_fem_to_vtk
import joblib
from sklearn.metrics import pairwise_distances


def save_fem_to_vtk(geom, u, vm_stress, filename, domain_size=(20.0, 10.0)):
    import vtk

    ny, nx = geom.shape
    Lx, Ly = domain_size

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetSpacing(Lx/(nx-1), Ly/(ny-1), 1.0)
    image.SetOrigin(0, 0, 0)

    # Density
    density = vtk.vtkFloatArray()
    density.SetName("Density")
    for j in range(ny):
        for i in range(nx):
            density.InsertNextValue(float(geom[j, i]))
    image.GetPointData().SetScalars(density)

    # Displacement
    disp = vtk.vtkFloatArray()
    disp.SetName("Displacement")
    disp.SetNumberOfComponents(3)
    for j in range(ny):
        for i in range(nx):
            idx = j*nx + i
            disp.InsertNextTuple3(
                u[2*idx], u[2*idx+1], 0.0
            )
    image.GetPointData().AddArray(disp)

    # Von Mises stress (cell data)
    vm = vtk.vtkFloatArray()
    vm.SetName("VonMisesStress")
    for j in range(vm_stress.shape[0]):
        for i in range(vm_stress.shape[1]):
            vm.InsertNextValue(float(vm_stress[j, i]))

    image.GetCellData().AddArray(vm)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image)
    writer.Write()


def compute_von_mises_stress(geom, u):
    ny, nx = geom.shape
    nel_y, nel_x = ny - 1, nx - 1

    xe = 20.0 / nel_x
    ye = 10.0 / nel_y

    E = 2.1e5
    nu = 0.3
    coeff = E / (1 - nu**2)
    D = coeff * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0]
    ])

    vm = np.zeros((nel_y, nel_x))

    def node_id(i, j):
        return i * nx + j

    for ey in range(nel_y):
        for ex in range(nel_x):
            nodes = [
                node_id(ey, ex),
                node_id(ey, ex+1),
                node_id(ey+1, ex+1),
                node_id(ey+1, ex)
            ]

            dofs = []
            for n in nodes:
                dofs.extend([2*n, 2*n+1])

            u_e = u[dofs]

            # B matrix at element center (ξ=0, η=0)
            B = np.array([
                [-1/xe, 0,  1/xe, 0,  1/xe, 0, -1/xe, 0],
                [0, -1/ye, 0, -1/ye, 0, 1/ye, 0, 1/ye],
                [-1/ye, -1/xe, -1/ye, 1/xe, 1/ye, 1/xe, 1/ye, -1/xe]
            ])

            stress = D @ (B @ u_e)
            sx, sy, txy = stress

            vm[ey, ex] = np.sqrt(
                sx**2 - sx*sy + sy**2 + 3*txy**2
            )

    return vm


# Configuration
MODEL_PATH = "dataset/merged_vae_train/vae_best.pth"
SURROGATE_PATH = "surrogate/gp_surrogate.pkl"
SCALER_PATH = "surrogate/z_scaler.pkl"
LATENT_DIM = 32
FEM_BUDGET = 250
POPSIZE = 12
FEM_PER_GEN = 8
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = TopologyVAE(latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
vae.eval()

gp = joblib.load(SURROGATE_PATH)
scaler = joblib.load(SCALER_PATH)

def surrogate_predict(z):
    """Cheap compliance prediction (with scaling)"""
    z_scaled = scaler.transform(z.reshape(1, -1))
    return gp.predict(z_scaled)[0]

LAMBDA = 5.0  # trust penalty weight (tune 2–10)

def surrogate_predict1(z):
    z_scaled = scaler.transform(z.reshape(1, -1))
    
    # GP mean prediction
    mu = gp.predict(z_scaled)[0]
    
    # Distance to nearest training point
    d = np.min(pairwise_distances(z_scaled, gp.X_train_))
    
    # Penalized surrogate objective
    return mu + LAMBDA * d


def real_objective(z):
    """Expensive FEM evaluation"""
    with torch.no_grad():
        rho = vae.decode(
            torch.from_numpy(z).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()
    
    if not is_feasible(rho):
        return 1e6
    
    compliance, _ = fem_physical_compliance(rho)
    return compliance

# CMA-ES with surrogate guidance
es = CMAEvolutionStrategy(
    x0=np.zeros(LATENT_DIM),
    sigma0=0.8,
    inopts={'popsize': POPSIZE}
)

fem_count = 0
best_compliance = float('inf')
best_z = None  # Track best latent vector

print("🚀 Starting surrogate-guided CMA-ES (budget: 200 FEM calls)")
print("-" * 60)

while fem_count < FEM_BUDGET:
    # 1. Propose POPSIZE candidates
    candidates = es.ask()
    
    # 2. Predict compliance for ALL candidates (cheap)
    preds = np.array([surrogate_predict(z) for z in candidates])
    #print(f"Gen {es.countiter:3d} | Surrogate predictions: min={preds.min():.2f}, mean={preds.mean():.2f}, max={preds.max():.2f}")
    # 3. Select TOP K for REAL FEM evaluation
    top_idx = np.argsort(preds)[:FEM_PER_GEN]
    
    # 4. Build FULL fitness array (surrogate + real FEM)
    fitness = preds.copy()
    
    # Replace top-K with REAL FEM values
    for idx in top_idx:
        z = candidates[idx]
        c = real_objective(z)
        print(f"Evaluating candidate {idx} with surrogate: Predicted={preds[idx]:.2f}")
        print(f"Evaluating candidate {idx} with real FEM: Compliance={c:.2f}")
        fitness[idx] = c
        fem_count += 1
        
        # Track global best
        if c < best_compliance:
            best_compliance = c
            best_z = z.copy()  # Save best latent vector
            print(f"✅ FEM #{fem_count:3d} | Compliance={c:.2f} (NEW BEST)")
        else:
            print(f"   FEM #{fem_count:3d} | Compliance={c:.2f}")
    
    # 5. Update CMA-ES with FULL fitness array
    es.tell(candidates, fitness.tolist())
    
    if fem_count >= FEM_BUDGET:
        break

print("-" * 60)
print(f"🏁 Optimization complete!")
print(f"   FEM calls used: {fem_count}/{FEM_BUDGET}")
print(f"   Best compliance: {best_compliance:.2f} N·mm")

# -----------------------------
# SAVE BEST DESIGN (matching vae_cmaes.py workflow)
# -----------------------------
if best_z is not None:
    print("\n💾 Saving best design...")
    
    # Decode best latent vector to density field
    with torch.no_grad():
        best_rho = vae.decode(
            torch.from_numpy(best_z).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()
    
    # Save raw density field
    np.save(os.path.join(RESULTS_DIR, "best_rho.npy"), best_rho)
    print(f"✅ Saved best_rho.npy")
    
    # Run FEM + compute stress
    compliance, u = fem_physical_compliance(best_rho)
    vm_stress = compute_von_mises_stress(best_rho, u)
    
    # Save VTK with density + displacement + stress
    save_fem_to_vtk(
        best_rho,
        u,
        vm_stress,
        filename=os.path.join(RESULTS_DIR, "best_design.vti")
    )
    print(f"✅ Saved best_design.vti with stress visualization")
    
    # Optional: Save post-processed version for visualization
    from scipy.ndimage import binary_closing, binary_opening
    rho_clean = binary_closing((best_rho > 0.5).astype(np.uint8), structure=np.ones((3,3)))
    rho_clean = binary_opening(rho_clean, structure=np.ones((3,3)))
    
    compliance_clean, u_clean = fem_physical_compliance(rho_clean.astype(np.float32))
    vm_clean = compute_von_mises_stress(rho_clean.astype(np.float32), u_clean)
    
    save_fem_to_vtk(
        rho_clean.astype(np.float32),
        u_clean,
        vm_clean,
        filename=os.path.join(RESULTS_DIR, "best_design_clean.vti")
    )
    print(f"✅ Saved best_design_clean.vti (post-processed)")
else:
    print("⚠️ No valid design found!")