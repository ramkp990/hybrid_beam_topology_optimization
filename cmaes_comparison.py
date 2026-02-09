# surrogate_cmaes_comparison.py
import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from vae_model import TopologyVAE
from fem_code import fem_physical_compliance, is_feasible, save_fem_to_vtk
import joblib
from sklearn.metrics import pairwise_distances

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
            B = np.array([
                [-1/xe, 0,  1/xe, 0,  1/xe, 0, -1/xe, 0],
                [0, -1/ye, 0, -1/ye, 0, 1/ye, 0, 1/ye],
                [-1/ye, -1/xe, -1/ye, 1/xe, 1/ye, 1/xe, 1/ye, -1/xe]
            ])
            stress = D @ (B @ u_e)
            sx, sy, txy = stress
            vm[ey, ex] = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    return vm

def save_fem_to_vtk(geom, u, vm_stress, filename, domain_size=(20.0, 10.0)):
    import vtk
    ny, nx = geom.shape
    Lx, Ly = domain_size
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetSpacing(Lx/(nx-1), Ly/(ny-1), 1.0)
    image.SetOrigin(0, 0, 0)
    density = vtk.vtkFloatArray()
    density.SetName("Density")
    for j in range(ny):
        for i in range(nx):
            density.InsertNextValue(float(geom[j, i]))
    image.GetPointData().SetScalars(density)
    disp = vtk.vtkFloatArray()
    disp.SetName("Displacement")
    disp.SetNumberOfComponents(3)
    for j in range(ny):
        for i in range(nx):
            idx = j*nx + i
            disp.InsertNextTuple3(u[2*idx], u[2*idx+1], 0.0)
    image.GetPointData().AddArray(disp)
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

# Configuration
MODEL_PATH = "dataset/merged_vae_train/vae_best.pth"
SURROGATE_PATH = "surrogate/gp_surrogate.pkl"
SCALER_PATH = "surrogate/z_scaler.pkl"
LATENT_DIM = 32
FEM_BUDGET = 250
POPSIZE = 12
FEM_PER_GEN = 8
RESULTS_DIR = "results_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = TopologyVAE(latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
vae.eval()

gp = joblib.load(SURROGATE_PATH)
scaler = joblib.load(SCALER_PATH)

def surrogate_predict(z):
    z_scaled = scaler.transform(z.reshape(1, -1))
    return gp.predict(z_scaled)[0]

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

# Initialize two CMA-ES instances with same seed for fair comparison
np.random.seed(42)
torch.manual_seed(42)

# Surrogate-guided CMA-ES
es_surrogate = CMAEvolutionStrategy(
    x0=np.zeros(LATENT_DIM),
    sigma0=0.8,
    inopts={'popsize': POPSIZE}
)

# Standard CMA-ES (no surrogate)
es_standard = CMAEvolutionStrategy(
    x0=np.zeros(LATENT_DIM),
    sigma0=0.8,
    inopts={'popsize': POPSIZE}
)

# Track results
fem_count_surrogate = 0
best_compliance_surrogate = float('inf')
best_z_surrogate = None

fem_count_standard = 0
best_compliance_standard = float('inf')
best_z_standard = None

print("🚀 Starting comparison: Surrogate vs Standard CMA-ES")
print("-" * 70)

generation = 0
while fem_count_surrogate < FEM_BUDGET and fem_count_standard < FEM_BUDGET:
    generation += 1
    
    # --- SURROGATE-GUIDED CMA-ES ---
    candidates_surrogate = es_surrogate.ask()
    preds_surrogate = np.array([surrogate_predict(z) for z in candidates_surrogate])
    top_idx_surrogate = np.argsort(preds_surrogate)[:FEM_PER_GEN]
    
    fitness_surrogate = preds_surrogate.copy()
    for idx in top_idx_surrogate:
        z = candidates_surrogate[idx]
        c = real_objective(z)
        fitness_surrogate[idx] = c
        fem_count_surrogate += 1
        
        if c < best_compliance_surrogate:
            best_compliance_surrogate = c
            best_z_surrogate = z.copy()
    
    es_surrogate.tell(candidates_surrogate, fitness_surrogate.tolist())
    
    # --- STANDARD CMA-ES (NO SURROGATE) ---
    candidates_standard = es_standard.ask()
    fitness_standard = []
    for z in candidates_standard:
        c = real_objective(z)
        fitness_standard.append(c)
        fem_count_standard += 1
        
        if c < best_compliance_standard:
            best_compliance_standard = c
            best_z_standard = z.copy()
    
    es_standard.tell(candidates_standard, fitness_standard)
    
    # Print progress every 5 generations
    if generation % 5 == 0:
        print(f"Gen {generation:3d} | "
              f"Surrogate: {best_compliance_surrogate:6.2f} ({fem_count_surrogate:3d} FEM) | "
              f"Standard: {best_compliance_standard:6.2f} ({fem_count_standard:3d} FEM)")

print("-" * 70)
print("🏁 Optimization complete!")
print(f"Surrogate CMA-ES: {best_compliance_surrogate:.2f} N·mm ({fem_count_surrogate} FEM calls)")
print(f"Standard CMA-ES:  {best_compliance_standard:.2f} N·mm ({fem_count_standard} FEM calls)")

# Calculate similarity between best designs
if best_z_surrogate is not None and best_z_standard is not None:
    z_diff = np.linalg.norm(best_z_surrogate - best_z_standard)
    print(f"Latent space distance between best designs: {z_diff:.4f}")
    
    # Decode both designs
    with torch.no_grad():
        rho_surrogate = vae.decode(
            torch.from_numpy(best_z_surrogate).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()
        rho_standard = vae.decode(
            torch.from_numpy(best_z_standard).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()
    
    # Pixel-wise similarity
    pixel_diff = np.mean(np.abs(rho_surrogate - rho_standard))
    print(f"Pixel-wise difference between designs: {pixel_diff:.4f}")

# Save both best designs
def save_design(z, name_suffix):
    with torch.no_grad():
        rho = vae.decode(
            torch.from_numpy(z).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()
    
    np.save(os.path.join(RESULTS_DIR, f"best_rho_{name_suffix}.npy"), rho)
    
    compliance, u = fem_physical_compliance(rho)
    vm_stress = compute_von_mises_stress(rho, u)
    save_fem_to_vtk(
        rho, u, vm_stress,
        filename=os.path.join(RESULTS_DIR, f"best_design_{name_suffix}.vti")
    )
    print(f"✅ Saved {name_suffix} design")

if best_z_surrogate is not None:
    save_design(best_z_surrogate, "surrogate")
if best_z_standard is not None:
    save_design(best_z_standard, "standard")

print(f"\nResults saved to {RESULTS_DIR}/")