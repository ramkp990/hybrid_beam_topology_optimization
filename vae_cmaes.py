# vae_cmaes.py
'''
import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy

# Import your modules
from vae_model import TopologyVAE  # Put your VAE class in a separate file
from fem_code import fem_physical_compliance, is_feasible  # Your FEM/feasibility

# Configuration
MODEL_PATH = "dataset/merged_vae_train/vae_best.pth"
LATENT_DIM = 32
FEM_BUDGET = 500
LOAD_VALUE = -100.0

def post_process_rho(rho):
    # Convert to binary
    rho_binary = (rho > 0.5).astype(np.uint8)
    
    # Clean up with morphological operations
    from scipy.ndimage import binary_closing, binary_opening
    rho_clean = binary_closing(rho_binary, structure=np.ones((3,3)))
    rho_clean = binary_opening(rho_clean, structure=np.ones((3,3)))
    
    return rho_clean


def main():
    # Load VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopologyVAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded VAE from {MODEL_PATH}")

    # CMA-ES objective
    def objective(z):
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            rho = model.decode(z_tensor).squeeze().cpu().numpy()
        
        if not is_feasible(rho):
            return 1e6
        
        compliance, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)
        return compliance

    # Run CMA-ES
    es = CMAEvolutionStrategy(
        x0=np.zeros(LATENT_DIM),
        sigma0=0.6,
        inopts={'popsize': 12}
    )

    best_compliance = float('inf')
    while es.countevals < FEM_BUDGET:
        solutions = es.ask()
        fitnesses = [objective(z) for z in solutions]
        es.tell(solutions, fitnesses)
        
        current_best = min(fitnesses)
        if current_best < best_compliance:
            best_compliance = current_best
            print(f"Eval {es.countevals}: Best compliance = {best_compliance:.2f}")

    # Save best design
    best_z = es.result.xbest
    with torch.no_grad():
        best_rho = model.decode(
            torch.from_numpy(best_z).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()

    compliance, _ = fem_physical_compliance(
        best_rho, 
        save_vtk=True, 
        vtk_filename="results/best_design.vti"
    )
    print(f"✅ Final compliance: {compliance:.2f}")
    np.save("results/best_rho.npy", best_rho)
    clean_rho = post_process_rho(best_rho)
    compliance, _ = fem_physical_compliance(
        clean_rho, 
        save_vtk=True, 
        vtk_filename="results/best_design_clean.vti"
    )
    print(f"✅ Final compliance: {compliance:.2f}")
    np.save("results/clean_rho.npy", clean_rho)
if __name__ == "__main__":
    main()

'''

# vae_cmaes.py
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


# vae_cmaes.py

import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy

from vae_model import TopologyVAE
from fem_code import fem_physical_compliance, is_feasible

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "dataset/merged_vae_train/vae_best.pth"
LATENT_DIM = 32
FEM_BUDGET = 500
LOAD_VALUE = -100.0

POP_SIZE = 12
SAVE_EVERY = 5          # save every generation
MAX_SAVED = 50          # hard cap on saved designs

RESULTS_DIR = "results"
DENSITY_DIR = os.path.join(RESULTS_DIR, "densities")
os.makedirs(DENSITY_DIR, exist_ok=True)

# -----------------------------
# Post-processing
# -----------------------------
def post_process_rho(rho):
    from scipy.ndimage import binary_closing, binary_opening
    rho_binary = (rho > 0.5).astype(np.uint8)
    rho_clean = binary_closing(rho_binary, structure=np.ones((3,3)))
    rho_clean = binary_opening(rho_clean, structure=np.ones((3,3)))
    return rho_clean

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    model = TopologyVAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded VAE from {MODEL_PATH}")

    # Objective function
    def objective(z):
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            rho = model.decode(z_tensor).squeeze().cpu().numpy()

        if not is_feasible(rho):
            return 1e6

        compliance, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)
        return compliance

    # CMA-ES
    es = CMAEvolutionStrategy(
        x0=np.zeros(LATENT_DIM),
        sigma0=0.9,
        inopts={'popsize': POP_SIZE}
    )

    best_compliance = float("inf")
    saved_count = 0
    generation = 0

    # -----------------------------
    # Optimization loop
    # -----------------------------
    while es.countevals < FEM_BUDGET:
        generation += 1

        solutions = es.ask()
        fitnesses = [objective(z) for z in solutions]
        es.tell(solutions, fitnesses)

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_val = fitnesses[gen_best_idx]
        gen_best_z = solutions[gen_best_idx]

        print(
            f"Gen {generation:03d} | "
            f"Evals {es.countevals:03d} | "
            f"Best compliance {gen_best_val:.2f}"
        )

        # Save representative density (best of generation)
        if generation % SAVE_EVERY == 0 and saved_count < MAX_SAVED:
            with torch.no_grad():
                rho = model.decode(
                    torch.from_numpy(gen_best_z)
                    .float().unsqueeze(0).to(device)
                ).squeeze().cpu().numpy()

            compliance, u = fem_physical_compliance(rho, load_value=LOAD_VALUE)
            vm = compute_von_mises_stress(rho, u)

            save_fem_to_vtk(
                rho,
                u,
                vm,
                filename=f"results/vti/gen_{generation:03d}.vti"
            )

            saved_count += 1

        # Track global best
        if gen_best_val < best_compliance:
            best_compliance = gen_best_val

    # -----------------------------
    # Final best design
    # -----------------------------
    best_z = es.result.xbest
    with torch.no_grad():
        best_rho = model.decode(
            torch.from_numpy(best_z).float().unsqueeze(0).to(device)
        ).squeeze().cpu().numpy()

    # Save raw
    np.save(os.path.join(RESULTS_DIR, "best_rho.npy"), best_rho)

    compliance, _ = fem_physical_compliance(
        best_rho,
        save_vtk=True,
        vtk_filename=os.path.join(RESULTS_DIR, "best_design.vti")
    )
    print(f"✅ Final compliance (raw): {compliance:.2f}")


    compliance, displacement = fem_physical_compliance(
        best_rho,
        load_value=LOAD_VALUE)
    u = displacement
    vm = compute_von_mises_stress(best_rho, u)
    rho = best_rho
    save_fem_to_vtk(
        rho,
        u,
        vm,
        filename=f"results/vti/best_design.vti"
    )

    print(f"✅ Final compliance (clean): {compliance:.2f}")

if __name__ == "__main__":
    main()
