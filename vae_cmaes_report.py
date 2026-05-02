
# vae_cmaes.py

def compute_von_mises_stress(geom, u, p=3.0):
    """
    Compute von Mises stress per element using
    - Q4 bilinear elements
    - plane stress
    - center-point evaluation
    - SIMP density scaling

    Parameters
    ----------
    geom : (ny, nx) ndarray
        Density field (node-based)
    u : (2 * nx * ny,) ndarray
        Global displacement vector
    p : float
        SIMP penalization exponent

    Returns
    -------
    vm : (ny-1, nx-1) ndarray
        Element-wise von Mises stress
    """

    ny, nx = geom.shape
    nel_y, nel_x = ny - 1, nx - 1

    # Domain size (must match FEM)
    Lx, Ly = 20.0, 10.0
    xe = Lx / nel_x
    ye = Ly / nel_y

    # Material parameters (plane stress)
    E = 2.1e5
    nu = 0.3

    D = (E / (1 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0]
    ])

    # Shape function derivatives at element center (ξ=0, η=0)
    dN_dxi = np.array([-0.25,  0.25,  0.25, -0.25])
    dN_deta = np.array([-0.25, -0.25,  0.25,  0.25])

    vm = np.zeros((nel_y, nel_x))

    def node_id(i, j):
        return i * nx + j

    for ey in range(nel_y):
        for ex in range(nel_x):

            # Element nodes (counter-clockwise)
            nodes = [
                node_id(ey,   ex),
                node_id(ey,   ex+1),
                node_id(ey+1, ex+1),
                node_id(ey+1, ex)
            ]

            # Element displacement vector
            dofs = []
            for n in nodes:
                dofs.extend([2*n, 2*n+1])
            u_e = u[dofs]

            # Jacobian (rectangular, constant)
            J = np.array([
                [xe / 2, 0],
                [0, ye / 2]
            ])
            invJ = np.linalg.inv(J)

            # Gradients in physical coordinates
            dN_dx = invJ[0, 0] * dN_dxi
            dN_dy = invJ[1, 1] * dN_deta

            # Strain-displacement matrix B
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i]     = dN_dx[i]
                B[1, 2*i+1]   = dN_dy[i]
                B[2, 2*i]     = dN_dy[i]
                B[2, 2*i+1]   = dN_dx[i]

            # SIMP density interpolation (element average)
            rho_e = np.mean(geom[ey:ey+2, ex:ex+2])
            E_e = (rho_e ** p) * E
            D_e = (E_e / (1 - nu**2)) * np.array([
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1 - nu) / 2.0]
            ])

            # Stress
            stress = D_e @ (B @ u_e)
            sx, sy, txy = stress

            # Von Mises stress (plane stress)
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

from evaluate_vae_report_com import TopologyVAE
from fem_code import fem_physical_compliance, is_feasible

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "dataset/merged_vae_train_report/vae_best.pth"
LATENT_DIM = 32
FEM_BUDGET = 1000
LOAD_VALUE = -100.0

POP_SIZE = 12
SAVE_EVERY = 1          # save every generation
MAX_SAVED = 600          # hard cap on saved designs

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
    global fem_calls, feasible_count, infeasible_count
    total_evaluations = 0
    fem_calls = 0
    feasible_count = 0
    infeasible_count = 0

    # Load VAE
    model = TopologyVAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded VAE from {MODEL_PATH}")

    # Objective function
    def objective(z):
        global fem_calls, feasible_count, infeasible_count
        
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            rho = model.decode(z_tensor).squeeze().cpu().numpy()

        if not is_feasible(rho):
            infeasible_count += 1
            return 1e6

        # Only reach here for feasible designs
        feasible_count += 1
        fem_calls += 1
        compliance, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)
        print(f"    [Eval {total_evaluations:03d}] Compliance: {compliance:.2f}")
        return compliance

    # CMA-ES
    es = CMAEvolutionStrategy(
        x0=np.zeros(LATENT_DIM),
        sigma0=0.2,
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
        fitnesses = []
        for z in solutions:
            total_evaluations += 1
            fitnesses.append(objective(z))
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
            # save every z and complaice as npz to visualize the search using umap
            np.savez(
                os.path.join(DENSITY_DIR, f"gen_{generation:03d}.npz"),
                z=gen_best_z,
                compliance=compliance
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
    # save best z as npz for later analysis
    np.savez(os.path.join(RESULTS_DIR, "best_z.npz"), z=best_z)
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
    print(f"\n📊 Optimization Statistics:")
    print(f"Total CMA-ES evaluations: {total_evaluations}")
    print(f"Actual FEM calls: {fem_calls}")
    print(f"Feasible designs: {feasible_count}")
    print(f"Infeasible designs: {infeasible_count}")
    print(f"FEM efficiency: {fem_calls/total_evaluations*100:.1f}%")

if __name__ == "__main__":
    main()
