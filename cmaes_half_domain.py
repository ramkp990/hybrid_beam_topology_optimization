import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy

# Import YOUR model
from evaluate_vae_half_domain import HalfTopologyVAE

# FEM utilities (same as before)
from fem_code import fem_physical_compliance, is_feasible

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


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "dataset_half/merged_vae_train_half_domain/vae_best.pth"

LATENT_DIM = 64
FEM_BUDGET = 500
LOAD_VALUE = -100.0

POP_SIZE = 12
SIGMA0 = 0.2

SAVE_EVERY = 1
MAX_SAVED = 600

RESULTS_DIR = "results_half"
VTI_DIR = os.path.join(RESULTS_DIR, "vti")

os.makedirs(VTI_DIR, exist_ok=True)

# -----------------------------
# Half → Full reconstruction
# -----------------------------
def reconstruct_full_from_half(rho_half):
    """
    rho_half: (32,64)
    returns: (64,64)
    """
    return np.vstack([rho_half, np.flipud(rho_half)])


# -----------------------------
# Post-processing
# -----------------------------
def post_process_rho(rho):
    from scipy.ndimage import binary_closing, binary_opening

    rho_binary = (rho > 0.5).astype(np.uint8)
    rho_clean = binary_closing(rho_binary, structure=np.ones((3, 3)))
    rho_clean = binary_opening(rho_clean, structure=np.ones((3, 3)))

    return rho_clean


# -----------------------------
# Load model
# -----------------------------
def load_model(device):
    model = HalfTopologyVAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"✅ Loaded Half-VAE from {MODEL_PATH}")
    return model


# -----------------------------
# Decode latent → full density
# -----------------------------
def decode_to_full(model, z, device):
    with torch.no_grad():
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)

        rho_half = model.decode(z_tensor).squeeze().cpu().numpy()
        rho_full = reconstruct_full_from_half(rho_half)

    return rho_full


# -----------------------------
# Objective function
# -----------------------------
def make_objective(model, device):
    def objective(z):

        rho = decode_to_full(model, z, device)

        rho = post_process_rho(rho)

        if not is_feasible(rho):
            return 1e6

        compliance, _ = fem_physical_compliance(
            rho,
            load_value=LOAD_VALUE
        )

        return compliance

    return objective


# -----------------------------
# MAIN
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    objective = make_objective(model, device)

    # -----------------------------
    # CMA-ES
    # -----------------------------
    es = CMAEvolutionStrategy(
        x0=np.zeros(LATENT_DIM),
        sigma0=SIGMA0,
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
            f"Best compliance {gen_best_val:.4f}"
        )

        # -----------------------------
        # Save representative design
        # -----------------------------
        if generation % SAVE_EVERY == 0 and saved_count < MAX_SAVED:

            rho = decode_to_full(model, gen_best_z, device)

            compliance, u = fem_physical_compliance(
                rho,
                load_value=LOAD_VALUE
            )

            vm = compute_von_mises_stress(rho, u)

            save_fem_to_vtk(
                rho,
                u,
                vm,
                filename=os.path.join(
                    VTI_DIR,
                    f"gen_{generation:03d}.vti"
                )
            )

            saved_count += 1

        # -----------------------------
        # Track global best
        # -----------------------------
        if gen_best_val < best_compliance:
            best_compliance = gen_best_val

    # -----------------------------
    # Final best
    # -----------------------------
    best_z = es.result.xbest

    best_rho = decode_to_full(model, best_z, device)

    np.save(
        os.path.join(RESULTS_DIR, "best_rho.npy"),
        best_rho
    )

    compliance, u = fem_physical_compliance(
        best_rho,
        load_value=LOAD_VALUE
    )

    vm = compute_von_mises_stress(best_rho, u)

    save_fem_to_vtk(
        best_rho,
        u,
        vm,
        filename=os.path.join(
            VTI_DIR,
            "best_design.vti"
        )
    )

    print(f"\n✅ Final compliance: {compliance:.4f}")


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    main()
