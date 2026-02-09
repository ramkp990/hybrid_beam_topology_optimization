
import os
import numpy as np
import torch
from cma import CMAEvolutionStrategy

from vae_model import TopologyVAE
from fem_code import fem_physical_compliance, is_feasible



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


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "dataset/merged_vae_train/vae_best.pth"
LATENT_DIM = 32
FEM_BUDGET = 800
LOAD_VALUE = -100.0
POP_SIZE = 12
SAVE_EVERY = 5
RESULTS_DIR = "results"
CMAES_DATASET_DIR = "dataset/dataset_cmaes"
os.makedirs(CMAES_DATASET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# Dataset Appender
# -----------------------------
def append_to_dataset(new_rho_list, new_metadata_list, dataset_dir):
    """
    Append new designs to dataset in a way that is
    consistent with surrogate training.
    """
    if not new_rho_list:
        return

    rho_path = os.path.join(dataset_dir, "rho_smooth.npy")
    meta_path = os.path.join(dataset_dir, "metadata_with_compliance.npy")

    # ---- Load existing ----
    if os.path.exists(rho_path):
        rho_existing = np.load(rho_path)
        print(f"Loaded {len(rho_existing)} existing rho designs")
    else:
        rho_existing = np.empty((0, 64, 64), dtype=np.float32)

    if os.path.exists(meta_path):
        meta_existing = np.load(meta_path, allow_pickle=True)
    else:
        meta_existing = np.empty((0,), dtype=[
            ('area_fraction', np.float32),
            ('connected', bool),
            ('main_component_area', np.int32),
            ('touches_left', bool),
            ('touches_load', bool),
            ('compliance', np.float32)
        ])

    # ---- Prepare new entries ----
    rho_new = np.array(new_rho_list, dtype=np.float32)

    meta_new = np.array([
        (
            m["area_fraction"],
            m["connected"],
            int(m["area_fraction"] * 64 * 64),
            m["touches_left"],
            m["touches_load"],
            m["compliance"]
        )
        for m in new_metadata_list
    ], dtype=meta_existing.dtype)

    # ---- Concatenate ----
    rho_all = np.concatenate([rho_existing, rho_new], axis=0)
    meta_all = np.concatenate([meta_existing, meta_new], axis=0)

    # ---- Save ----
    np.save(rho_path, rho_all)
    np.save(meta_path, meta_all)

    print(f"✅ Dataset updated: {len(rho_all)} total designs (+{len(rho_new)})")


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TopologyVAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded VAE from {MODEL_PATH}")

    # Track only NEW good designs
    new_low_compliance_rho = []
    new_low_compliance_metadata = []

    def objective(z):
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            rho = model.decode(z_tensor).squeeze().cpu().numpy()

        if not is_feasible(rho):
            return 1e6

        compliance, _ = fem_physical_compliance(rho, load_value=LOAD_VALUE)

        if compliance < 10.0:
            new_low_compliance_rho.append(rho.copy())
            new_low_compliance_metadata.append({
                "compliance": float(compliance),
                "area_fraction": float((rho > 0.5).sum() / (64 * 64)),
                "connected": True,
                "touches_left": bool((rho[:, 0] > 0.5).any()),
                "touches_load": bool(rho[32, 63] > 0.5)
            })
            print(f"🌟 NEW DESIGN FOUND | Compliance = {compliance:.2f}")

        return compliance

    es = CMAEvolutionStrategy(
        x0=np.zeros(LATENT_DIM),
        sigma0=0.9,
        inopts={'popsize': POP_SIZE}
    )

    best_compliance = float("inf")

    while es.countevals < FEM_BUDGET:
        solutions = es.ask()
        fitnesses = [objective(z) for z in solutions]
        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        best_compliance = min(best_compliance, gen_best)

        print(f"Evals {es.countevals:04d} | Best this gen {gen_best:.2f} | Overall {best_compliance:.2f}")

    # ---- Append discoveries ----
    if new_low_compliance_rho:
        append_to_dataset(
            new_low_compliance_rho,
            new_low_compliance_metadata,
            CMAES_DATASET_DIR
        )
    else:
        print("⚠️ No new designs with compliance < 15 found")


if __name__ == "__main__":
    main()