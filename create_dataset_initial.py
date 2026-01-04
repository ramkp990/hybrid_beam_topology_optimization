
import os
import math
import numpy as np
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count
from scipy.special import expit  # sigmoid
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import scipy.sparse as sp
import scipy.sparse.linalg as spla


N = 5
GRID = 64
SAVE_DIR = "dataset_cantilever_sym6_mmctest"
os.makedirs(SAVE_DIR, exist_ok=True)

MIN_AREA_FRAC = 0.10
MAX_AREA_FRAC = 0.50
MIN_AREA = int(MIN_AREA_FRAC * GRID * GRID)
MAX_AREA = int(MAX_AREA_FRAC * GRID * GRID)

BATCH_SIZE = 10
LARGE_COMPONENT_RATIO = 0.90
LOAD_POINT_X = GRID - 1
LOAD_POINT_Y = GRID // 2
LOAD_POINT_WINDOW = 3
M_MMC = 6
HEAVISIDE_CUTOFF = 0.0

global counter
counter = 0 

SIGMOID_K = 50.0  # for smoothed rho

BOUNDS = {
    "left_x_range": (0.01, 0.20),
    "mid_x_range": (0.25, 0.75),
    "right_x_range": (0.80, 0.99),
    "y_top_range": (0.50, 0.85),
    "theta_left": (-75.0, 75.0),
    "theta_mid": (-75.0, 75.0),
    "theta_right": (-75.0, 75.0),
    "L_range": (0.25, 0.55),
    "t_range": (0.015, 0.08),
}

xs = np.linspace(0.0, 1.0, GRID)
ys = np.linspace(0.0, 1.0, GRID)
Xg, Yg = np.meshgrid(xs, ys)

# -----------------------------
# MMC local phi evaluation (vectorized)
# -----------------------------
def compute_local_phi(xc, yc, theta_deg, L_half, t_full, m=M_MMC):
    """
    Compute phi_i(u) on the grid for a single MMC beam using equation (4) in the paper.
    - xc, yc: center (normalized)
    - theta_deg: orientation in degrees
    - L_half: half-length = li/2 in paper notation
    - t_full: full thickness ti; denominator uses ti/2
    Return: phi array shape (GRID, GRID), float
    """
    theta = math.radians(float(theta_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    # local coordinates relative to center
    Xr = c * (Xg - xc) + s * (Yg - yc)            # parallel projection (along beam axis)
    Yr = -s * (Xg - xc) + c * (Yg - yc)           # perpendicular projection

    # denominators as in eq. (4): li/2 -> L_half ; ti/2 -> t_full/2
    denom_parallel = (L_half + 1e-12)            # avoid zero
    denom_perp = (t_full / 2.0 + 1e-12)

    # super-elliptic terms (absolute value raised to power m)
    term_p = np.abs(Xr / denom_parallel) ** m
    term_q = np.abs(Yr / denom_perp) ** m

    phi = - (term_p + term_q - 1.0)   # as in paper (negative outside, >=0 inside)
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

# -----------------------------
# Helpers: endpoints (for diagnostics only)
# -----------------------------
def beam_endpoints_pixels(xc, yc, theta_deg, L_half):
    theta = math.radians(float(theta_deg))
    dx = math.cos(theta)
    dy = math.sin(theta)
    x1 = float(xc) - float(L_half) * dx
    y1 = float(yc) - float(L_half) * dy
    x2 = float(xc) + float(L_half) * dx
    y2 = float(yc) + float(L_half) * dy
    # clip and convert
    x1p = int(round(max(0.0, min(1.0, x1)) * (GRID - 1)))
    y1p = int(round(max(0.0, min(1.0, y1)) * (GRID - 1)))
    x2p = int(round(max(0.0, min(1.0, x2)) * (GRID - 1)))
    y2p = int(round(max(0.0, min(1.0, y2)) * (GRID - 1)))
    return (x1p, y1p), (x2p, y2p)

# -----------------------------
# Sampling functions using RNG passed in
# -----------------------------
def sample_beam_left(rng):
    xc = rng.uniform(*BOUNDS["left_x_range"])
    yc = rng.uniform(*BOUNDS["y_top_range"])
    theta = rng.uniform(*BOUNDS["theta_left"])
    L = rng.uniform(*BOUNDS["L_range"])
    t = rng.uniform(*BOUNDS["t_range"])
    return float(xc), float(yc), float(theta), float(L), float(t)

def sample_beam_middle(rng):
    xc = rng.uniform(*BOUNDS["mid_x_range"])
    yc = rng.uniform(*BOUNDS["y_top_range"])
    theta = rng.uniform(*BOUNDS["theta_mid"])
    L = rng.uniform(*BOUNDS["L_range"])
    t = rng.uniform(*BOUNDS["t_range"])
    return float(xc), float(yc), float(theta), float(L), float(t)

def sample_beam_right(rng):
    xc = rng.uniform(*BOUNDS["right_x_range"])
    yc = rng.uniform(*BOUNDS["y_top_range"])
    theta = rng.uniform(*BOUNDS["theta_right"])
    L = rng.uniform(*BOUNDS["L_range"])
    t = rng.uniform(*BOUNDS["t_range"])
    return float(xc), float(yc), float(theta), float(L), float(t)

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

'''
def fem_physical_compliance(geom, load_node=None, load_value=-100.0, save_vtk=False, vtk_filename=None):
    """
    Compute physical compliance (N·mm) for a geometry grid.
    - geom: (64, 64) array. Values in [0,1] (smooth) or {0,1} (binary).
    - Paper settings: 20x10 mm domain, E=2.1e5 MPa, nu=0.3, 1% void stiffness.
    - If save_vtk=True, saves displacement and density to a VTK file for ParaView.
    Returns: compliance (float in N·mm), displacement vector.
    """
    ny, nx = geom.shape  # 64x64 nodes → 63x63 elements
    nel_y, nel_x = ny - 1, nx - 1
    nelems = nel_x * nel_y
    nnodes = ny * nx
    ndof = 2 * nnodes

    # Physical domain: 20 mm x 10 mm
    xe = 20.0 / nel_x  # mm
    ye = 10.0 / nel_y  # mm

    # Material properties
    E_solid = 2.1e5  # MPa
    E_void = 0.01 * E_solid
    nu = 0.3

    # Precompute element stiffness
    Ke_solid = element_stiffness_plane_stress(E_solid, nu, xe, ye)
    Ke_void = element_stiffness_plane_stress(E_void, nu, xe, ye)

    # Assemble global stiffness matrix
    rows, cols, data = [], [], []
    def node_id(iy, ix):
        return iy * nx + ix

    # Element loop
    for ey in range(nel_y):
        for ex in range(nel_x):
            # Use average density to decide if solid (works for smooth rho)
            n1 = geom[ey,   ex]
            n2 = geom[ey,   ex+1]
            n3 = geom[ey+1, ex+1]
            n4 = geom[ey+1, ex]
            avg_rho = (n1 + n2 + n3 + n4) / 4.0
            #is_solid = avg_rho > 0.5  # Threshold for stiffness assignment

            #Ke = Ke_solid if is_solid else Ke_void
            p = 3.0
            E_e = E_void + (avg_rho ** p) * (E_solid - E_void)
            Ke = (E_e / E_solid) * Ke_solid

            nodes = [node_id(ey, ex), node_id(ey, ex+1),
                     node_id(ey+1, ex+1), node_id(ey+1, ex)]
            dofs = []
            for n in nodes:
                dofs.extend([2*n, 2*n+1])

            for i in range(8):
                for j in range(8):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    data.append(Ke[i, j])

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Boundary conditions: fix left edge (x=0)
    fixed = []
    for iy in range(ny):
        nid = node_id(iy, 0)
        fixed.extend([2*nid, 2*nid+1])
    free_dofs = np.setdiff1d(np.arange(ndof), fixed)

    # Load application
    if load_node is None:
        load_node = (ny//2, nx - 1)
    fy = np.zeros(ndof)
    lnid = node_id(*load_node)
    fy[2*lnid + 1] = load_value

    # Solve
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    f_f = fy[free_dofs]
    try:
        u_f = spla.spsolve(K_ff, f_f)
    except:
        reg = 1e-8 * sp.eye(K_ff.shape[0])
        u_f = spla.spsolve(K_ff + reg, f_f)

    u = np.zeros(ndof)
    u[free_dofs] = u_f
    compliance = float(fy @ u)

    # Optional: Save VTK for ParaView
    if save_vtk:
        if vtk_filename is None:
            import time
            vtk_filename = f"design_{int(time.time())}.vtk"
        save_fem_to_vtk(geom, u, vtk_filename, domain_size=(20.0, 10.0))
    print("here")
    return compliance, u
'''

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
# Validator: now returns phi and rho too
# -----------------------------
def validate_candidate(seed):
    global counter

    rng = np.random.RandomState(seed)
    b1 = sample_beam_left(rng)
    b2 = sample_beam_middle(rng)
    b3 = sample_beam_right(rng)
    top_beams = [b1, b2, b3]

    phi_global, phi_components = build_phi_global(top_beams)
    grid_full = phi_to_binary(phi_global)  # binary for validation
    rho_smooth = expit(SIGMOID_K * phi_global).astype(np.float32)
    grid_for_feasibility = (rho_smooth > 0.5).astype(np.uint8)
    #area = int(grid_full.sum())
    area = int(grid_for_feasibility.sum())
    if area < MIN_AREA or area > MAX_AREA:
        return None

    #labeled, num = label(grid_full, structure=np.ones((3,3), dtype=np.int8))
    labeled, num = label(grid_for_feasibility, structure=np.ones((3,3), dtype=np.int8))
    if num == 0:
        return None
    sizes = [(labeled == k).sum() for k in range(1, num+1)]
    largest = max(sizes)
    if largest < LARGE_COMPONENT_RATIO * area:
        return None
    main_idx = 1 + sizes.index(largest)
    main_mask = (labeled == main_idx)

    if main_mask[:, 0].sum() == 0:
        return None  # No material on left edge → not connected to support

    # ✅ CHECK: Is load point inside main component?
    if main_mask[LOAD_POINT_Y, LOAD_POINT_X] == 0:
        return None  # Load point not in main component

        
    # Fix: use realistic MIN_BEAM_OVERLAP = 5

    MIN_BEAM_OVERLAP = 5
    comp_masks = [phi_to_binary(pc) for pc in phi_components]

    for cm in comp_masks:
        if (cm & main_mask).sum() < MIN_BEAM_OVERLAP:
            return None

    #if main_mask[:, 0].sum() == 0:
        #return None

    #if main_mask[LOAD_POINT_Y, LOAD_POINT_X] == 0:
        #return None

    #y_low = max(0, LOAD_POINT_Y - LOAD_POINT_WINDOW)
    #y_high = min(GRID, LOAD_POINT_Y + LOAD_POINT_WINDOW + 1)
    #if main_mask[y_low:y_high, LOAD_POINT_X].sum() == 0:
        #return None

    # Compute smoothed rho
    rho_smooth = expit(SIGMOID_K * phi_global).astype(np.float32)

    # FEM on rho_smooth, NOT binary grid
    compliance, accept_c = fem_check_threshold(rho_smooth, 30)
    print(compliance)
    #print(compliance)
    if not accept_c:
        return None

    symmetry_axis_y = GRID // 2
    #if main_mask[max(0, symmetry_axis_y - 1):min(GRID, symmetry_axis_y + 2), LOAD_POINT_X].sum() < 1:
        #return None

    params_top = np.array(list(b1) + list(b2) + list(b3), dtype=np.float32)
    
    # Compute smoothed rho
    rho_smooth = expit(SIGMOID_K * phi_global).astype(np.float32)
    
    metadata = {
    "area_fraction": float(area) / (GRID * GRID),
    "connected": True,  # Since it passed all checks
    "main_component_area": int(largest),
    "touches_left": bool(main_mask[:, 0].sum() > 0),
    "touches_load": bool(main_mask[LOAD_POINT_Y, LOAD_POINT_X] > 0),
    "compliance": float(compliance),
    }
    # Return 4 values: params, binary, phi, rho
    #counter = counter + 1
    #print(counter)
    #return params_top, grid_full, phi_global.astype(np.float32), rho_smooth, metadata
    return params_top, grid_for_feasibility, phi_global.astype(np.float32), rho_smooth, metadata

PROGRESS_FILE = os.path.join(SAVE_DIR, "progress.txt")

def load_seed_offset():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_seed_offset(offset):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(offset))

'''
def generate_dataset(num_valid=N, batch_size=BATCH_SIZE):
    params_list = []
    geom_list = []
    phi_list = []
    rho_list = []
    metadata_list = [] 
    count = 0
    seed_offset = 0
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        while count < num_valid:
            seeds = [seed_offset + i for i in range(batch_size)]
            seed_offset += batch_size
            results = pool.map(validate_candidate, seeds)
            for r in results:
                if r is not None and count < num_valid:
                    params, grid, phi, rho, meta = r
                    params_list.append(params)
                    geom_list.append(grid)
                    phi_list.append(phi)
                    rho_list.append(rho)
                    metadata_list.append(meta) 
                    count += 1
                    if count % 5 == 0:
                        print(f"Biased: Accepted {count}/{num_valid}")
                if count >= num_valid:
                    break

    # Save
    params_arr = np.vstack(params_list).astype(np.float32)
    geom_arr = np.stack(geom_list).astype(np.uint8)
    phi_arr = np.stack(phi_list).astype(np.float32)
    rho_arr = np.stack(rho_list).astype(np.float32)
    meta_array = np.array([
        (
            m["area_fraction"],
            m["connected"],
            m["main_component_area"],
            m["touches_left"],
            m["touches_load"]
        )
        for m in metadata_list
    ], dtype=[
        ('area_fraction', np.float32),
        ('connected', bool),
        ('main_component_area', np.int32),
        ('touches_left', bool),
        ('touches_load', bool)
    ])

    np.save(os.path.join(SAVE_DIR, "metadata.npy"), meta_array)
    np.save(os.path.join(SAVE_DIR, "params.npy"), params_arr)
    np.save(os.path.join(SAVE_DIR, "geometry.npy"), geom_arr)
    np.save(os.path.join(SAVE_DIR, "phi.npy"), phi_arr)          # NEW
    np.save(os.path.join(SAVE_DIR, "rho_smooth.npy"), rho_arr)   # NEW

    print("✅ Saved to", SAVE_DIR)
    print("params.npy shape:", params_arr.shape)
    print("geometry.npy shape:", geom_arr.shape)
    print("phi.npy shape:", phi_arr.shape)
    print("rho_smooth.npy shape:", rho_arr.shape)
    return params_arr, geom_arr, phi_arr, rho_arr
'''

def generate_dataset(num_valid=N, batch_size=BATCH_SIZE):
    params_list = []
    geom_list = []
    phi_list = []
    rho_list = []
    metadata_list = [] 
    count = 0
    seed_offset = load_seed_offset()  # ← Start from last saved offset
    total_tried = seed_offset       # ← Track total candidates tried

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        while count < num_valid:
            seeds = [seed_offset + i for i in range(batch_size)]
            seed_offset += batch_size
            total_tried += batch_size

            results = pool.map(validate_candidate, seeds)
            for r in results:
                if r is not None and count < num_valid:
                    params, grid, phi, rho, meta = r
                    params_list.append(params)
                    geom_size = len(geom_list)
                    geom_list.append(grid)
                    phi_list.append(phi)
                    rho_list.append(rho)
                    metadata_list.append(meta) 
                    count += 1
                    print(f"Biased: Accepted {count}/{num_valid}, Tried {total_tried}")

                if count >= num_valid:
                    break

            # Save progress after each batch
            save_seed_offset(seed_offset)

    # Save final dataset
    if count > 0:
        params_arr = np.vstack(params_list).astype(np.float32)
        geom_arr = np.stack(geom_list).astype(np.uint8)
        phi_arr = np.stack(phi_list).astype(np.float32)
        rho_arr = np.stack(rho_list).astype(np.float32)
        meta_array = np.array([
            (
                m["area_fraction"],
                m["connected"],
                m["main_component_area"],
                m["touches_left"],
                m["touches_load"]
            )
            for m in metadata_list
        ], dtype=[
            ('area_fraction', np.float32),
            ('connected', bool),
            ('main_component_area', np.int32),
            ('touches_left', bool),
            ('touches_load', bool)
        ])

        np.save(os.path.join(SAVE_DIR, "metadata.npy"), meta_array)
        np.save(os.path.join(SAVE_DIR, "params.npy"), params_arr)
        np.save(os.path.join(SAVE_DIR, "geometry.npy"), geom_arr)
        np.save(os.path.join(SAVE_DIR, "phi.npy"), phi_arr)
        np.save(os.path.join(SAVE_DIR, "rho_smooth.npy"), rho_arr)

        print("✅ Saved to", SAVE_DIR)
        print("params.npy shape:", params_arr.shape)
        print("geometry.npy shape:", geom_arr.shape)
        print("phi.npy shape:", phi_arr.shape)
        print("rho_smooth.npy shape:", rho_arr.shape)
        return params_arr, geom_arr, phi_arr, rho_arr
    else:
        print("❌ No valid samples generated.")
        return None, None, None, None

'''
if __name__ == "__main__":
    generate_dataset()
    params = np.load(os.path.join(SAVE_DIR, "params.npy"))
    param_names = [
    "b1_x", "b1_y", "b1_theta", "b1_L", "b1_t",
    "b2_x", "b2_y", "b2_theta", "b2_L", "b2_t",
    "b3_x", "b3_y", "b3_theta", "b3_L", "b3_t"
]

    # Plot histograms for key parameters
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(15):
        axes[i].hist(params[:, i], bins=20, alpha=0.7, color='blue')
        axes[i].set_title(param_names[i])
        axes[i].grid(True, linestyle='--', alpha=0.5)

    #plt.tight_layout()
    #plt.suptitle("Parameter Space Distribution (lhs Sampling)", y=1.02)
    #plt.savefig(os.path.join(SAVE_DIR, "param_space_lhs.png"), dpi=150, bbox_inches='tight')
    #plt.show()
    meta = np.load("dataset_cantilever_sym6_mmc/metadata.npy", allow_pickle=False)
    print(meta["area_fraction"])  # → array of floats
    print(meta["connected"])      # → array of bools
    print(meta["touches_load"]) 
    print(counter)
    for i in range(min(10, len(rho))):
        compliance, u = fem_physical_compliance(
        rho[i],
        save_vtk=True,
        vtk_filename=os.path.join(SAVE_DIR, f"design_{i:03d}.vtk")
    )
    print(f"Exported design {i} with compliance = {compliance:.6f}")
'''
if __name__ == "__main__":
    # Step 1: Generate dataset
    result = generate_dataset()
    if result[0] is None:
        print("No data generated.")
        exit()

    # Step 2: Load data for VTK export
    params = np.load(os.path.join(SAVE_DIR, "params.npy"))
    rho = np.load(os.path.join(SAVE_DIR, "rho_smooth.npy"))  # ✅ LOAD RHO HERE

    # Step 3: Export first 10 designs to VTK
    for i in range(min(10, len(rho))):
        compliance, u = fem_physical_compliance(
            rho[i],
            save_vtk=True,
            vtk_filename=os.path.join(SAVE_DIR, f"design_{i:03d}.vtk")
        )
        print(f"Exported design {i} with compliance = {compliance:.6f}")

    # Optional: Plot parameter distributions
    param_names = [
        "b1_x", "b1_y", "b1_theta", "b1_L", "b1_t",
        "b2_x", "b2_y", "b2_theta", "b2_L", "b2_t",
        "b3_x", "b3_y", "b3_theta", "b3_L", "b3_t"
    ]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(15):
        axes[i].hist(params[:, i], bins=20, alpha=0.7, color='blue')
        axes[i].set_title(param_names[i])
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "param_space.png"), dpi=150, bbox_inches='tight')
    plt.show()

