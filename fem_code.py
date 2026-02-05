# fem_code.py


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


# Constants (or import from constants.py)
N = 1000
GRID = 64
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

def is_feasible(rho_recon):
    """
    Check if a density field is feasible.
    - rho_recon: (64, 64) numpy array in [0,1]
    Returns: bool
    """
    # Convert to binary using same threshold as dataset generation
    geom = (rho_recon > 0.5).astype(np.uint8)
    
    # Volume constraint
    area = geom.sum()
    if area < MIN_AREA or area > MAX_AREA:
        return False

    # Connectivity constraint
    labeled, num = label(geom, structure=np.ones((3,3), dtype=np.int8))
    if num == 0:
        return False
    sizes = [(labeled == k).sum() for k in range(1, num + 1)]
    largest = max(sizes)
    if largest < LARGE_COMPONENT_RATIO * area:
        return False
    main_idx = 1 + sizes.index(largest)
    main_mask = (labeled == main_idx)

    # Support contact (left edge)
    if main_mask[:, 0].sum() == 0:
        return False

    # Load contact (right-center window)
    y_low = max(0, LOAD_POINT_Y - LOAD_POINT_WINDOW)
    y_high = min(GRID, LOAD_POINT_Y + LOAD_POINT_WINDOW + 1)
    if main_mask[y_low:y_high, LOAD_POINT_X].sum() == 0:
        return False

    return True

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



def params_to_rho(params):
    """
    Convert 15 beam parameters to 64x64 rho field.
    params: array of shape (15,) → [b1_x, b1_y, b1_theta, b1_L, b1_t, b2_x, ..., b3_t]
    Returns: rho (64, 64) float32 in [0,1]
    """
    # Split into 3 beams
    b1 = params[0:5]
    b2 = params[5:10]
    b3 = params[10:15]
    
    # Build phi
    phi_global, _ = build_phi_global([b1, b2, b3])
    
    # Convert to smooth rho
    rho = expit(SIGMOID_K * phi_global).astype(np.float32)
    return rho