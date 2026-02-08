# compute_compliance.py - Run ONCE to add compliance to metadata
import os
import numpy as np
from fem_code import fem_physical_compliance

DATASETS = [
    "dataset/dataset_cantilever_sym6_mmc",
    "dataset/dataset_cantilever_sym6_mmc1",
    "dataset/dataset_cantilever_sym6_mmc2",
    "dataset/dataset_cantilever_sym6_mmc3",
    "dataset/dataset_cantilever_sym6_mmc4",
    "dataset/dataset_cantilever_sym6_mmc5",
    "dataset/dataset_cantilever_sym6_mmc6",
    "dataset/dataset_cantilever_sym6_mmc7",
    "dataset/dataset_cantilever_sym6_mmc8",
    "dataset/dataset_cantilever_sym6_mmc9"
]

for d in DATASETS:
    print(f"Processing {d}...")
    
    # Load existing data
    rho = np.load(os.path.join(d, "rho_smooth.npy"))
    meta = np.load(os.path.join(d, "metadata.npy"), allow_pickle=True)
    
    # Compute compliance for ALL designs (one-time cost)
    compliance_vals = []
    for i in range(len(rho)):
        c, _ = fem_physical_compliance(rho[i])
        compliance_vals.append(c)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(rho)}: compliance = {c:.2f}")
    
    # Create NEW metadata with compliance field
    new_meta = np.array([
        (
            m["area_fraction"],
            m["connected"],
            m["main_component_area"],
            m["touches_left"],
            m["touches_load"],
            compliance_vals[i]  # ← ADD COMPLIANCE HERE
        )
        for i, m in enumerate(meta)
    ], dtype=[
        ('area_fraction', np.float32),
        ('connected', bool),
        ('main_component_area', np.int32),
        ('touches_left', bool),
        ('touches_load', bool),
        ('compliance', np.float32)  # ← NEW FIELD
    ])
    
    # Save updated metadata
    np.save(os.path.join(d, "metadata_with_compliance.npy"), new_meta)
    print(f"✅ Saved updated metadata to {d}/metadata_with_compliance.npy")