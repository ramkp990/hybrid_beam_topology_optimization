# preprocess_half_dataset_corrected.py
import numpy as np
import os

# Load your existing datasets
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

half_rho_list = []
metadata_list = []

for d in DATASETS:
    rho_full = np.load(os.path.join(d, "rho_smooth.npy"))
    metadata = np.load(os.path.join(d, "metadata.npy"), allow_pickle=True)
    
    # ✅ CORRECT: Extract TOP HALF (Y=32-63) for proper X-direction beams
    # Your data generation creates beams pointing left-to-right, so use top half
    rho_half = rho_full[:, :32, :]
    # Shape: [N, 32, 64] - rows 32-63 (top half)
    
    half_rho_list.append(rho_half)
    metadata_list.append(metadata)

# Save half-domain dataset
half_rho_combined = np.concatenate(half_rho_list, axis=0)
metadata_combined = np.concatenate(metadata_list, axis=0)

np.save("dataset_half/half_rho_smooth.npy", half_rho_combined)
np.save("dataset_half/half_metadata.npy", metadata_combined)

print(f"✅ Created half-domain dataset: {half_rho_combined.shape}")
print("💡 Key insight: Using TOP HALF (Y=32-63) instead of bottom half")
print("   This preserves the X-direction beam orientation from your data generation")