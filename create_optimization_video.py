import pyvista as pv
import imageio
import glob
import numpy as np

directory = "results/vti/"
files = sorted(
    glob.glob(directory + "gen_*.vti"),
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

frames = []

for f in files:
    grid = pv.read(f)
    
    # Get density array
    arr = grid["Density"]
    
    # Dimensions
    nx, ny, _ = grid.dimensions
    
    img = arr.reshape(ny, nx)
    
    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    
    frames.append(img)

# Save video
with imageio.get_writer("evolution.mp4", fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)

print("✅ Video saved as evolution.mp4")
