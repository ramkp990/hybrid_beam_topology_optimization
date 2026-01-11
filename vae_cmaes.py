# vae_cmaes.py
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
        sigma0=0.8,
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

if __name__ == "__main__":
    main()