import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
sys.path.insert(0, str(src_dir))

from swarm.environment.qarray_base_class import QarrayBaseClass
from swarm.capacitance_model.CapacitancePrediction import create_model

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model()
    model = model.to(device)
    weights_path = "./outputs/best_model.pth"

    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights['model_state_dict'])

    qarray = QarrayBaseClass(
        num_dots=4,
        obs_voltage_min=-1.0,
        obs_voltage_max=1.0,
        obs_image_size=128,
    )

    gt_voltages = qarray.calculate_ground_truth()

    rng = np.random.default_rng(42)  # Fixed seed for testing
    voltage_offset = rng.uniform(
        -0.1, 
        0.1, 
        size=len(gt_voltages)
    )
    gate_voltages = gt_voltages + voltage_offset

    barrier_voltages = [0.0] * 3  # 4 dots means 3 barriers
            
    # Generate observation
    obs = qarray._get_obs(gate_voltages, barrier_voltages)

    cgd_matrix = qarray.model.Cgd.copy()

    image = obs['image'][:, :, 1:2]

    # save image
    plt.imshow(image, cmap='viridis')
    plt.axis('off')
    plt.savefig("observation_image.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    means, logvars = model(image)
    means = means.cpu().detach().numpy().squeeze()
    logvars = logvars.cpu().detach().numpy().squeeze()

    print("Predicted means:", means)
    print("Predicted logvars:", logvars)

    

if __name__ == '__main__':
    main()