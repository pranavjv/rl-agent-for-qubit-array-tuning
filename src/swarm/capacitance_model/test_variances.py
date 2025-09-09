
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt

from pathlib import Path
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
sys.path.insert(0, str(src_dir))

from swarm.capacitance_model.dataloader import PercentileNormalize, get_channel_targets
from swarm.capacitance_model.CapacitancePrediction import CapacitancePredictionModel, create_model
from swarm.capacitance_model.dataloader import create_data_loaders, get_transforms


def load_trained_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = create_model()
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model


def main():
    # Configuration
    root_data_dir = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/'
    data_dirs = ['dataset', '4dot_dataset']
    model_path = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/weights/best_model.pth'
    batch_size = 64
    val_split = 0.2
    num_workers = 4
    load_to_memory = False
    num_samples = 1000
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data paths
    data_dirs_full = [os.path.join(root_data_dir, dir_) for dir_ in data_dirs]
    print(f"Loading data from: {data_dirs_full}")
    
    # Create data loaders
    transform = get_transforms(normalize=True)
    train_loader, val_loader = create_data_loaders(
        data_dirs=data_dirs_full,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        load_to_memory=load_to_memory,
        transform=transform
    )
    
    # Load trained model
    model = load_trained_model(model_path, device)
    
    # Collect validation images and run inference
    all_predictions = []
    
    loader = iter(val_loader)
    collected_samples = 0
    
    print(f"Collecting {num_samples} validation samples...")
    
    with torch.no_grad():
        while collected_samples < num_samples:
            try:
                images, targets = next(loader)
                batch_size_actual = images.size(0)
                
                # Only process as many samples as we need
                remaining = num_samples - collected_samples
                samples_to_process = min(batch_size_actual, remaining)
                
                # Process this batch
                images_batch = images[:samples_to_process].to(device)
                values, log_vars = model(images_batch)
                
                # Convert to numpy and add to predictions list
                predicted_values = values.cpu().detach().numpy()
                predicted_log_vars = log_vars.cpu().detach().numpy()
                predicted_uncertainties = np.exp(0.5 * predicted_log_vars)
                
                # Add each sample to the list
                for i in range(samples_to_process):
                    prediction = {
                        'values': predicted_values[i],
                        'log_vars': predicted_log_vars[i], 
                        'uncertainties': predicted_uncertainties[i]
                    }
                    all_predictions.append(prediction)
                
                collected_samples += samples_to_process
                
                if collected_samples % 100 == 0:
                    print(f"Processed {collected_samples}/{num_samples} samples")
                    
            except StopIteration:
                print(f"Reached end of validation set with {collected_samples} samples")
                break
    
    print(f"Completed! Collected {len(all_predictions)} predictions")
    
    # Print some statistics
    all_values = np.array([p['values'] for p in all_predictions])
    all_uncertainties = np.array([p['uncertainties'] for p in all_predictions])
    
    print(f"\nPrediction Statistics:")
    print(f"Values - Mean: {all_values.mean(axis=0)}")
    print(f"Values - Std: {all_values.std(axis=0)}")
    print(f"Uncertainties - Mean: {all_uncertainties.mean(axis=0)}")
    print(f"Uncertainties - Std: {all_uncertainties.std(axis=0)}")


if __name__ == "__main__":
    main()