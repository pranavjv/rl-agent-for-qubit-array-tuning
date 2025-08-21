import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import CapacitanceDataset, create_data_loaders
import os

def save_dataset_samples(data_dir: str, num_samples: int = 12, output_dir: str = "plots"):
    """
    Save sample images from the dataset with their corresponding ground truth capacitances.
    
    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to plot
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = CapacitanceDataset(data_dir, load_to_memory=False, validate_data=True)
    
    # Calculate grid size
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"Processing {num_samples} samples from dataset...")
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        # Get sample
        image, targets = dataset[i]
        
        # Convert to numpy for plotting
        img_np = image.squeeze().numpy()  # Remove channel dimension
        targets_np = targets.numpy()
        
        # Calculate subplot position
        row = i // cols
        col = i % cols
        
        # Plot image
        im = axes[row, col].imshow(img_np, cmap='viridis', aspect='equal')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        # Set title with capacitance values
        title = f"Sample {i}\n"
        title += f"Cgd[1,2]: {targets_np[0]:.3f}\n"
        title += f"Cgd[1,3]: {targets_np[1]:.3f}\n"
        title += f"Cgd[0,2]: {targets_np[2]:.3f}"
        
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].set_xlabel("X Position")
        axes[row, col].set_ylabel("Y Position")
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, "dataset_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Dataset samples plot saved to: {save_path}")
    plt.close()

def save_target_distribution_analysis(data_dir: str, num_samples: int = 1000, output_dir: str = "plots"):
    """
    Analyze and save the distribution of target values in the dataset.
    
    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to analyze (set to None for all)
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CapacitanceDataset(data_dir, load_to_memory=False)
    
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    print(f"Analyzing target distributions for {num_samples} samples...")
    
    # Collect targets
    targets_list = []
    for i in range(num_samples):
        _, targets = dataset[i]
        targets_list.append(targets.numpy())
    
    targets_array = np.array(targets_list)  # Shape: (num_samples, 3)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    target_names = ['Cgd[1,2]', 'Cgd[1,3]', 'Cgd[0,2]']
    
    for i, name in enumerate(target_names):
        values = targets_array[:, i]
        
        axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{name} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, "target_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Target distributions plot saved to: {save_path}")
    plt.close()
    
    # Print and save statistics
    print("\nTarget Statistics:")
    print("=" * 50)
    stats_text = "Target Statistics:\n" + "=" * 50 + "\n"
    for i, name in enumerate(target_names):
        values = targets_array[:, i]
        stats = f"{name}:\n  Mean: {np.mean(values):.6f}\n  Std:  {np.std(values):.6f}\n  Min:  {np.min(values):.6f}\n  Max:  {np.max(values):.6f}\n"
        print(stats)
        stats_text += stats + "\n"
    
    # Save statistics to file
    stats_path = os.path.join(output_dir, "target_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    print(f"Statistics saved to: {stats_path}")

def print_dataset_info(data_dir: str):
    """
    Print basic dataset information.
    
    Args:
        data_dir: Path to dataset directory
    """
    print("Dataset Information")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=8,
        val_split=0.2,
        num_workers=2,
        load_to_memory=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test one batch from each loader
    print("\nTrain batch example:")
    for batch_imgs, batch_targets in train_loader:
        print(f"  Images shape: {batch_imgs.shape}")
        print(f"  Targets shape: {batch_targets.shape}")
        print(f"  Image range: [{batch_imgs.min():.3f}, {batch_imgs.max():.3f}]")
        print(f"  Target ranges:")
        for i, name in enumerate(['Cgd[1,2]', 'Cgd[1,3]', 'Cgd[0,2]']):
            values = batch_targets[:, i]
            print(f"    {name}: [{values.min():.4f}, {values.max():.4f}]")
        break
    
    print("\nValidation batch example:")
    for batch_imgs, batch_targets in val_loader:
        print(f"  Images shape: {batch_imgs.shape}")
        print(f"  Targets shape: {batch_targets.shape}")
        print(f"  Image range: [{batch_imgs.min():.3f}, {batch_imgs.max():.3f}]")
        print(f"  Target ranges:")
        for i, name in enumerate(['Cgd[1,2]', 'Cgd[1,3]', 'Cgd[0,2]']):
            values = batch_targets[:, i]
            print(f"    {name}: [{values.min():.4f}, {values.max():.4f}]")
        break

def main():
    """Main function to run all tests."""
    # Set data directory
    data_dir = "/home/rahul/Summer2025/rl-agent-for-qubit-array-tuning/Swarm/Qarray/dataset_v1"
    output_dir = "dataset_analysis"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please update the data_dir path in the script.")
        return
    
    print("Dataset Analysis (Non-GUI Version)")
    print("=" * 50)
    
    try:
        # Test 1: Save sample images with capacitances
        print("1. Saving sample images...")
        save_dataset_samples(data_dir, num_samples=12, output_dir=output_dir)
        
        print("\n" + "=" * 50)
        
        # Test 2: Analyze and save target distributions
        print("2. Analyzing and saving target distributions...")
        save_target_distribution_analysis(data_dir, num_samples=2000, output_dir=output_dir)
        
        print("\n" + "=" * 50)
        
        # Test 3: Print dataset information
        print("3. Dataset information...")
        print_dataset_info(data_dir)
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print(f"Check the '{output_dir}' directory for saved plots and statistics.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 