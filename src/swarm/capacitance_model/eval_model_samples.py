#!/usr/bin/env python3
"""
Simple evaluation script to test trained capacitance model on random validation samples.

This script:
1. Loads the same data as used in training 
2. Loads the trained model
3. Extracts 10 random samples from validation set
4. Runs inference and prints model outputs
"""

import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Local imports
from CapacitancePrediction import create_model
from dataloader import create_data_loaders, get_transforms


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
    
    print(f"Model loaded successfully! Epoch: {checkpoint.get('epoch', 'Unknown')}")
    return model


def evaluate_random_samples(model, val_loader, device, num_samples=10):
    """Evaluate model on random validation samples."""
    print(f"\nEvaluating {num_samples} random validation samples...")
    
    # Collect all validation data
    all_images = []
    all_targets = []
    
    # total_samples = len(val_loader)
    # random_sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    
    # with torch.no_grad():
    #     for idx in random_sample_indices:
    #         images, targets = val_loader.dataset[idx]
    #         all_images.append(images.cpu())
    #         all_targets.append(targets.cpu())

    loader = iter(val_loader)
    batch_size = val_loader.batch_size
    num_batches = num_samples // batch_size + 1 if num_samples % batch_size != 0 else num_samples // batch_size

    for _ in range(num_batches):
        images, targets = next(loader)
        all_images.append(images.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all data
    val_images = torch.cat(all_images, dim=0)[:num_samples, ...]
    val_targets = torch.cat(all_targets, dim=0)[:num_samples, ...]

    val_images = torch.split(val_images, 1, dim=0)
    val_targets = torch.split(val_targets, 1, dim=0)
    
    print(f"Total validation samples available: {len(val_images)}")
    
    # Select random samples
    # total_samples = len(all_images)
    # sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    
    results = []
    
    for idx, (image, target) in enumerate(zip(val_images, val_targets)):
        image = image.to(device)

        values, log_vars = model(image)
        
        # Convert to numpy for easier handling
        predicted_values = values.cpu().squeeze().detach().numpy()  # Remove batch dimension
        predicted_log_vars = log_vars.cpu().squeeze().detach().numpy()
        predicted_uncertainties = np.exp(0.5 * predicted_log_vars)  # Convert to standard deviations
        target_values = target.squeeze().numpy()
        
        # Calculate errors
        errors = np.abs(predicted_values - target_values)
        percentage_errors = 100 * errors / (np.abs(target_values) + 1e-8)
        
        # Store results
        result = {
            'sample_idx': int(idx),
            'target_values': target_values,
            'predicted_values': predicted_values,
            'predicted_uncertainties': predicted_uncertainties,
            'absolute_errors': errors,
            'percentage_errors': percentage_errors,
            #'image': all_images[idx].squeeze().numpy()  # For visualization
        }
        results.append(result)
        
        # Print detailed results
        print(f"\n=== Sample {idx+1}/10 ===")
        print(f"Target CGD values:     [{target_values[0]:.6f}, {target_values[1]:.6f}, {target_values[2]:.6f}]")
        print(f"Predicted CGD values:  [{predicted_values[0]:.6f}, {predicted_values[1]:.6f}, {predicted_values[2]:.6f}]")
        print(f"Predicted uncertainties: [{predicted_uncertainties[0]:.6f}, {predicted_uncertainties[1]:.6f}, {predicted_uncertainties[2]:.6f}]")
        print(f"Absolute errors:       [{errors[0]:.6f}, {errors[1]:.6f}, {errors[2]:.6f}]")
        print(f"Percentage errors:     [{percentage_errors[0]:.2f}%, {percentage_errors[1]:.2f}%, {percentage_errors[2]:.2f}%]")
    
    return results


def create_visualization(results, save_path="model_evaluation_results.png"):
    """Create visualization of results."""
    num_samples = len(results)
    fig, axes = plt.subplots(3, min(5, num_samples), figsize=(15, 9))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    elif num_samples < 5:
        axes = axes[:, :num_samples]
    
    for i, result in enumerate(results[:5]):  # Show first 5 samples
        col = i
        
        # Plot the input image (charge stability diagram)
        im = axes[0, col].imshow(result['image'], cmap='viridis', aspect='equal')
        axes[0, col].set_title(f'Sample {result["sample_idx"]}\n(Input CSD)', fontsize=10)
        axes[0, col].set_xlabel('Gate Voltage')
        axes[0, col].set_ylabel('Gate Voltage')
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)
        
        # Plot target vs predicted values
        target_names = ['Cgd[m,r]', 'Cgd[r,r+1]', 'Cgd[l,m]']
        x_pos = np.arange(3)
        
        target_vals = result['target_values']
        pred_vals = result['predicted_values']
        uncertainties = result['predicted_uncertainties']
        
        axes[1, col].bar(x_pos - 0.2, target_vals, 0.4, label='Target', alpha=0.7, color='blue')
        axes[1, col].bar(x_pos + 0.2, pred_vals, 0.4, label='Predicted', alpha=0.7, color='red')
        axes[1, col].errorbar(x_pos + 0.2, pred_vals, yerr=uncertainties, 
                             fmt='none', color='black', capsize=3, alpha=0.8)
        
        axes[1, col].set_xlabel('CGD Component')
        axes[1, col].set_ylabel('Capacitance Value')
        axes[1, col].set_title('Target vs Predicted', fontsize=10)
        axes[1, col].set_xticks(x_pos)
        axes[1, col].set_xticklabels(target_names, rotation=45, fontsize=8)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        
        # Plot percentage errors
        colors = ['green' if pe < 10 else 'orange' if pe < 25 else 'red' for pe in result['percentage_errors']]
        bars = axes[2, col].bar(x_pos, result['percentage_errors'], color=colors, alpha=0.7)
        axes[2, col].set_xlabel('CGD Component')
        axes[2, col].set_ylabel('Percentage Error (%)')
        axes[2, col].set_title('Prediction Errors', fontsize=10)
        axes[2, col].set_xticks(x_pos)
        axes[2, col].set_xticklabels(target_names, rotation=45, fontsize=8)
        axes[2, col].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pe in zip(bars, result['percentage_errors']):
            height = bar.get_height()
            axes[2, col].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{pe:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots if fewer than 5 samples
    if num_samples < 5:
        for col in range(num_samples, 5):
            for row in range(3):
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def print_summary_statistics(results):
    """Print summary statistics across all samples."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Collect all errors
    all_abs_errors = np.array([r['absolute_errors'] for r in results])
    all_perc_errors = np.array([r['percentage_errors'] for r in results])
    all_uncertainties = np.array([r['predicted_uncertainties'] for r in results])
    
    target_names = ['Cgd[middle,right]', 'Cgd[right,right+1]', 'Cgd[left,middle]']
    
    print(f"Statistics across {len(results)} samples:\n")
    
    # Overall statistics
    print("OVERALL PERFORMANCE:")
    print(f"  Mean Absolute Error:    {all_abs_errors.mean():.6f} ± {all_abs_errors.std():.6f}")
    print(f"  Mean Percentage Error:  {all_perc_errors.mean():.2f}% ± {all_perc_errors.std():.2f}%")
    print(f"  Mean Uncertainty:       {all_uncertainties.mean():.6f} ± {all_uncertainties.std():.6f}")
    
    # Per-component statistics
    print("\nPER-COMPONENT PERFORMANCE:")
    for i, name in enumerate(target_names):
        print(f"\n  {name}:")
        print(f"    Absolute Error:  {all_abs_errors[:, i].mean():.6f} ± {all_abs_errors[:, i].std():.6f}")
        print(f"    Percentage Error: {all_perc_errors[:, i].mean():.2f}% ± {all_perc_errors[:, i].std():.2f}%")
        print(f"    Uncertainty:     {all_uncertainties[:, i].mean():.6f} ± {all_uncertainties[:, i].std():.6f}")
    
    # Error distribution
    print("\nERROR DISTRIBUTION:")
    print(f"  Samples with <5% error:   {np.sum(all_perc_errors < 5)} / {all_perc_errors.size} ({100*np.sum(all_perc_errors < 5)/all_perc_errors.size:.1f}%)")
    print(f"  Samples with <10% error:  {np.sum(all_perc_errors < 10)} / {all_perc_errors.size} ({100*np.sum(all_perc_errors < 10)/all_perc_errors.size:.1f}%)")
    print(f"  Samples with <25% error:  {np.sum(all_perc_errors < 25)} / {all_perc_errors.size} ({100*np.sum(all_perc_errors < 25)/all_perc_errors.size:.1f}%)")


def main():
    # Configuration (matching training script)
    root_data_dir = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/'
    data_dirs = ['dataset', '4dot_dataset']
    model_path = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/weights/best_model.pth'
    batch_size = 64
    val_split = 0.2
    num_workers = 4
    load_to_memory = False  # Set to True if you have enough RAM
    num_samples = 10
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data paths
    data_dirs_full = [os.path.join(root_data_dir, dir_) for dir_ in data_dirs]
    print(f"Loading data from: {data_dirs_full}")
    
    # Create data loaders (same as in training)
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
    
    # Evaluate random samples
    results = evaluate_random_samples(model, val_loader, device, num_samples)
    
    # Create visualization
    create_visualization(results)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    print(f"\nEvaluation completed! Processed {len(results)} samples.")


if __name__ == "__main__":
    main()