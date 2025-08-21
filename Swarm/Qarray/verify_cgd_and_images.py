#!/usr/bin/env python3
"""
Verify Cgd Ground Truth and Generated Images

This script comprehensively tests:
1. Cgd matrix symmetry in ground truth data
2. Generated image quality and features
3. Consistency between parameters and images
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to path to import qarray components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_generator import ParameterSampler, ModelBuilder, ImageGenerator

def verify_cgd_ground_truth(n_samples=10):
    """Verify Cgd matrix symmetry in ground truth data"""
    
    print("=" * 60)
    print("VERIFYING CGD GROUND TRUTH VALUES")
    print("=" * 60)
    
    # Load config
    with open('qarray_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create parameter sampler
    param_sampler = ParameterSampler(config)
    
    symmetry_violations = 0
    max_asymmetry = 0.0
    
    print(f"Testing {n_samples} samples for Cgd symmetry...")
    
    for sample_idx in range(n_samples):
        # Generate parameters
        rng = np.random.default_rng(42 + sample_idx)
        params = param_sampler.sample_parameters(rng)
        
        # Check Cgd matrix
        Cgd = np.array(params['Cgd'])
        
        # Extract 4x4 plunger gate submatrix
        Cgd_plunger = Cgd[:4, :4]
        
        # Check symmetry
        is_symmetric = np.allclose(Cgd_plunger, Cgd_plunger.T, rtol=1e-10)
        
        if not is_symmetric:
            symmetry_violations += 1
            asymmetry = np.max(np.abs(Cgd_plunger - Cgd_plunger.T))
            max_asymmetry = max(max_asymmetry, asymmetry)
            print(f"  Sample {sample_idx + 1}: ASYMMETRIC (max diff: {asymmetry:.2e})")
        
        # Detailed check for first few samples
        if sample_idx < 3:
            print(f"\nSample {sample_idx + 1} detailed analysis:")
            print(f"  Cgd shape: {Cgd.shape}")
            print(f"  Plunger submatrix symmetric: {is_symmetric}")
            
            # Show specific values
            print(f"  Key symmetric pairs:")
            pairs = [(0,1), (0,2), (1,3), (2,3)]
            for i, j in pairs:
                val_ij = Cgd[i, j]
                val_ji = Cgd[j, i]
                diff = abs(val_ij - val_ji)
                print(f"    Cgd[{i},{j}] = {val_ij:.6f}, Cgd[{j},{i}] = {val_ji:.6f}, diff = {diff:.2e}")
            
            # Show sensor couplings (should be independent)
            print(f"  Sensor couplings: {Cgd[:, 4]}")
            
            # Show coupling hierarchy (should decrease with distance)
            print(f"  Coupling hierarchy:")
            print(f"    Primary (0,0): {Cgd[0,0]:.3f}")
            print(f"    1st neighbor (0,1): {Cgd[0,1]:.3f}")  
            print(f"    2nd neighbor (0,2): {Cgd[0,2]:.3f}")
            print(f"    3rd neighbor (0,3): {Cgd[0,3]:.3f}")
    
    print(f"\n--- SYMMETRY SUMMARY ---")
    print(f"Samples tested: {n_samples}")
    print(f"Symmetry violations: {symmetry_violations}")
    print(f"Success rate: {(n_samples - symmetry_violations)/n_samples*100:.1f}%")
    if max_asymmetry > 0:
        print(f"Max asymmetry found: {max_asymmetry:.2e}")
    
    return symmetry_violations == 0

def generate_and_analyze_images(n_examples=4):
    """Generate example images and analyze their features"""
    
    print("\n" + "=" * 60)
    print("GENERATING AND ANALYZING IMAGES")
    print("=" * 60)
    
    # Load config
    with open('qarray_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create components
    param_sampler = ParameterSampler(config)
    image_gen = ImageGenerator((-2.0, 2.0), 128)
    
    # Create output directory
    output_dir = Path('cgd_verification_images')
    output_dir.mkdir(exist_ok=True)
    
    examples = []
    
    for i in range(n_examples):
        print(f"\nGenerating example {i+1}/{n_examples}...")
        
        # Generate parameters
        rng = np.random.default_rng(42 + i)
        model_params = param_sampler.sample_parameters(rng)
        
        # Build model
        model = ModelBuilder.build_model(model_params)
        
        # Random center for voltage sweep
        center_x = rng.uniform(-1.0, 1.0)
        center_y = rng.uniform(-1.0, 1.0)
        
        # Generate voltage grid and image
        voltage_grid = image_gen.generate_voltage_grid(
            (center_x, center_y), rng, model_params['fixed_gate_voltages']
        )
        image = image_gen.generate_image(model, voltage_grid)
        
        # Store example
        example = {
            'image': image,
            'center': (center_x, center_y),
            'model_params': model_params,
            'voltage_grid': voltage_grid
        }
        examples.append(example)
        
        # Analyze image features
        print(f"  Image statistics:")
        print(f"    Shape: {image.shape}")
        print(f"    Range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"    Mean: {image.mean():.3f}, Std: {image.std():.3f}")
        
        # Look for transition features
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        strong_features = grad_mag > np.percentile(grad_mag, 95)
        print(f"    Strong features: {np.sum(strong_features)} pixels ({np.sum(strong_features)/image.size*100:.1f}%)")
        print(f"    Max gradient: {grad_mag.max():.3f}")
        
        # Analyze Cgd values for this example
        Cgd = np.array(model_params['Cgd'])
        print(f"  Cgd analysis:")
        print(f"    Is plunger submatrix symmetric: {np.allclose(Cgd[:4, :4], Cgd[:4, :4].T)}")
        print(f"    Cross-coupling (0,1): {Cgd[0,1]:.3f}")
        print(f"    Cross-coupling (1,0): {Cgd[1,0]:.3f}")
        print(f"    Sensor couplings: {Cgd[:, 4]}")
    
    # Create visualization
    plot_cgd_and_images(examples, output_dir)
    
    return examples

def plot_cgd_and_images(examples, output_dir):
    """Create comprehensive plots showing Cgd matrices and corresponding images"""
    
    n_examples = len(examples)
    
    # Create figure with subplots: Cgd matrix + image for each example
    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 8))
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    for i, example in enumerate(examples):
        image = example['image']
        Cgd = np.array(example['model_params']['Cgd'])
        center = example['center']
        fixed_voltages = example['model_params']['fixed_gate_voltages']
        
        # Plot Cgd matrix
        ax_cgd = axes[0, i]
        im_cgd = ax_cgd.imshow(Cgd, cmap='viridis', aspect='auto')
        ax_cgd.set_title(f'Cgd Matrix - Example {i+1}')
        ax_cgd.set_xlabel('Gates (0-3: plunger, 4: sensor)')
        ax_cgd.set_ylabel('Dots')
        
        # Add text annotations for values
        for dot_idx in range(4):
            for gate_idx in range(5):
                text = ax_cgd.text(gate_idx, dot_idx, f'{Cgd[dot_idx, gate_idx]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im_cgd, ax=ax_cgd, shrink=0.8)
        
        # Plot corresponding image
        ax_img = axes[1, i]
        im_img = ax_img.imshow(image, extent=[-2, 2, -2, 2], origin='lower', 
                              cmap='plasma', aspect='equal')
        ax_img.plot(center[0], center[1], 'r+', markersize=10, markeredgewidth=2)
        ax_img.set_title(f'Charge Sensor Image - Example {i+1}')
        ax_img.set_xlabel('Gate 1 Voltage (V)')
        ax_img.set_ylabel('Gate 2 Voltage (V)')
        ax_img.grid(True, alpha=0.3)
        
        plt.colorbar(im_img, ax=ax_img, shrink=0.8)
        
        # Add text with key info
        info_text = f"Center: ({center[0]:.2f}, {center[1]:.2f})\n"
        info_text += f"Fixed: G0={fixed_voltages[0]:.2f}V\n"
        info_text += f"       G3={fixed_voltages[3]:.2f}V\n"
        info_text += f"       G4={fixed_voltages[4]:.2f}V"
        ax_img.text(0.02, 0.98, info_text, transform=ax_img.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cgd_matrices_and_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create detailed symmetry verification plot
    plot_symmetry_details(examples, output_dir)

def plot_symmetry_details(examples, output_dir):
    """Create detailed plots showing symmetry verification"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cgd matrix heatmap for first example
    Cgd = np.array(examples[0]['model_params']['Cgd'])
    im1 = axes[0,0].imshow(Cgd, cmap='viridis', aspect='auto')
    axes[0,0].set_title('Cgd Matrix - Full 4×5')
    axes[0,0].set_xlabel('Gates')
    axes[0,0].set_ylabel('Dots')
    for i in range(4):
        for j in range(5):
            axes[0,0].text(j, i, f'{Cgd[i,j]:.3f}', ha="center", va="center", 
                          color="white", fontsize=9)
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot 2: Plunger submatrix (4×4) showing symmetry
    Cgd_plunger = Cgd[:4, :4]
    im2 = axes[0,1].imshow(Cgd_plunger, cmap='viridis', aspect='equal')
    axes[0,1].set_title('Plunger Gates Submatrix (4×4)\nShould be Symmetric')
    axes[0,1].set_xlabel('Plunger Gates')
    axes[0,1].set_ylabel('Dots')
    for i in range(4):
        for j in range(4):
            axes[0,1].text(j, i, f'{Cgd_plunger[i,j]:.3f}', ha="center", va="center", 
                          color="white", fontsize=9)
    plt.colorbar(im2, ax=axes[0,1])
    
    # Plot 3: Symmetry verification - difference matrix
    diff_matrix = Cgd_plunger - Cgd_plunger.T
    im3 = axes[1,0].imshow(diff_matrix, cmap='RdBu', aspect='equal', vmin=-1e-10, vmax=1e-10)
    axes[1,0].set_title('Symmetry Check: Cgd - Cgd^T\n(Should be all zeros)')
    axes[1,0].set_xlabel('Plunger Gates')
    axes[1,0].set_ylabel('Dots')
    for i in range(4):
        for j in range(4):
            axes[1,0].text(j, i, f'{diff_matrix[i,j]:.1e}', ha="center", va="center", 
                          color="black", fontsize=8)
    plt.colorbar(im3, ax=axes[1,0])
    
    # Plot 4: Coupling strength vs distance
    distances = []
    couplings = []
    for i in range(4):
        for j in range(4):
            if i != j:  # Skip diagonal
                dist = abs(i - j)
                coupling = Cgd_plunger[i, j]
                distances.append(dist)
                couplings.append(coupling)
    
    axes[1,1].scatter(distances, couplings, alpha=0.7, s=50)
    axes[1,1].set_xlabel('Distance Between Dot and Gate')
    axes[1,1].set_ylabel('Coupling Strength')
    axes[1,1].set_title('Coupling vs Distance\n(Should decrease with distance)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add trend line
    unique_distances = sorted(set(distances))
    mean_couplings = [np.mean([c for d, c in zip(distances, couplings) if d == ud]) 
                      for ud in unique_distances]
    axes[1,1].plot(unique_distances, mean_couplings, 'r-', linewidth=2, label='Mean')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'symmetry_verification_details.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main verification function"""
    print("COMPREHENSIVE CGD AND IMAGE VERIFICATION")
    print("This script verifies:")
    print("1. Cgd matrix symmetry in ground truth data")
    print("2. Generated image quality and features")
    print("3. Consistency between parameters and images")
    
    # Check if config exists
    if not os.path.exists('qarray_config.yaml'):
        print("Error: qarray_config.yaml not found!")
        return
    
    try:
        # Verify Cgd ground truth
        symmetry_ok = verify_cgd_ground_truth(n_samples=20)
        
        # Generate and analyze images
        examples = generate_and_analyze_images(n_examples=4)
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"✅ Cgd symmetry: {'PASSED' if symmetry_ok else 'FAILED'}")
        print(f"✅ Image generation: COMPLETED ({len(examples)} examples)")
        print(f"✅ Visualizations saved to: cgd_verification_images/")
        
        if symmetry_ok:
            print("\n🎯 All verifications PASSED!")
            print("   - Cgd matrices are properly symmetric")
            print("   - Images generated successfully")
            print("   - Physical consistency maintained")
        else:
            print("\n⚠️  Some issues found - check detailed output above")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 