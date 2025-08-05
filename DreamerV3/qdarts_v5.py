import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add qdarts to path
sys.path.append(os.path.dirname(__file__))

from qdarts_config_loader_v5 import load_qdarts_config, print_config_summary
from qdarts.experiment_with_barriers import Experiment


def main():
    """
    """
    
    print("Loading configuration...")
    config = load_qdarts_config('qdarts_config_v5.yaml')
    print_config_summary(config)
    
    capacitance_config = config['simulator']['capacitance']
    tunneling_config = config['simulator']['tunneling']
    sensor_config = config['simulator']['sensor']
    barrier_config = config['simulator']['barrier']
    
    print("\nInitializing experiment with barriers...")
    experiment = Experiment(
        capacitance_config=capacitance_config,
        tunneling_config=tunneling_config,
        sensor_config=sensor_config,
        barrier_config=barrier_config,
        print_logs=True
    )
    
    print("\nSetting initial barrier voltages...")
    initial_barrier_voltages = barrier_config['default_barrier_voltages']
    experiment.update_tunnel_couplings(initial_barrier_voltages)
    
    measurement = config['measurement']
    x_voltages = np.linspace(
        measurement['voltage_range']['min'][0], 
        measurement['voltage_range']['max'][0], 
        measurement['resolution']
    )
    y_voltages = np.linspace(
        measurement['voltage_range']['min'][1], 
        measurement['voltage_range']['max'][1], 
        measurement['resolution']
    )
    
    # Convert sweep matrix to plane axes (gate indices)
    sweep_matrix = measurement['sweep_matrix']
    plane_axes = []
    for i in range(2):
        for j, row in enumerate(sweep_matrix):
            if abs(row[i]) > 1e-10:
                plane_axes.append(j)
                break
    
    print(f"Plane axes: {plane_axes} (gates {plane_axes[0]} and {plane_axes[1]})")
    
    # Generate CSD with initial barrier voltages
    print("\nGenerating CSD with initial barrier voltages...")
    start_time = time.time()
    
    xout, yout, CSD_data, polytopes, sensor_values, v_offset = experiment.generate_CSD(
        x_voltages=x_voltages,
        y_voltages=y_voltages,
        plane_axes=plane_axes,
        target_state=config['device']['target_state'],
        use_sensor_signal=True,  # Get sensor response
        compute_polytopes=False
    )
    
    print(f"CSD generation completed in {time.time() - start_time:.2f} seconds")
    
    # Demonstrate barrier voltage control
    print("\nDemonstrating barrier voltage control...")
    
    # Update barrier voltages
    new_barrier_voltages = {
        'barrier_2': 0.5,   # Increase reservoir-dot 0 coupling
        'barrier_3': -0.2,  # Decrease dot-dot coupling
        'barrier_4': 0.3    # Increase dot 1-reservoir coupling
    }
    
    experiment.update_tunnel_couplings(new_barrier_voltages)
    
    # Generate CSD with new barrier voltages
    print("Generating CSD with updated barrier voltages...")
    start_time = time.time()
    
    xout2, yout2, CSD_data2, polytopes2, sensor_values2, v_offset2 = experiment.generate_CSD(
        x_voltages=x_voltages,
        y_voltages=y_voltages,
        plane_axes=plane_axes,
        target_state=config['device']['target_state'],
        use_sensor_signal=True,
        compute_polytopes=False
    )
    
    print(f"CSD generation completed in {time.time() - start_time:.2f} seconds")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    
    # Plot 1: Initial barrier voltages
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.pcolormesh(xout, yout, sensor_values[:, :, 0].T)
    plt.xlabel('Gate 0 Voltage (V)')
    plt.ylabel('Gate 1 Voltage (V)')
    plt.title(f'Sensor Response\nInitial Barriers:')
    plt.colorbar(label='Sensor Conductance')
    
    # Plot 2: Updated barrier voltages
    plt.subplot(1, 3, 2)
    plt.pcolormesh(xout2, yout2, sensor_values2[:, :, 0].T)
    plt.xlabel('Gate 0 Voltage (V)')
    plt.ylabel('Gate 1 Voltage (V)')
    plt.title(f'Sensor Response\nUpdated Barriers:')
    plt.colorbar(label='Sensor Conductance')
    
    # Plot 3: Difference
    plt.subplot(1, 3, 3)
    difference = sensor_values2[:, :, 0] - sensor_values[:, :, 0]
    plt.pcolormesh(xout, yout, difference.T)
    plt.xlabel('Gate 0 Voltage (V)')
    plt.ylabel('Gate 1 Voltage (V)')
    plt.title('Difference (Updated - Initial)')
    plt.colorbar(label='Δ Sensor Conductance')
    
    plt.tight_layout()
    plt.savefig('qdarts_v5_barrier_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nSimulation Statistics:")
    print(f"  Initial sensor data shape: {sensor_values.shape}")
    print(f"  Initial sensor range: [{sensor_values[:, :, 0].min():.4f}, {sensor_values[:, :, 0].max():.4f}]")
    print(f"  Updated sensor data shape: {sensor_values2.shape}")
    print(f"  Updated sensor range: [{sensor_values2[:, :, 0].min():.4f}, {sensor_values2[:, :, 0].max():.4f}]")
    print(f"  Maximum difference: {np.abs(difference).max():.4f}")
    
    
    print("  - qdarts_v5_barrier_comparison.png")


if __name__ == "__main__":
    main()
