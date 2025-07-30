import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add qdarts to path
sys.path.append(os.path.dirname(__file__))

from qdarts_config_loader_v4 import load_qdarts_config, print_config_summary
from qdarts.experiment import Experiment


def _calculate_barrier_tunnel_couplings(base_couplings, barrier_config, barrier_voltages):
    """
    Calculate tunnel couplings based on barrier voltages.
    
    This function implements the barrier voltage to tunnel coupling mapping:
    - Barrier 2: reservoir <-> dot 0 coupling
    - Barrier 3: dot 0 <-> dot 1 coupling  
    - Barrier 4: dot 1 <-> reservoir coupling
    
    Args:
        base_couplings: Base tunnel coupling matrix
        barrier_config: Barrier configuration from config
        barrier_voltages: Dictionary of barrier voltages
        
    Returns:
        Updated tunnel coupling matrix
    """
    if base_couplings is None:
        return None
    
    # Start with base tunnel couplings
    new_couplings = base_couplings.copy()
    barrier_mappings = barrier_config['barrier_mappings']
    
    # Apply barrier voltage effects
    for barrier_name, mapping in barrier_mappings.items():
        if barrier_name not in barrier_voltages:
            continue
            
        barrier_voltage = barrier_voltages[barrier_name]
        coupling_type = mapping['coupling_type']
        voltage_scale = mapping.get('voltage_scale', 1.0)
        voltage_offset = mapping.get('voltage_offset', 0.0)
        
        # Calculate effective barrier voltage
        effective_voltage = (barrier_voltage - voltage_offset) * voltage_scale
        
        if coupling_type == "dot_to_dot":
            # Barrier controls coupling between two dots
            target_dots = mapping['target_dots']
            dot1, dot2 = target_dots[0], target_dots[1]
            base_coupling = mapping.get('base_coupling', 30e-6)
            
            # Update both directions (symmetric matrix)
            coupling_strength = base_coupling * np.exp(-effective_voltage**2)
            new_couplings[dot1, dot2] = coupling_strength
            new_couplings[dot2, dot1] = coupling_strength
            
        elif coupling_type in ["reservoir_to_dot", "dot_to_reservoir"]:
            # Barrier controls coupling between reservoir and dot
            target_dot = mapping['target_dot']
            base_coupling = mapping.get('base_coupling', 30e-6)
            
            # For reservoir coupling, we modify the diagonal element
            # This represents the coupling strength to the reservoir
            coupling_strength = base_coupling * np.exp(-effective_voltage**2)
            new_couplings[target_dot, target_dot] = coupling_strength
    
    return new_couplings


class SimpleQDartsV4:
    """
    Simplified QDarts v4 simulator with barrier voltage support.
    
    This class provides a clean interface for quantum device simulation
    with barrier voltage control, leveraging experiment.py for all physics.
    """
    
    def __init__(self, config_path='qdarts_config_v4.yaml'):
        """
        Initialize SimpleQDartsV4 simulator.
        
        Args:
            config_path: Path to configuration file
        """
        self.start_time = time.time()
        
        # Load and validate configuration
        self.config = load_qdarts_config(config_path)
        
        # Print configuration summary
        if self.config.get('logging', {}).get('print_logs', True):
            print_config_summary(self.config)
        
        # Store barrier configuration
        self.barrier_config = self.config['simulator']['barrier']
        self.barrier_mappings = self.barrier_config['barrier_mappings']
        self.default_barrier_voltages = self.barrier_config.get('default_barrier_voltages', {})
        
        # Initialize current barrier voltages
        self.current_barrier_voltages = self.default_barrier_voltages.copy()
        
        # Store base tunnel couplings for dynamic updates
        self.base_tunnel_couplings = np.array(self.config['simulator']['tunneling']['tunnel_couplings'])
        
        # Calculate initial tunnel couplings with default barrier voltages
        initial_couplings = _calculate_barrier_tunnel_couplings(
            self.base_tunnel_couplings, 
            self.barrier_config, 
            self.current_barrier_voltages
        )
        
        # Update tunneling config with calculated couplings
        tunneling_config = self.config['simulator']['tunneling'].copy()
        tunneling_config['tunnel_couplings'] = initial_couplings
        
        # Initialize experiment with updated tunnel couplings
        self.experiment = Experiment(
            capacitance_config=self.config['simulator']['capacitance'],
            tunneling_config=tunneling_config,
            sensor_config=self.config['simulator']['sensor'],
            print_logs=self.config.get('logging', {}).get('print_logs', True)
        )
        
        if self.config.get('logging', {}).get('print_logs', True):
            print(f"SimpleQDartsV4 initialized in {time.time() - self.start_time:.2f} seconds")
    
    def set_barrier_voltages(self, barrier_voltages):
        """
        Set barrier voltages and update tunnel couplings.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages (e.g., {'barrier_2': 0.5})
        """
        # Validate barrier names
        for barrier_name in barrier_voltages:
            if barrier_name not in self.barrier_mappings:
                raise ValueError(f"Unknown barrier: {barrier_name}. Available: {list(self.barrier_mappings.keys())}")
        
        # Update current barrier voltages
        for barrier_name, voltage in barrier_voltages.items():
            if barrier_name in self.barrier_mappings:
                self.current_barrier_voltages[barrier_name] = voltage
        
        # Calculate new tunnel couplings
        new_couplings = _calculate_barrier_tunnel_couplings(
            self.base_tunnel_couplings,
            self.barrier_config,
            self.current_barrier_voltages
        )
        
        # Update tunnel couplings directly (no reinitialization needed!)
        if hasattr(self.experiment, 'tunneling_sim'):
            self.experiment.tunneling_sim.tunnel_matrix = new_couplings
        
        if self.config.get('logging', {}).get('print_logs', True):
            print(f"Barrier voltages updated: {self.current_barrier_voltages}")
    
    def compute_csd(self, x_voltages, y_voltages, plane_axes, **kwargs):
        """
        Compute charge stability diagram with barrier voltage support.
        
        Args:
            x_voltages: X-axis voltage range
            y_voltages: Y-axis voltage range
            plane_axes: Gate indices for x and y axes
            **kwargs: Additional arguments passed to experiment.generate_CSD()
                     including advanced features like use_virtual_gates, 
                     compensate_sensors, center_transition, etc.
        
        Returns:
            CSD data tuple: (xout, yout, CSD_data, polytopes, sensor_values, v_offset)
        """
        # Get defaults from config
        features = self.config.get('features', {})
        target_state = kwargs.get('target_state', self.config['device']['target_state'])
        
        # Call experiment.generate_CSD() with all parameters
        return self.experiment.generate_CSD(
            x_voltages=x_voltages,
            y_voltages=y_voltages,
            plane_axes=plane_axes,
            target_state=target_state,
            target_transition=kwargs.get('target_transition', features.get('target_transition', [1, -1])) if kwargs.get('center_transition', features.get('center_transition', False)) else None,
            use_virtual_gates=kwargs.get('use_virtual_gates', features.get('use_virtual_gates', False)),
            compensate_sensors=kwargs.get('compensate_sensors', features.get('compensate_sensors', False)),
            compute_polytopes=kwargs.get('compute_polytopes', features.get('compute_polytopes', False)),
            use_sensor_signal=kwargs.get('use_sensor_signal', features.get('use_sensor_signal', True)),
            **{k: v for k, v in kwargs.items() if k not in ['use_virtual_gates', 'compensate_sensors', 'center_transition', 'target_transition', 'compute_polytopes', 'use_sensor_signal', 'target_state']}
        )
    
    def plot_csd(self, csd_data, save_path=None):
        """
        Plot charge stability diagram with simple visualization.
        
        Args:
            csd_data: CSD data tuple from compute_csd()
            save_path: Optional custom save path (if None, uses config filenames)
        """
        # Extract data from csd_data tuple
        xout, yout, CSD_data, polytopes, sensor_values, v_offset = csd_data
        
        # Create coordinate arrays
        xs = xout
        ys = yout
        
        # Handle CSD data dimensions - extract first slice if 3D
        if CSD_data.ndim > 2:
            plot_data = CSD_data[:, :, 0]  # Take first slice along last dimension
        else:
            plot_data = CSD_data
        
        # Get config for plotting
        output_config = self.config.get('output', {})
        plots_config = output_config.get('plots', {})
        filenames_config = output_config.get('filenames', {})
        
        # Plot 1: Charge Stability Diagram
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(xs, ys, plot_data.T)
        plt.xlabel('Gate 0 Voltage (V)')
        plt.ylabel('Gate 1 Voltage (V)')
        plt.title('Charge Stability Diagram')
        plt.colorbar(label='Charge State')
        
        # Auto-save with config filename
        csd_filename = save_path or filenames_config.get('csd_plot', 'qdarts_v4_csd_plot.png')
        plt.savefig(csd_filename, 
                   dpi=plots_config.get('dpi', 300), 
                   bbox_inches=plots_config.get('bbox_inches', 'tight'))
        plt.close()
        
        if self.config.get('logging', {}).get('print_logs', True):
            print(f"CSD plot saved as {csd_filename}")
        
  
    
    def get_current_tunnel_couplings(self):
        """
        Get current tunnel coupling matrix.
        
        Returns:
            Current tunnel coupling matrix
        """
        if hasattr(self.experiment, 'tunneling_sim'):
            return self.experiment.tunneling_sim.tunnel_matrix
        else:
            return self.base_tunnel_couplings
    
    def get_barrier_voltage_effect(self, barrier_name, voltage_range, resolution=50):
        """
        Calculate the effect of a barrier voltage on tunnel couplings.
        
        Args:
            barrier_name: Name of the barrier (e.g., 'barrier_2')
            voltage_range: [min_voltage, max_voltage]
            resolution: Number of voltage points
            
        Returns:
            Dictionary with voltage points and corresponding coupling strengths
        """
        if barrier_name not in self.barrier_mappings:
            raise ValueError(f"Unknown barrier: {barrier_name}")
        
        mapping = self.barrier_mappings[barrier_name]
        coupling_type = mapping['coupling_type']
        base_coupling = mapping.get('base_coupling', 30e-6)
        voltage_scale = mapping.get('voltage_scale', 1.0)
        voltage_offset = mapping.get('voltage_offset', 0.0)
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], resolution)
        couplings = []
        
        for voltage in voltages:
            effective_voltage = (voltage - voltage_offset) * voltage_scale
            coupling_strength = base_coupling * np.exp(-effective_voltage**2)
            couplings.append(coupling_strength)
        
        return {
            'voltages': voltages,
            'couplings': couplings,
            'coupling_type': coupling_type,
            'base_coupling': base_coupling
        }


def main():
    """
    Example usage of SimpleQDartsV4.
    """
    # Initialize simulator
    print("Initializing SimpleQDartsV4...")
    simulator = SimpleQDartsV4()
    
    # Set barrier voltages
    print("Setting barrier voltages...")
    simulator.set_barrier_voltages({
        'barrier_2': 0.0,
        'barrier_3': 0.0,
        'barrier_4': 0.0
    })
    
    # Get measurement configuration
    measurement_config = simulator.config['measurement']['sweep']
    x_config = measurement_config['x_axis']
    y_config = measurement_config['y_axis']
    
    # Generate voltage ranges
    x_voltages = np.linspace(x_config['voltage_range'][0], 
                           x_config['voltage_range'][1], 
                           x_config['resolution'])
    y_voltages = np.linspace(y_config['voltage_range'][0], 
                           y_config['voltage_range'][1], 
                           y_config['resolution'])
    
    # Set plane axes
    plane_axes = [x_config['gate'], y_config['gate']]
    
    # Compute CSD with advanced features
    print("Computing charge stability diagram...")
    csd_data = simulator.compute_csd(
        x_voltages=x_voltages,
        y_voltages=y_voltages,
        plane_axes=plane_axes,
    )
    print(csd_data[2])
    # Plot results
    print("Creating plots...")
    simulator.plot_csd(csd_data)
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main() 