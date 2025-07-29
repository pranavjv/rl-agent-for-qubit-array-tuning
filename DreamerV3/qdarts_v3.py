import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
import sys

# Add qdarts to path
sys.path.append(os.path.dirname(__file__))

from qdarts_config_loader import load_qdarts_config, print_config_summary
from qdarts.plotting import get_polytopes, plot_polytopes, get_CSD_data
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator
from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator
from qdarts.noise_processes import OU_process


class QDartsSimulator:
    """
    Configuration-driven QDarts simulator.
    """
    
    def __init__(self, config_path: str = 'qdarts_config.yaml'):
        """
        Initialize the QDarts simulator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.start_time = time.time()
        
        # Load configuration
        self.config = load_qdarts_config(config_path)
        
        # Print configuration summary
        if self.config['logging']['print_logs']:
            print_config_summary(self.config)
        
        # Initialize simulators
        self._init_capacitance_simulator()
        
        if self.config['simulator']['tunneling']['enabled']:
            self._init_tunneling_simulator()
        
        if self.config['logging']['print_logs']:
            print(f"Initialization completed in {time.time() - self.start_time:.2f} seconds")
    
    def _init_capacitance_simulator(self):
        """Initialize the capacitance simulator."""
        if self.config['logging']['print_logs']:
            print("Creating capacitance model...")
        
        # Extract capacitance parameters
        cap_config = self.config['simulator']['capacitance']
        
        # Create capacitance model
        self.capacitance_model = CapacitanceModel(
            cap_config['C_DG'],
            cap_config['C_DD'],
            cap_config['bounds_limits'],
            ks=cap_config['ks']
        )
        
        # Create device simulator
        self.capacitive_sim = CapacitiveDeviceSimulator(self.capacitance_model)
        
        if self.config['logging']['print_logs']:
            print("Capacitance simulator initialized successfully")
    
    def _init_tunneling_simulator(self):
        """Initialize the tunneling simulator with sensor model."""
        if self.config['logging']['print_logs']:
            print("Creating tunneling simulator...")
        
        # Extract tunneling parameters
        tunnel_config = self.config['simulator']['tunneling']
        sensor_config = self.config['simulator']['sensor']
        
        # Create tunnel couplings matrix
        tunnel_couplings = tunnel_config['couplings']
        
        # Create sensor model
        sensor_dots = self.config['device']['topology']['sensor_dots']
        self.sensor_model = NoisySensorDot(sensor_dots)
        
        # Configure sensor peak
        self.sensor_model.config_peak(
            g_max=sensor_config['g_max'],
            peak_width_multiplier=sensor_config['peak_width_multiplier']
        )
        
        # Configure noise if specified
        if 'noise' in sensor_config:
            noise_config = sensor_config['noise']
            
            # Create slow noise generator
            slow_noise_gen = OU_process(
                sig=noise_config['slow_noise_amplitude'],
                tc=noise_config['correlation_time'],
                dt=1,
                num_points=noise_config['virtual_samples']
            )
            
            # Configure sensor noise
            self.sensor_model.config_noise(
                sigma=noise_config['fast_noise_amplitude'],
                n_virtual_samples=noise_config['virtual_samples'],
                slow_noise_gen=slow_noise_gen
            )
        
        # Create tunneling simulator
        self.tunneling_sim = ApproximateTunnelingSimulator(
            self.capacitive_sim,
            tunnel_couplings,
            tunnel_config['temperature'],
            self.sensor_model
        )
        
        # Add slack for noise calculations
        self.capacitive_sim.set_maximum_polytope_slack(5/self.tunneling_sim.beta)
        self.tunneling_sim.num_additional_neighbours[sensor_dots] = 2
        
        if self.config['logging']['print_logs']:
            print("Tunneling simulator initialized successfully")
    
    def compute_charge_stability_diagram(self):
        """Compute the charge stability diagram."""
        if self.config['logging']['print_logs']:
            print(f"Computing boundaries for target state: {self.config['device']['target_state']}")
        
        # Get target state and compute boundaries
        target_state = self.config['device']['target_state']
        m = self.capacitive_sim.boundaries(target_state).point_inside
        
        # Extract measurement parameters
        measurement = self.config['measurement']
        minV = measurement['voltage_range']['min']
        maxV = measurement['voltage_range']['max']
        resolution = measurement['resolution']
        P = measurement['sweep_matrix']
        
        if self.config['logging']['print_logs']:
            print(f"Computing CSD data with {resolution}x{resolution} = {resolution**2} points...")
        
        # Compute CSD data
        sliced_csim, CSD_data, states = get_CSD_data(
            self.capacitive_sim, m, P, minV, maxV, resolution, target_state
        )
        
        return CSD_data, states, m, P, minV, maxV, resolution
    
    def compute_sensor_response(self, m, P, minV, maxV, resolution):
        """Compute sensor response with noise."""
        if not self.config['simulator']['tunneling']['enabled']:
            raise ValueError("Tunneling must be enabled for sensor response computation")
        
        if self.config['logging']['print_logs']:
            print("Computing sensor response...")
        
        # Find initial state for sensor scan
        state = self.tunneling_sim.poly_sim.find_state_of_voltage(m, [0, 0, 2])
        
        # Perform sensor scan
        sensor_values = self.tunneling_sim.sensor_scan_2D(m, P, minV, maxV, resolution, state)
        
        if self.config['logging']['print_logs']:
            print("Sensor scan completed")
        
        return sensor_values
    
    def create_plots(self, CSD_data, sensor_values=None, m=None, P=None, minV=None, maxV=None, resolution=None):
        """Create and save plots."""
        if self.config['logging']['print_logs']:
            print("Creating plots...")
        
        # Create coordinate arrays
        xs = np.linspace(minV[0], maxV[0], resolution)
        ys = np.linspace(minV[1], maxV[1], resolution)
        
        # Plot 1: Charge Stability Diagram
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(xs, ys, CSD_data.T)
        plt.xlim(minV[0], maxV[0])
        plt.ylim(minV[1], maxV[1])
        plt.xlabel('Gate 0 Voltage (V)')
        plt.ylabel('Gate 1 Voltage (V)')
        plt.title('Charge Stability Diagram')
        plt.colorbar(label='Charge State')
        
        # Save CSD plot
        csd_filename = self.config['output']['filenames']['csd_plot']
        plt.savefig(
            csd_filename,
            dpi=self.config['output']['plots']['dpi'],
            bbox_inches=self.config['output']['plots']['bbox_inches']
        )
        plt.close()
        
        if self.config['logging']['print_logs']:
            print(f"CSD plot saved as {csd_filename}")
        
        # Plot 2: Sensor Response (if available)
        if sensor_values is not None:
            plt.figure(figsize=(10, 8))
            # Extract first sensor data
            sensor_data = sensor_values[:, :, 0]
            plt.pcolormesh(xs, ys, sensor_data.T)
            plt.xlim(minV[0], maxV[0])
            plt.ylim(minV[1], maxV[1])
            plt.xlabel('Gate 0 Voltage (V)')
            plt.ylabel('Gate 1 Voltage (V)')
            plt.title('Sensor Response with Noise')
            plt.colorbar(label='Sensor Conductance')
            
            # Save sensor plot
            sensor_filename = self.config['output']['filenames']['sensor_plot']
            plt.savefig(
                sensor_filename,
                dpi=self.config['output']['plots']['dpi'],
                bbox_inches=self.config['output']['plots']['bbox_inches']
            )
            plt.close()
            
            if self.config['logging']['print_logs']:
                print(f"Sensor plot saved as {sensor_filename}")
        
        # Plot 3: Comparison (if both available)
        if sensor_values is not None:
            plt.figure(figsize=(15, 6))
            
            # Original CSD
            plt.subplot(1, 2, 1)
            plt.pcolormesh(xs, ys, CSD_data.T)
            plt.xlim(minV[0], maxV[0])
            plt.ylim(minV[1], maxV[1])
            plt.xlabel('Gate 0 Voltage (V)')
            plt.ylabel('Gate 1 Voltage (V)')
            plt.title('Original CSD (No Noise)')
            plt.colorbar(label='Charge State')
            
            # Noisy sensor response
            plt.subplot(1, 2, 2)
            sensor_data = sensor_values[:, :, 0]
            plt.pcolormesh(xs, ys, sensor_data.T)
            plt.xlim(minV[0], maxV[0])
            plt.ylim(minV[1], maxV[1])
            plt.xlabel('Gate 0 Voltage (V)')
            plt.ylabel('Gate 1 Voltage (V)')
            plt.title('Sensor Response with Noise')
            plt.colorbar(label='Sensor Conductance')
            
            plt.tight_layout()
            
            # Save comparison plot
            noised_filename = self.config['output']['filenames']['noised_plot']
            plt.savefig(
                noised_filename,
                dpi=self.config['output']['plots']['dpi'],
                bbox_inches=self.config['output']['plots']['bbox_inches']
            )
            plt.close()
            
            if self.config['logging']['print_logs']:
                print(f"Comparison plot saved as {noised_filename}")
    
    def run_simulation(self):
        """Run the complete simulation."""
        if self.config['logging']['print_logs']:
            print("\n" + "="*60)
            print("STARTING QDARTS SIMULATION")
            print("="*60)
        
        # Compute charge stability diagram
        CSD_data, states, m, P, minV, maxV, resolution = self.compute_charge_stability_diagram()
        
        # Compute sensor response if tunneling is enabled
        sensor_values = None
        if self.config['simulator']['tunneling']['enabled']:
            sensor_values = self.compute_sensor_response(m, P, minV, maxV, resolution)
        
        # Create plots
        self.create_plots(CSD_data, sensor_values, m, P, minV, maxV, resolution)
        
        # Print timing information
        if self.config['logging']['timing']:
            total_time = time.time() - self.start_time
            print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        if self.config['logging']['print_logs']:
            print("="*60)
            print("SIMULATION COMPLETED")
            print("="*60)
        
        return CSD_data, sensor_values


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='QDarts v3 - Configuration-driven quantum device simulator')
    parser.add_argument('--config', '-c', default='qdarts_config.yaml',
                       help='Path to configuration file (default: qdarts_config.yaml)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        # Create simulator
        simulator = QDartsSimulator(args.config)
        
        # Override debug setting if specified
        if args.debug:
            simulator.config['logging']['debug'] = True
        
        # Run simulation
        CSD_data, sensor_values = simulator.run_simulation()
        
        # Print some statistics
        if simulator.config['logging']['print_logs']:
            print(f"\nSimulation Statistics:")
            print(f"  CSD data shape: {CSD_data.shape}")
            print(f"  CSD data range: [{CSD_data.min():.2f}, {CSD_data.max():.2f}]")
            
            if sensor_values is not None:
                sensor_data = sensor_values[:, :, 0]
                print(f"  Sensor data shape: {sensor_data.shape}")
                print(f"  Sensor data range: [{sensor_data.min():.4f}, {sensor_data.max():.4f}]")
                print(f"  Sensor data mean: {sensor_data.mean():.4f}")
                print(f"  Sensor data std: {sensor_data.std():.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 