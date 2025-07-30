import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
import sys

# Add qdarts to path
sys.path.append(os.path.dirname(__file__))

from qdarts_config_loader_v4 import load_qdarts_config, print_config_summary, get_barrier_voltage_vector
from qdarts.plotting import get_polytopes, plot_polytopes, get_CSD_data
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator
from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator
from qdarts.noise_processes import OU_process
from qdarts.experiment import Experiment


def plot_charge_stability_diagram(csd_data, output_prefix='qdarts_v4', 
                                 save_plots=True, show_plots=False, dpi=300):
    """
    Plot charge stability diagram with comprehensive visualization.
    
    Args:
        csd_data: Tuple of (xout, yout, data, polytopes, sensor_values, v_offset)
        output_prefix: Prefix for output files
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots
        dpi: DPI for saved plots
    """
    xout, yout, data, polytopes, sensor_values, v_offset = csd_data
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QDarts v4 - Charge Stability Diagram Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Original CSD data
    ax1 = axes[0, 0]
    # Handle data dimensions - data might be 3D, take first slice if needed
    plot_data = data[:, :, 0] if data.ndim > 2 else data
    # Handle NaN values
    plot_data_clean = np.nan_to_num(plot_data, nan=0.0)
    im1 = ax1.pcolormesh(xout, yout, plot_data_clean.T, cmap='viridis', shading='auto')
    ax1.set_xlabel('Gate 0 Voltage (V)')
    ax1.set_ylabel('Gate 1 Voltage (V)')
    ax1.set_title('Charge Stability Diagram')
    plt.colorbar(im1, ax=ax1, label='Charge State')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sensor response (if available)
    ax2 = axes[0, 1]
    if sensor_values is not None and sensor_values.size > 0:
        # Use first sensor if multiple sensors
        sensor_data = sensor_values[:, :, 0] if sensor_values.ndim > 2 else sensor_values
        im2 = ax2.pcolormesh(xout, yout, sensor_data.T, cmap='plasma', shading='auto')
        ax2.set_xlabel('Gate 0 Voltage (V)')
        ax2.set_ylabel('Gate 1 Voltage (V)')
        ax2.set_title('Sensor Response')
        plt.colorbar(im2, ax=ax2, label='Sensor Conductance')
    else:
        ax2.text(0.5, 0.5, 'No sensor data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Sensor Response (No Data)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CSD with polytopes (if available)
    ax3 = axes[1, 0]
    im3 = ax3.pcolormesh(xout, yout, plot_data_clean.T, cmap='viridis', shading='auto')
    if polytopes is not None and len(polytopes) > 0:
        plot_polytopes(ax3, polytopes, fontsize=8, color='white')
    ax3.set_xlabel('Gate 0 Voltage (V)')
    ax3.set_ylabel('Gate 1 Voltage (V)')
    ax3.set_title('CSD with Polytope Boundaries')
    plt.colorbar(im3, ax=ax3, label='Charge State')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data statistics and information
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text box with statistics
    stats_text = f"""
    CSD Statistics:
    • Data shape: {data.shape}
    • Plot data shape: {plot_data.shape}
    • Voltage range X: [{xout.min():.3f}, {xout.max():.3f}] V
    • Voltage range Y: [{yout.min():.3f}, {yout.max():.3f}] V
    • Charge states: {plot_data_clean.min():.0f} to {plot_data_clean.max():.0f}
    • Unique states: {len(np.unique(plot_data_clean))}
    • NaN values: {np.isnan(plot_data).sum()}
    
    Sensor Data:
    • Available: {'Yes' if sensor_values is not None else 'No'}
    • Shape: {sensor_values.shape if sensor_values is not None else 'N/A'}
    
    Polytopes:
    • Available: {'Yes' if polytopes is not None else 'No'}
    • Count: {len(polytopes) if polytopes is not None else 0}
    
    Voltage Offset: {v_offset}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        # Save combined plot
        combined_filename = f"{output_prefix}_csd_combined.png"
        plt.savefig(combined_filename, dpi=dpi, bbox_inches='tight')
        print(f"Combined CSD plot saved as: {combined_filename}")
        
        # Save individual plots
        for i, (ax, title) in enumerate([(ax1, 'csd'), (ax2, 'sensor'), (ax3, 'polytopes')]):
            fig_single = plt.figure(figsize=(8, 6))
            ax_single = fig_single.add_subplot(111)
            
            if title == 'csd':
                im = ax_single.pcolormesh(xout, yout, plot_data_clean.T, cmap='viridis', shading='auto')
                ax_single.set_title('Charge Stability Diagram')
                plt.colorbar(im, ax=ax_single, label='Charge State')
            elif title == 'sensor' and sensor_values is not None:
                sensor_data = sensor_values[:, :, 0] if sensor_values.ndim > 2 else sensor_values
                im = ax_single.pcolormesh(xout, yout, sensor_data.T, cmap='plasma', shading='auto')
                ax_single.set_title('Sensor Response')
                plt.colorbar(im, ax=ax_single, label='Sensor Conductance')
            elif title == 'polytopes':
                im = ax_single.pcolormesh(xout, yout, plot_data_clean.T, cmap='viridis', shading='auto')
                if polytopes is not None:
                    plot_polytopes(ax_single, polytopes, fontsize=8, color='white')
                ax_single.set_title('CSD with Polytope Boundaries')
                plt.colorbar(im, ax=ax_single, label='Charge State')
            
            ax_single.set_xlabel('Gate 0 Voltage (V)')
            ax_single.set_ylabel('Gate 1 Voltage (V)')
            ax_single.grid(True, alpha=0.3)
            
            filename = f"{output_prefix}_{title}.png"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"{title.title()} plot saved as: {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_barrier_sweep_results(sweep_results, barrier_gate, output_prefix='qdarts_v4', 
                              save_plots=True, show_plots=False, dpi=300):
    """
    Plot barrier sweep results with comprehensive visualization.
    
    Args:
        sweep_results: List of dictionaries with 'barrier_voltage' and 'csd_data' keys
        barrier_gate: Barrier gate number being swept
        output_prefix: Prefix for output files
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots
        dpi: DPI for saved plots
    """
    if not sweep_results:
        print("No sweep results to plot")
        return
    
    # Extract data
    barrier_voltages = [r['barrier_voltage'] for r in sweep_results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'QDarts v4 - Barrier {barrier_gate} Sweep Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: CSD evolution (first, middle, last)
    ax1 = axes[0, 0]
    indices_to_plot = [0, len(sweep_results)//2, len(sweep_results)-1]
    colors = ['blue', 'orange', 'red']
    
    for i, idx in enumerate(indices_to_plot):
        if idx < len(sweep_results):
            csd_data = sweep_results[idx]['csd_data']
            xout, yout, data, _, _, _ = csd_data
            voltage = sweep_results[idx]['barrier_voltage']
            
            # Handle 3D data
            plot_data = data[:, :, 0] if data.ndim > 2 else data
            plot_data_clean = np.nan_to_num(plot_data, nan=0.0)
            
            # Plot a slice through the middle of the data
            mid_y = len(yout) // 2
            ax1.plot(xout, plot_data_clean[:, mid_y], color=colors[i], 
                    label=f'V_barrier = {voltage:.2f} V', linewidth=2)
    
    ax1.set_xlabel('Gate 0 Voltage (V)')
    ax1.set_ylabel('Charge State')
    ax1.set_title(f'CSD Evolution (Y-slice)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Barrier voltage effect on coupling strength
    ax2 = axes[0, 1]
    # This would need to be calculated from the barrier configuration
    # For now, show a theoretical curve
    theoretical_couplings = [30e-6 * np.exp(-v**2) for v in barrier_voltages]
    ax2.plot(barrier_voltages, theoretical_couplings, 'b-', linewidth=2, label='Theoretical')
    ax2.set_xlabel(f'Barrier {barrier_gate} Voltage (V)')
    ax2.set_ylabel('Tunnel Coupling (eV)')
    ax2.set_title(f'Barrier {barrier_gate} Effect on Coupling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: CSD comparison (first vs last)
    ax3 = axes[1, 0]
    if len(sweep_results) >= 2:
        # First result
        csd_data_first = sweep_results[0]['csd_data']
        xout, yout, data_first, _, _, _ = csd_data_first
        # Handle 3D data
        plot_data_first = data_first[:, :, 0] if data_first.ndim > 2 else data_first
        plot_data_first_clean = np.nan_to_num(plot_data_first, nan=0.0)
        im3 = ax3.pcolormesh(xout, yout, plot_data_first_clean.T, cmap='viridis', shading='auto')
        ax3.set_xlabel('Gate 0 Voltage (V)')
        ax3.set_ylabel('Gate 1 Voltage (V)')
        ax3.set_title(f'CSD at V_barrier = {sweep_results[0]["barrier_voltage"]:.2f} V')
        plt.colorbar(im3, ax=ax3, label='Charge State')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sweep statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    num_states_evolution = []
    for result in sweep_results:
        csd_data = result['csd_data']
        _, _, data, _, _, _ = csd_data
        # Handle 3D data
        plot_data = data[:, :, 0] if data.ndim > 2 else data
        plot_data_clean = np.nan_to_num(plot_data, nan=0.0)
        num_states_evolution.append(len(np.unique(plot_data_clean)))
    
    stats_text = f"""
    Barrier Sweep Statistics:
    • Barrier gate: {barrier_gate}
    • Voltage range: [{min(barrier_voltages):.2f}, {max(barrier_voltages):.2f}] V
    • Number of points: {len(sweep_results)}
    
    CSD Evolution:
    • Initial states: {num_states_evolution[0] if num_states_evolution else 'N/A'}
    • Final states: {num_states_evolution[-1] if num_states_evolution else 'N/A'}
    • Max states: {max(num_states_evolution) if num_states_evolution else 'N/A'}
    
    Data Quality:
    • All CSD data valid: {'Yes' if all('csd_data' in r for r in sweep_results) else 'No'}
    • Sensor data available: {'Yes' if any(len(r['csd_data']) > 4 and r['csd_data'][4] is not None for r in sweep_results) else 'No'}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        # Save combined plot
        combined_filename = f"{output_prefix}_barrier_sweep_combined.png"
        plt.savefig(combined_filename, dpi=dpi, bbox_inches='tight')
        print(f"Combined barrier sweep plot saved as: {combined_filename}")
        
        # Save individual plots
        for i, (ax, title) in enumerate([(ax1, 'evolution'), (ax2, 'coupling'), (ax3, 'comparison')]):
            fig_single = plt.figure(figsize=(8, 6))
            ax_single = fig_single.add_subplot(111)
            
            if title == 'evolution':
                for i, idx in enumerate(indices_to_plot):
                    if idx < len(sweep_results):
                        csd_data = sweep_results[idx]['csd_data']
                        xout, yout, data, _, _, _ = csd_data
                        voltage = sweep_results[idx]['barrier_voltage']
                        # Handle 3D data
                        plot_data = data[:, :, 0] if data.ndim > 2 else data
                        plot_data_clean = np.nan_to_num(plot_data, nan=0.0)
                        mid_y = len(yout) // 2
                        ax_single.plot(xout, plot_data_clean[:, mid_y], color=colors[i], 
                                     label=f'V_barrier = {voltage:.2f} V', linewidth=2)
                ax_single.set_xlabel('Gate 0 Voltage (V)')
                ax_single.set_ylabel('Charge State')
                ax_single.set_title(f'CSD Evolution (Barrier {barrier_gate})')
                ax_single.legend()
                ax_single.grid(True, alpha=0.3)
            
            elif title == 'coupling':
                ax_single.plot(barrier_voltages, theoretical_couplings, 'b-', linewidth=2)
                ax_single.set_xlabel(f'Barrier {barrier_gate} Voltage (V)')
                ax_single.set_ylabel('Tunnel Coupling (eV)')
                ax_single.set_title(f'Barrier {barrier_gate} Effect on Coupling')
                ax_single.grid(True, alpha=0.3)
                ax_single.set_yscale('log')
            
            elif title == 'comparison' and len(sweep_results) >= 2:
                csd_data_first = sweep_results[0]['csd_data']
                xout, yout, data_first, _, _, _ = csd_data_first
                # Handle 3D data
                plot_data_first = data_first[:, :, 0] if data_first.ndim > 2 else data_first
                plot_data_first_clean = np.nan_to_num(plot_data_first, nan=0.0)
                im = ax_single.pcolormesh(xout, yout, plot_data_first_clean.T, cmap='viridis', shading='auto')
                ax_single.set_xlabel('Gate 0 Voltage (V)')
                ax_single.set_ylabel('Gate 1 Voltage (V)')
                ax_single.set_title(f'CSD at V_barrier = {sweep_results[0]["barrier_voltage"]:.2f} V')
                plt.colorbar(im, ax=ax_single, label='Charge State')
                ax_single.grid(True, alpha=0.3)
            
            filename = f"{output_prefix}_barrier_sweep_{title}.png"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Barrier sweep {title} plot saved as: {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_barrier_voltage_effect(effect_data, barrier_name, output_prefix='qdarts_v4',
                               save_plots=True, show_plots=False, dpi=300):
    """
    Plot barrier voltage effect on tunnel couplings.
    
    Args:
        effect_data: Dictionary with 'voltages', 'couplings', 'coupling_type', 'base_coupling'
        barrier_name: Name of the barrier
        output_prefix: Prefix for output files
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots
        dpi: DPI for saved plots
    """
    voltages = effect_data['voltages']
    couplings = effect_data['couplings']
    coupling_type = effect_data['coupling_type']
    base_coupling = effect_data['base_coupling']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'QDarts v4 - {barrier_name} Voltage Effect Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Coupling strength vs voltage
    ax1.plot(voltages, couplings, 'b-', linewidth=2, label='Tunnel Coupling')
    ax1.axhline(y=base_coupling, color='r', linestyle='--', alpha=0.7, label='Base Coupling')
    ax1.set_xlabel(f'{barrier_name} Voltage (V)')
    ax1.set_ylabel('Tunnel Coupling (eV)')
    ax1.set_title(f'{barrier_name} Effect on {coupling_type.replace("_", " ").title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for better visualization
    ax2.semilogy(voltages, couplings, 'b-', linewidth=2, label='Tunnel Coupling')
    ax2.axhline(y=base_coupling, color='r', linestyle='--', alpha=0.7, label='Base Coupling')
    ax2.set_xlabel(f'{barrier_name} Voltage (V)')
    ax2.set_ylabel('Tunnel Coupling (eV)')
    ax2.set_title(f'{barrier_name} Effect (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{output_prefix}_barrier_effect_{barrier_name}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Barrier effect plot saved as: {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


class BarrierExperiment:
    """
    Wrapper around Experiment class with barrier voltage support.
    
    This class extends the Experiment functionality to support:
    1. Dynamic barrier voltage control
    2. Reservoir coupling simulation
    3. Barrier voltage sweeps
    4. Integration with experiment.py's advanced features
    """
    
    def __init__(self, capacitance_config, tunneling_config=None, sensor_config=None, 
                 barrier_config=None, print_logs=True):
        """
        Initialize BarrierExperiment with barrier support.
        
        Args:
            capacitance_config: Capacitance configuration (plunger gates only)
            tunneling_config: Tunneling configuration
            sensor_config: Sensor configuration
            barrier_config: Barrier control configuration
            print_logs: Whether to print logs
        """
        self.barrier_config = barrier_config
        self.print_logs = print_logs
        
        # Initialize the base experiment
        self.experiment = Experiment(
            capacitance_config=capacitance_config,
            tunneling_config=tunneling_config,
            sensor_config=sensor_config,
            print_logs=print_logs
        )
        
        # Store barrier configuration
        if barrier_config is not None:
            self.barrier_gates = barrier_config['barrier_gates']
            self.barrier_mappings = barrier_config['barrier_mappings']
            self.default_barrier_voltages = barrier_config.get('default_barrier_voltages', {})
            
            # Initialize barrier voltages
            self.current_barrier_voltages = self.default_barrier_voltages.copy()
        else:
            self.barrier_gates = []
            self.barrier_mappings = {}
            self.current_barrier_voltages = {}
        
        # Store base tunnel couplings for dynamic updates
        if tunneling_config is not None:
            self.base_tunnel_couplings = np.array(tunneling_config['tunnel_couplings'])
        else:
            self.base_tunnel_couplings = None
        
        # Update tunnel couplings with current barrier voltages
        self._update_tunnel_couplings()
    
    def set_barrier_voltages(self, barrier_voltages):
        """
        Set barrier voltages and update tunnel couplings.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages (e.g., {'barrier_2': 0.5})
        """
        # Update current barrier voltages
        for barrier_name, voltage in barrier_voltages.items():
            if barrier_name in self.barrier_mappings:
                self.current_barrier_voltages[barrier_name] = voltage
        
        # Update tunnel couplings based on new barrier voltages
        self._update_tunnel_couplings()
        
        if self.print_logs:
            print(f"Barrier voltages updated: {self.current_barrier_voltages}")
    
    def _update_tunnel_couplings(self):
        """
        Update tunnel couplings based on current barrier voltages.
        
        This method implements the barrier voltage to tunnel coupling mapping:
        - Barrier 2: reservoir <-> dot 0 coupling
        - Barrier 3: dot 0 <-> dot 1 coupling  
        - Barrier 4: dot 1 <-> reservoir coupling
        """
        if self.base_tunnel_couplings is None:
            return
        
        # Start with base tunnel couplings
        new_couplings = self.base_tunnel_couplings.copy()
        
        # Apply barrier voltage effects
        for barrier_name, mapping in self.barrier_mappings.items():
            if barrier_name not in self.current_barrier_voltages:
                continue
                
            barrier_voltage = self.current_barrier_voltages[barrier_name]
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
        
        # Update the tunneling simulator with new couplings
        if hasattr(self.experiment, 'tunneling_sim'):
            self.experiment.tunneling_sim.tunnel_matrix = new_couplings
    
    def generate_CSD(self, x_voltages, y_voltages, plane_axes, target_state=None,
                    target_transition=None, use_virtual_gates=False,
                    compensate_sensors=False, compute_polytopes=False,
                    use_sensor_signal=False, v_offset=None):
        """
        Generate charge stability diagram with barrier voltage support.
        
        Args:
            x_voltages: X-axis voltage range
            y_voltages: Y-axis voltage range
            plane_axes: Gate indices for x and y axes
            target_state: Target charge state
            target_transition: Target transition
            use_virtual_gates: Whether to use virtual gates
            compensate_sensors: Whether to compensate sensors
            compute_polytopes: Whether to compute polytopes
            use_sensor_signal: Whether to use sensor signal
            v_offset: Voltage offset
            
        Returns:
            CSD data and metadata
        """
        # Update tunnel couplings before generating CSD
        self._update_tunnel_couplings()
        
        # Call the base experiment's generate_CSD method
        return self.experiment.generate_CSD(
            x_voltages=x_voltages,
            y_voltages=y_voltages,
            plane_axes=plane_axes,
            target_state=target_state,
            target_transition=target_transition,
            use_virtual_gates=use_virtual_gates,
            compensate_sensors=compensate_sensors,
            compute_polytopes=compute_polytopes,
            use_sensor_signal=use_sensor_signal,
            v_offset=v_offset
        )
    
    def barrier_sweep(self, barrier_gate, voltage_range, resolution, 
                     fixed_voltages=None, target_state=None):
        """
        Perform a barrier voltage sweep.
        
        Args:
            barrier_gate: Barrier gate to sweep
            voltage_range: [min_voltage, max_voltage]
            resolution: Number of voltage points
            fixed_voltages: Fixed voltages for other barriers
            target_state: Target charge state
            
        Returns:
            Sweep results
        """
        if fixed_voltages is None:
            fixed_voltages = {}
        
        # Set fixed barrier voltages
        self.set_barrier_voltages(fixed_voltages)
        
        # Generate voltage points
        voltages = np.linspace(voltage_range[0], voltage_range[1], resolution)
        
        # Store results
        results = []
        
        for voltage in voltages:
            # Set current barrier voltage
            barrier_name = f"barrier_{barrier_gate}"
            self.set_barrier_voltages({barrier_name: voltage})
            
            # Generate CSD for this barrier voltage
            # Use a simple 2D sweep for demonstration
            x_voltages = np.linspace(-0.5, 0.5, 50)
            y_voltages = np.linspace(-0.5, 0.5, 50)
            plane_axes = [0, 1]  # Plunger gates
            
            csd_data = self.generate_CSD(
                x_voltages=x_voltages,
                y_voltages=y_voltages,
                plane_axes=plane_axes,
                target_state=target_state,
                use_sensor_signal=True
            )
            
            results.append({
                'barrier_voltage': voltage,
                'csd_data': csd_data
            })
        
        return results
    
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


class QDartsV4Simulator:
    """
    Main QDarts v4 simulator with barrier voltage support.
    
    This class provides a high-level interface for quantum device simulation
    with advanced features including barrier voltage control, sensor detuning,
    virtual gates, and transition centering.
    """
    
    def __init__(self, config_path='qdarts_config_v4.yaml'):
        """
        Initialize QDarts v4 simulator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load and validate configuration
        self.config = load_qdarts_config(config_path)
        
        # Print configuration summary
        print_config_summary(self.config)
        
        # Initialize barrier experiment
        self.barrier_experiment = BarrierExperiment(
            capacitance_config=self.config['simulator']['capacitance'],
            tunneling_config=self.config['simulator']['tunneling'],
            sensor_config=self.config['simulator']['sensor'],
            barrier_config=self.config['simulator']['barrier'],
            print_logs=True
        )
        
        # Store configuration references
        self.device_config = self.config['device']
        self.measurement_config = self.config['measurement']
        self.features_config = self.config['features']
    
    def compute_charge_stability_diagram(self, barrier_voltages=None, 
                                       use_virtual_gates=None, 
                                       compensate_sensors=None,
                                       center_transition=None,
                                       target_transition=None):
        """
        Compute charge stability diagram with optional barrier control.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages
            use_virtual_gates: Whether to use virtual gates
            compensate_sensors: Whether to compensate sensors
            center_transition: Whether to center on transition
            target_transition: Target transition to center on
            
        Returns:
            CSD data and metadata
        """
        # Set barrier voltages if provided
        if barrier_voltages is not None:
            self.barrier_experiment.set_barrier_voltages(barrier_voltages)
        
        # Get measurement configuration
        sweep_config = self.measurement_config['sweep']
        x_config = sweep_config['x_axis']
        y_config = sweep_config['y_axis']
        
        # Generate voltage ranges
        x_voltages = np.linspace(x_config['voltage_range'][0], 
                               x_config['voltage_range'][1], 
                               x_config['resolution'])
        y_voltages = np.linspace(y_config['voltage_range'][0], 
                               y_config['voltage_range'][1], 
                               y_config['resolution'])
        
        # Set plane axes
        plane_axes = [x_config['gate'], y_config['gate']]
        
        # Get feature settings
        if use_virtual_gates is None:
            use_virtual_gates = self.features_config.get('use_virtual_gates', False)
        if compensate_sensors is None:
            compensate_sensors = self.features_config.get('compensate_sensors', False)
        if center_transition is None:
            center_transition = self.features_config.get('center_transition', False)
        if target_transition is None:
            target_transition = self.features_config.get('target_transition', [1, -1])
        
        # Generate CSD
        csd_data = self.barrier_experiment.generate_CSD(
            x_voltages=x_voltages,
            y_voltages=y_voltages,
            plane_axes=plane_axes,
            target_state=self.device_config['target_state'],
            target_transition=target_transition if center_transition else None,
            use_virtual_gates=use_virtual_gates,
            compensate_sensors=compensate_sensors,
            compute_polytopes=self.features_config.get('compute_polytopes', False),
            use_sensor_signal=self.features_config.get('use_sensor_signal', True)
        )
        
        return csd_data
    
    def barrier_voltage_sweep(self, barrier_gate, voltage_range=None, resolution=None):
        """
        Perform a barrier voltage sweep.
        
        Args:
            barrier_gate: Barrier gate to sweep
            voltage_range: [min_voltage, max_voltage]
            resolution: Number of voltage points
            
        Returns:
            Sweep results
        """
        # Get default values from config
        if voltage_range is None:
            barrier_sweep_config = self.measurement_config.get('barrier_sweep', {})
            voltage_range = barrier_sweep_config.get('voltage_range', [-1.0, 1.0])
        if resolution is None:
            barrier_sweep_config = self.measurement_config.get('barrier_sweep', {})
            resolution = barrier_sweep_config.get('resolution', 50)
        
        # Get fixed barrier voltages
        fixed_voltages = {}
        barrier_sweep_config = self.measurement_config.get('barrier_sweep', {})
        if 'fixed_barrier_voltages' in barrier_sweep_config:
            fixed_voltages = barrier_sweep_config['fixed_barrier_voltages']
        
        # Perform sweep
        results = self.barrier_experiment.barrier_sweep(
            barrier_gate=barrier_gate,
            voltage_range=voltage_range,
            resolution=resolution,
            fixed_voltages=fixed_voltages,
            target_state=self.device_config['target_state']
        )
        
        return results
    
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
        return self.barrier_experiment.get_barrier_voltage_effect(
            barrier_name=barrier_name,
            voltage_range=voltage_range,
            resolution=resolution
        )
    
    def set_barrier_voltages(self, barrier_voltages):
        """
        Set barrier voltages.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages
        """
        self.barrier_experiment.set_barrier_voltages(barrier_voltages)
    
    def get_current_tunnel_couplings(self):
        """
        Get current tunnel coupling matrix.
        
        Returns:
            Current tunnel coupling matrix
        """
        return self.barrier_experiment.get_current_tunnel_couplings()
    
    def plot_csd(self, csd_data=None, output_prefix='qdarts_v4', 
                save_plots=True, show_plots=False, dpi=300):
        """
        Plot charge stability diagram.
        
        Args:
            csd_data: CSD data tuple (if None, will compute new CSD)
            output_prefix: Prefix for output files
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            dpi: DPI for saved plots
        """
        if csd_data is None:
            csd_data = self.compute_charge_stability_diagram()
        
        plot_charge_stability_diagram(csd_data, output_prefix, save_plots, show_plots, dpi)
        return csd_data
    
    def plot_barrier_sweep(self, sweep_results, barrier_gate, output_prefix='qdarts_v4',
                          save_plots=True, show_plots=False, dpi=300):
        """
        Plot barrier sweep results.
        
        Args:
            sweep_results: List of sweep result dictionaries
            barrier_gate: Barrier gate number
            output_prefix: Prefix for output files
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            dpi: DPI for saved plots
        """
        plot_barrier_sweep_results(sweep_results, barrier_gate, output_prefix, 
                                 save_plots, show_plots, dpi)
    
    def plot_barrier_effect(self, barrier_name, voltage_range=None, resolution=100,
                           output_prefix='qdarts_v4', save_plots=True, show_plots=False, dpi=300):
        """
        Plot barrier voltage effect on tunnel couplings.
        
        Args:
            barrier_name: Name of the barrier (e.g., 'barrier_2')
            voltage_range: [min_voltage, max_voltage] (if None, uses [-2.0, 2.0])
            resolution: Number of voltage points
            output_prefix: Prefix for output files
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            dpi: DPI for saved plots
        """
        if voltage_range is None:
            voltage_range = [-2.0, 2.0]
        
        effect_data = self.get_barrier_voltage_effect(barrier_name, voltage_range, resolution)
        plot_barrier_voltage_effect(effect_data, barrier_name, output_prefix, 
                                  save_plots, show_plots, dpi)
        return effect_data


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='QDarts v4 Quantum Device Simulator')
    parser.add_argument('--config', type=str, default='qdarts_config_v4.yaml',
                       help='Configuration file path')
    parser.add_argument('--barrier-sweep', type=int, metavar='GATE',
                       help='Perform barrier voltage sweep on specified gate')
    parser.add_argument('--barrier-voltage', type=float, nargs=2, metavar=('GATE', 'VOLTAGE'),
                       help='Set barrier voltage for specified gate')
    parser.add_argument('--output', type=str, default='qdarts_v4_output',
                       help='Output file prefix')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots (in addition to saving)')
    parser.add_argument('--barrier-effect', type=str, metavar='BARRIER_NAME',
                       help='Plot barrier voltage effect for specified barrier (e.g., barrier_2)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved plots (default: 300)')
    
    args = parser.parse_args()
    
    # Initialize simulator
    print("Initializing QDarts v4 Simulator...")
    simulator = QDartsV4Simulator(args.config)
    
    # Handle barrier voltage effect plotting
    if args.barrier_effect is not None:
        print(f"Calculating barrier voltage effect for {args.barrier_effect}...")
        effect_data = simulator.get_barrier_voltage_effect(
            barrier_name=args.barrier_effect,
            voltage_range=[-2.0, 2.0],
            resolution=100
        )
        
        # Save effect data
        np.save(f"{args.output}_barrier_effect_{args.barrier_effect}.npy", effect_data)
        print(f"Barrier effect data saved to {args.output}_barrier_effect_{args.barrier_effect}.npy")
        
        if not args.no_plot:
            plot_barrier_voltage_effect(effect_data, args.barrier_effect, args.output, 
                                      show_plots=args.show_plots, dpi=args.dpi)
    
    # Handle barrier voltage sweep
    elif args.barrier_sweep is not None:
        print(f"Performing barrier voltage sweep on gate {args.barrier_sweep}...")
        results = simulator.barrier_voltage_sweep(args.barrier_sweep)
        
        # Save results
        np.save(f"{args.output}_barrier_sweep.npy", results)
        print(f"Barrier sweep results saved to {args.output}_barrier_sweep.npy")
        
        if not args.no_plot:
            # Plot barrier sweep results
            plot_barrier_sweep_results(results, args.barrier_sweep, args.output,
                                     show_plots=args.show_plots, dpi=args.dpi)
    
    # Handle single barrier voltage setting
    elif args.barrier_voltage is not None:
        gate, voltage = args.barrier_voltage
        barrier_name = f"barrier_{gate}"
        simulator.set_barrier_voltages({barrier_name: voltage})
        print(f"Set {barrier_name} to {voltage} V")
        
        # Also compute and plot CSD with new barrier voltage
        print("Computing charge stability diagram with new barrier voltage...")
        csd_data = simulator.compute_charge_stability_diagram()
        
        # Save results
        results_dict = {
            'xout': csd_data[0],
            'yout': csd_data[1],
            'data': csd_data[2],
            'polytopes': csd_data[3],
            'sensor_values': csd_data[4],
            'v_offset': csd_data[5],
            'barrier_voltages': simulator.barrier_experiment.current_barrier_voltages
        }
        np.save(f"{args.output}_csd_with_barrier.npy", results_dict)
        print(f"CSD data with barrier voltage saved to {args.output}_csd_with_barrier.npy")
        
        if not args.no_plot:
            plot_charge_stability_diagram(csd_data, f"{args.output}_with_barrier",
                                        show_plots=args.show_plots, dpi=args.dpi)
    
    # Default: compute charge stability diagram
    else:
        print("Computing charge stability diagram...")
        csd_data = simulator.compute_charge_stability_diagram()
        
        # Save results - csd_data is a tuple (xout, yout, data, polytopes, sensor_values, v_offset)
        xout, yout, data, polytopes, sensor_values, v_offset = csd_data
        
        # Save as dictionary for easier access
        results_dict = {
            'xout': xout,
            'yout': yout,
            'data': data,
            'polytopes': polytopes,
            'sensor_values': sensor_values,
            'v_offset': v_offset,
            'barrier_voltages': simulator.barrier_experiment.current_barrier_voltages
        }
        np.save(f"{args.output}_csd.npy", results_dict)
        print(f"CSD data saved to {args.output}_csd.npy")
        
        # Also save individual components for easier access
        np.save(f"{args.output}_xout.npy", xout)
        np.save(f"{args.output}_yout.npy", yout)
        np.save(f"{args.output}_data.npy", data)
        np.save(f"{args.output}_sensor_values.npy", sensor_values)
        print(f"Individual components saved to {args.output}_*.npy")
        
        if not args.no_plot:
            # Plot CSD
            plot_charge_stability_diagram(csd_data, args.output, 
                                        show_plots=args.show_plots, dpi=args.dpi)
    
    print("Simulation completed.")


if __name__ == "__main__":
    main() 