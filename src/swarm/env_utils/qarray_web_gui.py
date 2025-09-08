#!/usr/bin/env python3
"""
Web GUI for visualizing QarrayBaseClass _get_obs method results.
Uses Flask to create a web interface accessible via localhost.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

# Add the src/swarm directory to the path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.qarray_base_class import QarrayBaseClass

app = Flask(__name__)

# Global experiment instance
experiment = None
num_dots = 8

def initialize_experiment(n_dots=8):
    """Initialize the qarray experiment with specified number of dots."""
    global experiment, num_dots
    num_dots = n_dots
    experiment = QarrayBaseClass(num_dots=num_dots)
    return True

def create_plot(images_normal, images_remap, obs_voltage_min, obs_voltage_max, 
                normal_gate_voltages, normal_barrier_voltages, 
                remap_gate_voltages, remap_barrier_voltages):
    """Create matplotlib plots comparing normal and remapped observations."""
    num_channels = images_normal.shape[2]
    
    # Create figure with two rows - normal and remapped
    fig, axes = plt.subplots(2, num_channels, figsize=(4*num_channels, 8))
    
    # Handle single channel case
    if num_channels == 1:
        axes = axes.reshape(2, 1)
    
    # Plot normal observations (top row)
    for i in range(num_channels):
        im = axes[0, i].imshow(
            images_normal[:, :, i], 
            cmap='viridis',
            extent=[obs_voltage_min, obs_voltage_max, obs_voltage_min, obs_voltage_max],
            origin='lower',
            aspect='auto'
        )
        
        axes[0, i].set_xlabel('ΔVoltage (V)')
        axes[0, i].set_ylabel('ΔVoltage (V)')
        
        # Create title with voltage values
        gate1_v = normal_gate_voltages[i]
        gate2_v = normal_gate_voltages[i+1]
        barrier_v = normal_barrier_voltages[i]
        title = f'Normal - Channel {i+1} (Gates {i+1}-{i+2})\nH: {gate1_v:.3f}V, V: {gate2_v:.3f}V, B: {barrier_v:.3f}V'
        axes[0, i].set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)
    
    # Plot remapped observations (bottom row)
    for i in range(num_channels):
        im = axes[1, i].imshow(
            images_remap[:, :, i], 
            cmap='viridis',
            extent=[obs_voltage_min, obs_voltage_max, obs_voltage_min, obs_voltage_max],
            origin='lower',
            aspect='auto'
        )
        
        axes[1, i].set_xlabel('ΔVoltage (V)')
        axes[1, i].set_ylabel('ΔVoltage (V)')
        
        # Create title with remapped voltage values
        gate1_v = remap_gate_voltages[i]
        gate2_v = remap_gate_voltages[i+1]
        barrier_v = remap_barrier_voltages[i]
        title = f'Remapped - Channel {i+1} (Gates {i+1}-{i+2})\nH: {gate1_v:.3f}V, V: {gate2_v:.3f}V, B: {barrier_v:.3f}V'
        axes[1, i].set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)
    
    plt.suptitle('Charge Stability Diagrams: Normal vs Remapped', fontsize=16)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', num_dots=num_dots)

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the experiment with specified number of dots."""
    try:
        data = request.json
        n_dots = int(data.get('num_dots', 8))
        
        if n_dots < 2:
            return jsonify({'success': False, 'message': 'Number of dots must be at least 2'})
        
        initialize_experiment(n_dots)
        
        return jsonify({
            'success': True, 
            'message': f'Experiment initialized with {n_dots} dots',
            'num_dots': n_dots
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Initialization failed: {str(e)}'})

@app.route('/get_optimal_voltages', methods=['POST'])
def get_optimal_voltages():
    """Get optimal gate voltages using the model's optimal_Vg method."""
    global experiment
    
    if experiment is None:
        return jsonify({'success': False, 'message': 'Please initialize the experiment first'})
    
    try:
        # Get optimal gate voltages
        optimal_vg = experiment.model.optimal_Vg(experiment.optimal_VG_center)
        
        # The optimal_Vg returns voltages for all gates including sensor
        # We only want the plunger gate voltages (exclude the last sensor gate)
        optimal_gate_voltages = optimal_vg[:-1]  # Exclude sensor gate
        
        # Keep barrier voltages as zeros for now
        barrier_voltages = [0.0] * (num_dots - 1)
        
        return jsonify({
            'success': True,
            'message': 'Optimal voltages calculated successfully',
            'gate_voltages': optimal_gate_voltages.tolist(),
            'barrier_voltages': barrier_voltages
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to calculate optimal voltages: {str(e)}'})

@app.route('/simulate', methods=['POST'])
def simulate():
    """Run the simulation with provided voltages."""
    global experiment
    
    if experiment is None:
        return jsonify({'success': False, 'message': 'Please initialize the experiment first'})
    
    try:
        data = request.json
        gate_voltages = np.array(data['gate_voltages'], dtype=float)
        barrier_voltages = np.array(data['barrier_voltages'], dtype=float)
        
        # Validate input dimensions
        if len(gate_voltages) != num_dots:
            return jsonify({
                'success': False, 
                'message': f'Expected {num_dots} gate voltages, got {len(gate_voltages)}'
            })
        
        if len(barrier_voltages) != num_dots - 1:
            return jsonify({
                'success': False, 
                'message': f'Expected {num_dots - 1} barrier voltages, got {len(barrier_voltages)}'
            })
        
        # Get both normal and remapped observations
        obs_normal = experiment._get_obs(gate_voltages, barrier_voltages, force_remap=False)
        obs_remap = experiment._get_obs(gate_voltages, barrier_voltages, force_remap=True)
        
        images_normal = obs_normal["image"]
        images_remap = obs_remap["image"]
        
        # Extract voltage information from observations
        normal_gate_voltages = obs_normal["obs_gate_voltages"]
        normal_barrier_voltages = obs_normal["obs_barrier_voltages"]
        remap_gate_voltages = obs_remap["obs_gate_voltages"]
        remap_barrier_voltages = obs_remap["obs_barrier_voltages"]
        
        # Create comparison plot with voltage information
        plot_base64 = create_plot(
            images_normal, images_remap, 
            experiment.obs_voltage_min, experiment.obs_voltage_max,
            normal_gate_voltages, normal_barrier_voltages,
            remap_gate_voltages, remap_barrier_voltages
        )
        
        return jsonify({
            'success': True,
            'message': 'Simulation completed successfully',
            'plot': plot_base64,
            'gate_voltages': gate_voltages.tolist(),
            'barrier_voltages': barrier_voltages.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Simulation failed: {str(e)}'})


if __name__ == '__main__':
    print("Starting Qarray Web GUI...")
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=5000, debug=True)