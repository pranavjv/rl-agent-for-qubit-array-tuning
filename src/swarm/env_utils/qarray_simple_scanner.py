#!/usr/bin/env python3
"""
Simple Interactive Web GUI for 2-dot QarrayBaseClass - just like qarray_web_gui.py but with panning.
Shows a single scan that can be dragged to recenter at new coordinates.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add the environment directory to the path
current_dir = os.path.dirname(__file__)
# Go up one level from env_utils to swarm, then into environment
env_dir = os.path.join(current_dir, '..', 'environment')
sys.path.insert(0, env_dir)

try:
    from qarray_base_class import QarrayBaseClass
except ImportError:
    # Try alternative import paths
    try:
        # Try relative import from parent directory
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, os.path.join(parent_dir, 'environment'))
        from qarray_base_class import QarrayBaseClass
    except ImportError:
        try:
            # Try absolute import
            from environment.qarray_base_class import QarrayBaseClass
        except ImportError as e:
            st.error(f"Could not import QarrayBaseClass. Please ensure the environment module is available. Error: {e}")
            st.stop()

# Set page configuration
st.set_page_config(
    page_title="Simple 2-Dot Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_experiment():
    """Initialize the 2-dot qarray experiment - same as working web GUI."""
    try:
        return QarrayBaseClass(num_dots=2, remap=False)
    except Exception as e:
        st.error(f"Failed to initialize experiment: {e}")
        st.stop()

def get_single_scan(experiment, gate1_voltage, gate2_voltage):
    """
    Get a single scan exactly like the working web GUI does.
    Just call _get_obs once with the center voltages.
    """
    # Same as qarray_web_gui.py - just call _get_obs once
    gate_voltages = np.array([gate1_voltage, gate2_voltage])
    barrier_voltages = np.array([0.0])  # Single barrier for 2 dots
    
    # Get observation - this gives us the scan
    obs = experiment._get_obs(gate_voltages, barrier_voltages, force_remap=False)
    
    # Extract the image - for 2 dots there's only 1 channel
    scan_image = obs["image"][:, :, 0]
    
    # Create voltage arrays that match the scan dimensions
    # The scan covers from gate_voltage + obs_voltage_min to gate_voltage + obs_voltage_max
    v1_min = gate1_voltage + experiment.obs_voltage_min
    v1_max = gate1_voltage + experiment.obs_voltage_max
    v2_min = gate2_voltage + experiment.obs_voltage_min
    v2_max = gate2_voltage + experiment.obs_voltage_max
    
    v1_array = np.linspace(v1_min, v1_max, scan_image.shape[1])
    v2_array = np.linspace(v2_min, v2_max, scan_image.shape[0])
    
    return scan_image, v1_array, v2_array

def create_interactive_plot(scan_image, v1_array, v2_array, center_v1, center_v2):
    """Create a simple interactive plot with drag capability."""
    
    fig = go.Figure()
    
    # Add the heatmap
    fig.add_trace(go.Heatmap(
        z=scan_image,
        x=v1_array,
        y=v2_array,
        colorscale='Viridis',
        showscale=True,
        hoverongaps=False,
        hovertemplate='Gate 1: %{x:.3f} V<br>Gate 2: %{y:.3f} V<br>Signal: %{z:.4f}<extra></extra>',
        colorbar=dict(
            title=dict(text="Signal Amplitude", side="right")
        )
    ))
    
    # Add center point marker
    fig.add_trace(go.Scatter(
        x=[center_v1],
        y=[center_v2],
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color='red',
            line=dict(width=3, color='white')
        ),
        name='Center Point',
        hovertemplate='Center: (%{x:.3f}, %{y:.3f})<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': '2-Dot Charge Stability Diagram<br><sub>Drag to pan • Click to recenter</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Gate 1 Voltage (V)',
        yaxis_title='Gate 2 Voltage (V)',
        width=800,
        height=600,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    return fig

def main():
    st.title("🔬 Simple 2-Dot Scanner")
    st.markdown("**Interactive charge stability diagram with panning**")
    st.markdown("---")
    
    # Initialize experiment
    if 'experiment' not in st.session_state:
        with st.spinner('Initializing 2-dot quantum array simulator...'):
            try:
                st.session_state.experiment = initialize_experiment()
                st.success('✅ Simulator initialized successfully!')
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.stop()
    
    experiment = st.session_state.experiment
    
    # Initialize session state
    if 'center_v1' not in st.session_state:
        st.session_state.center_v1 = 0.0
    if 'center_v2' not in st.session_state:
        st.session_state.center_v2 = 0.0
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Real-time voltage display
    st.sidebar.subheader("📍 Current Center")
    st.sidebar.metric("Gate 1 Voltage", f"{st.session_state.center_v1:.3f} V")
    st.sidebar.metric("Gate 2 Voltage", f"{st.session_state.center_v2:.3f} V")
    
    # Manual voltage inputs
    st.sidebar.subheader("⌨️ Set Center Point")
    new_v1 = st.sidebar.number_input(
        "Gate 1 (V)",
        value=st.session_state.center_v1,
        step=0.1,
        format="%.3f"
    )
    
    new_v2 = st.sidebar.number_input(
        "Gate 2 (V)",
        value=st.session_state.center_v2,
        step=0.1,
        format="%.3f"
    )
    
    # Update button
    if st.sidebar.button("🎯 Update Center", type="primary", use_container_width=True):
        st.session_state.center_v1 = new_v1
        st.session_state.center_v2 = new_v2
        st.session_state.needs_update = True
        st.rerun()
    
    # Quick navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🏠 Origin", use_container_width=True):
            st.session_state.center_v1 = 0.0
            st.session_state.center_v2 = 0.0
            st.session_state.needs_update = True
            st.rerun()
    
    with col2:
        if st.button("🎯 Optimal", use_container_width=True):
            try:
                optimal_vg = experiment.model.optimal_Vg(experiment.optimal_VG_center)
                optimal_gate_voltages = optimal_vg[:-1]  # Exclude sensor gate
                st.session_state.center_v1 = float(optimal_gate_voltages[0])
                st.session_state.center_v2 = float(optimal_gate_voltages[1])
                st.session_state.needs_update = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Scan")
        
        # Check if we need to update the scan
        needs_update = getattr(st.session_state, 'needs_update', True)
        
        if needs_update or 'scan_data' not in st.session_state:
            with st.spinner('Computing scan...'):
                scan_image, v1_array, v2_array = get_single_scan(
                    experiment,
                    st.session_state.center_v1,
                    st.session_state.center_v2
                )
                
                # Store results
                st.session_state.scan_data = {
                    'image': scan_image,
                    'v1_array': v1_array,
                    'v2_array': v2_array,
                    'center_v1': st.session_state.center_v1,
                    'center_v2': st.session_state.center_v2
                }
                st.session_state.needs_update = False
        
        # Display the plot
        if 'scan_data' in st.session_state:
            data = st.session_state.scan_data
            
            fig = create_interactive_plot(
                data['image'],
                data['v1_array'],
                data['v2_array'],
                data['center_v1'],
                data['center_v2']
            )
            
            # Display with event handling
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="main_plot"
            )
            
            # Instructions
            st.info(
                "💡 **How to interact:**\n"
                "• **Drag** to pan around the voltage space\n"
                "• **Scroll** to zoom in/out\n" 
                "• **Click 'Update Center'** to recalculate scan at new position\n"
                "• **Use sidebar controls** for precise positioning"
            )
    
    with col2:
        st.subheader("📊 Scan Info")
        
        if 'scan_data' in st.session_state:
            data = st.session_state.scan_data
            scan_image = data['image']
            
            # Statistics
            st.metric("Max Signal", f"{np.max(scan_image):.4f}")
            st.metric("Min Signal", f"{np.min(scan_image):.4f}")
            st.metric("Signal Range", f"{np.ptp(scan_image):.4f}")
            st.metric("Mean Signal", f"{np.mean(scan_image):.4f}")
            
            # Current view bounds
            st.markdown("**Current Scan Range:**")
            v1_min, v1_max = data['v1_array'][0], data['v1_array'][-1]
            v2_min, v2_max = data['v2_array'][0], data['v2_array'][-1]
            
            st.text(f"Gate 1: [{v1_min:.3f}, {v1_max:.3f}] V")
            st.text(f"Gate 2: [{v2_min:.3f}, {v2_max:.3f}] V")
            
            # Scan details
            st.markdown("**Scan Details:**")
            st.text(f"Image size: {scan_image.shape[0]}×{scan_image.shape[1]}")
            st.text(f"Voltage range: {experiment.obs_voltage_max - experiment.obs_voltage_min:.1f} V")
        
        # Model information
        with st.expander("🔬 Model Info"):
            st.text(f"Dots: {experiment.num_dots}")
            st.text(f"Gates: {experiment.num_gate_voltages}")
            st.text(f"Image size: {experiment.obs_image_size}")
            st.text(f"Obs range: {experiment.obs_voltage_min} to {experiment.obs_voltage_max} V")

if __name__ == "__main__":
    main()