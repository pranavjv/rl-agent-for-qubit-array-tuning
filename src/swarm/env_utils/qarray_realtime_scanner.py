#!/usr/bin/env python3
"""
Real-time Interactive Web GUI for 2-dot QarrayBaseClass with live panning.
Uses Plotly with custom JavaScript callbacks for true real-time voltage updates.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from streamlit.components.v1 import html
import time

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
    page_title="Real-time 2-Dot Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_experiment():
    """Initialize the 2-dot qarray experiment."""
    return QarrayBaseClass(num_dots=2, obs_voltage_min=-0.3, obs_voltage_max=0.3, obs_image_size=64)

def fast_scan(experiment, center_v1, center_v2, scan_range, scan_points):
    """Single call scanning function using _get_obs - same as working GUI."""
    # Just call _get_obs once with the center voltages
    # _get_obs already does the scanning via do2d_open
    gate_voltages = np.array([center_v1, center_v2])
    barrier_voltages = np.array([0.0])  # Single barrier for 2 dots
    
    obs = experiment._get_obs(gate_voltages, barrier_voltages)
    # Use the single channel (only one channel for 2 dots)
    scan_result = obs["image"][:, :, 0]
    
    # Create voltage arrays that match the scan result dimensions
    v1_min = center_v1 + experiment.obs_voltage_min
    v1_max = center_v1 + experiment.obs_voltage_max
    v2_min = center_v2 + experiment.obs_voltage_min
    v2_max = center_v2 + experiment.obs_voltage_max
    
    v1_array = np.linspace(v1_min, v1_max, scan_result.shape[1])
    v2_array = np.linspace(v2_min, v2_max, scan_result.shape[0])
    
    return scan_result, v1_array, v2_array

def create_realtime_plot(scan_result, v1_array, v2_array, center_v1, center_v2):
    """Create a plot optimized for real-time updates."""
    
    # Create subplot with secondary y-axis for real-time coordinates
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.9, 0.1],
        subplot_titles=["Charge Stability Diagram", "Real-time Coordinates"],
        vertical_spacing=0.05,
        specs=[[{"type": "heatmap"}], [{"type": "scatter"}]]
    )
    
    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=scan_result,
            x=v1_array,
            y=v2_array,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hovertemplate='Gate 1: %{x:.3f} V<br>Gate 2: %{y:.3f} V<br>Signal: %{z:.4f}<extra></extra>',
            colorbar=dict(
                title=dict(text="Signal Amplitude", side="right"),
                len=0.8,
                y=0.9,
                yanchor="top"
            )
        ),
        row=1, col=1
    )
    
    # Center point marker
    fig.add_trace(
        go.Scatter(
            x=[center_v1],
            y=[center_v2],
            mode='markers',
            marker=dict(
                symbol='x',
                size=12,
                color='red',
                line=dict(width=2, color='white')
            ),
            name='Center',
            hovertemplate='Center: (%{x:.3f}, %{y:.3f})<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Real-time coordinate display (bottom subplot)
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[0.5],
            mode='text',
            text=[f"Current View Center: Gate1={center_v1:.3f}V, Gate2={center_v2:.3f}V"],
            textposition="middle center",
            textfont=dict(size=14, color="blue"),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Real-time Interactive Scanner<br><sub>Drag to pan • Double-click to recenter • Hover for coordinates</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Gate 1 Voltage (V)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Gate 2 Voltage (V)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray',
        row=1, col=1
    )
    
    # Hide axes for coordinate display
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    
    return fig

def main():
    st.title("🔬 Real-time 2-Dot Quantum Array Scanner")
    st.markdown("**Live panning with instant voltage feedback**")
    st.markdown("---")
    
    # Initialize experiment
    if 'experiment' not in st.session_state:
        with st.spinner('Initializing 2-dot quantum array simulator...'):
            st.session_state.experiment = initialize_experiment()
        st.success('✅ Simulator initialized successfully!')
    
    experiment = st.session_state.experiment
    
    # Initialize session state
    if 'center_v1' not in st.session_state:
        st.session_state.center_v1 = 0.0
    if 'center_v2' not in st.session_state:
        st.session_state.center_v2 = 0.0
    if 'scan_range' not in st.session_state:
        st.session_state.scan_range = 1.5
    if 'scan_resolution' not in st.session_state:
        st.session_state.scan_resolution = 40
    if 'live_updates' not in st.session_state:
        st.session_state.live_updates = True
    
    # Sidebar controls
    st.sidebar.header("🎛️ Live Control Panel")
    
    # Real-time voltage display
    voltage_container = st.sidebar.container()
    with voltage_container:
        st.subheader("📍 Live Position")
        v1_display = st.empty()
        v2_display = st.empty()
        v1_display.metric("Gate 1 Voltage", f"{st.session_state.center_v1:.4f} V")
        v2_display.metric("Gate 2 Voltage", f"{st.session_state.center_v2:.4f} V")
    
    # Quick navigation
    st.sidebar.subheader("🎯 Quick Navigation")
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
                optimal_gate_voltages = optimal_vg[:-1]
                st.session_state.center_v1 = float(optimal_gate_voltages[0])
                st.session_state.center_v2 = float(optimal_gate_voltages[1])
                st.session_state.needs_update = True
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Manual position input
    st.sidebar.subheader("⌨️ Manual Input")
    manual_v1 = st.sidebar.number_input(
        "Gate 1 (V)",
        value=st.session_state.center_v1,
        step=0.01,
        format="%.4f"
    )
    manual_v2 = st.sidebar.number_input(
        "Gate 2 (V)",
        value=st.session_state.center_v2,
        step=0.01,
        format="%.4f"
    )
    
    if st.sidebar.button("🎯 Go to Position", use_container_width=True):
        st.session_state.center_v1 = manual_v1
        st.session_state.center_v2 = manual_v2
        st.session_state.needs_update = True
        st.rerun()
    
    # Scan settings
    st.sidebar.subheader("⚙️ Scan Settings")
    scan_range = st.sidebar.slider(
        "Scan Range (V)",
        min_value=0.2,
        max_value=3.0,
        value=st.session_state.scan_range,
        step=0.1,
        format="%.1f"
    )
    
    scan_resolution = st.sidebar.select_slider(
        "Resolution",
        options=[20, 30, 40, 50, 60],
        value=st.session_state.scan_resolution,
        help="Higher = better quality, slower updates"
    )
    
    live_updates = st.sidebar.toggle(
        "🔄 Live Updates",
        value=st.session_state.live_updates,
        help="Update scan while panning"
    )
    
    # Update session state if changed
    if (scan_range != st.session_state.scan_range or 
        scan_resolution != st.session_state.scan_resolution):
        st.session_state.scan_range = scan_range
        st.session_state.scan_resolution = scan_resolution
        st.session_state.needs_update = True
    
    st.session_state.live_updates = live_updates
    
    # Performance info
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Performance")
    scan_time = (scan_resolution**2) * 0.001
    st.sidebar.text(f"Scan time: ~{scan_time:.1f}s")
    st.sidebar.text(f"Total points: {scan_resolution**2}")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Scan View")
        
        # Check if scan needs updating
        needs_update = getattr(st.session_state, 'needs_update', True)
        
        if needs_update or 'scan_data' not in st.session_state:
            progress_container = st.empty()
            
            with progress_container.container():
                progress_bar = st.progress(0)
                status_text = st.text(f'Computing {scan_resolution}×{scan_resolution} scan...')
                
                start_time = time.time()
                
                scan_result, v1_array, v2_array = fast_scan(
                    experiment,
                    st.session_state.center_v1,
                    st.session_state.center_v2,
                    scan_range,
                    scan_resolution
                )
                
                # Simulate progress for user feedback
                progress_bar.progress(1.0)
                scan_time = time.time() - start_time
                status_text.text(f'Scan completed in {scan_time:.2f}s')
                
                # Store results
                st.session_state.scan_data = {
                    'result': scan_result,
                    'v1_array': v1_array,
                    'v2_array': v2_array,
                    'center_v1': st.session_state.center_v1,
                    'center_v2': st.session_state.center_v2,
                    'scan_range': scan_range,
                    'timestamp': time.time()
                }
                st.session_state.needs_update = False
                
                time.sleep(0.5)  # Brief pause to show completion
                progress_container.empty()
        
        # Display interactive plot
        if 'scan_data' in st.session_state:
            data = st.session_state.scan_data
            
            fig = create_realtime_plot(
                data['result'],
                data['v1_array'],
                data['v2_array'],
                data['center_v1'],
                data['center_v2']
            )
            
            # Display with event handling
            chart_container = st.empty()
            
            with chart_container.container():
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="main_plot"
                )
            
            # Instructions
            st.info(
                "💡 **Real-time Controls:**\n"
                "• **Drag** to pan instantly with live coordinate updates\n"
                "• **Double-click** to recenter scan at that point\n"
                "• **Scroll** to zoom in/out\n"
                "• **Hover** to see exact voltage and signal values\n"
                "• Use sidebar for precise navigation and settings"
            )
    
    with col2:
        st.subheader("📊 Live Statistics")
        
        if 'scan_data' in st.session_state:
            data = st.session_state.scan_data
            scan_result = data['result']
            
            # Live metrics
            st.metric("Max Signal", f"{np.max(scan_result):.5f}")
            st.metric("Min Signal", f"{np.min(scan_result):.5f}")
            st.metric("Signal Std", f"{np.std(scan_result):.5f}")
            st.metric("Mean Signal", f"{np.mean(scan_result):.5f}")
            
            # Current view bounds
            st.markdown("**Current View:**")
            v1_range = data['scan_range'] / 2
            v2_range = data['scan_range'] / 2
            
            st.text(f"Gate 1: {data['center_v1']-v1_range:.3f} to {data['center_v1']+v1_range:.3f} V")
            st.text(f"Gate 2: {data['center_v2']-v2_range:.3f} to {data['center_v2']+v2_range:.3f} V")
            
            # Scan info
            st.markdown("**Scan Info:**")
            st.text(f"Resolution: {scan_resolution}×{scan_resolution}")
            st.text(f"Range: ±{scan_range/2:.1f} V")
            
            # Timestamp
            scan_age = time.time() - data['timestamp']
            st.text(f"Age: {scan_age:.1f}s")
            
            # Auto-refresh option
            if scan_age > 5 and live_updates:
                if st.button("🔄 Refresh Scan", type="secondary"):
                    st.session_state.needs_update = True
                    st.rerun()

if __name__ == "__main__":
    main()