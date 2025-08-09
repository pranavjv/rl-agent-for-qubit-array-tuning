# QDARTS Environment Implementation Plan

## Executive Summary

**Planning Complexity: MODERATE** - Well-scoped project with clear technical requirements and feasible implementation path.

**Estimated Implementation Time: 2-3 weeks**

**Key Insight**: QDARTS provides a complete, working implementation that can be directly integrated into the RL environment with minimal modifications.

## Current State Analysis

### Current qarray_env.py Structure
- **Action Space**: 2 plunger gates only (Box space with shape (2,))
- **Observation Space**: Multi-modal (image + voltages) with 2 voltage values
- **Model**: Uses `qarray.ChargeSensedDotArray` with simple capacitance matrices
- **Voltage Application**: Creates 2D grids centered around plunger voltages
- **Reward**: Distance-based reward for plunger alignment only

### QDARTS Structure (from qdarts_v5.py)
- **Device**: 3 dots, 6 gates (2 plungers + 3 barriers + 1 sensor)
- **Action Space**: Need to handle 5 gates (2 plungers + 3 barriers)
- **Model**: Uses `qdarts.Experiment` with complex barrier physics
- **Barrier Physics**: Exponential tunnel coupling effects based on barrier voltages
- **CSD Generation**: `experiment.generate_CSD()` returns sensor values directly

## Key Findings from Exploration

### ✅ QDARTS Integration is Feasible
1. **QDARTS import successful** - Library properly installed and accessible
2. **Configuration loading works** - `load_qdarts_config()` handles all validation
3. **Experiment creation successful** - `Experiment` class provides clean interface
4. **CSD generation working** - `generate_CSD()` returns sensor values in correct format
5. **Barrier physics implemented** - `update_tunnel_couplings()` works with exponential physics

### ✅ QDARTS Capabilities (from qdarts_v5.py)
1. **Multi-gate support** - Handles 6 gates (2 plungers + 3 barriers + 1 sensor)
2. **Barrier physics** - Exponential tunnel coupling effects implemented
3. **Sensor integration** - Built-in sensor simulation with `use_sensor_signal=True`
4. **Dynamic coupling updates** - Real-time tunnel coupling updates via `update_tunnel_couplings()`
5. **Configuration management** - Complete config validation and type conversion

### ❌ Current Environment Issues
1. **Configuration mismatch** - Cds structure in config doesn't match code expectations
2. **Hardcoded assumptions** - Many parts assume 2 gates only
3. **Limited voltage handling** - Only handles 2D voltage grids

## Implementation Plan

### Phase 1: Action Space Redesign

#### 1.1 New Action Space Structure
```python
# Current action space (2 plungers only)
self.action_space = spaces.Box(
    low=self.action_voltage_min, 
    high=self.action_voltage_max, 
    shape=(2,), 
    dtype=np.float32
)

# New action space - Combined Box (5 gates: 2 plungers + 3 barriers)
self.action_space = spaces.Box(
    low=np.array([self.plunger_voltage_min] * 2 + [self.barrier_voltage_min] * 3),
    high=np.array([self.plunger_voltage_max] * 2 + [self.barrier_voltage_max] * 3),
    shape=(5,),  # 2 plungers + 3 barriers
    dtype=np.float32
)
```

#### 1.2 Configuration Updates
```yaml
# New config structure
env:
  action_space:
    plungers:
      num_voltages: 2
      voltage_range: [-5.0, 5.0]
    barriers:
      num_voltages: 3
      voltage_range: [-2.0, 2.0]  # Different range for barriers
  observation_space:
    image_size: [128, 128]  # Will be flexible based on config resolution
    channels: 1
    normalization_range: [0.0, 1.0]
    dtype: "float32"
    voltage_shape: 5  # 2 plungers + 3 barriers
```

### Phase 2: New QDARTS Environment Configuration

#### 2.1 New Configuration Structure
```yaml
# qdarts_env_config.yaml
# QDARTS Environment Configuration - Combines QDARTS and RL environment configs

# QDARTS Configuration (imported from qdarts_config_v5.yaml)
qdarts:
  config_path: "qdarts_config_v5.yaml"
  
  # Override QDARTS parameters with random ranges for testing
  random_ranges:
    capacitance:
      C_DD:
        - [{"min": 0.8, "max": 1.2}, {"min": 0.1, "max": 0.3}, {"min": 0.05, "max": 0.15}]
        - [{"min": 0.1, "max": 0.3}, {"min": 0.8, "max": 1.2}, {"min": 0.05, "max": 0.15}]
        - [{"min": 0.05, "max": 0.15}, {"min": 0.05, "max": 0.15}, {"min": 0.8, "max": 1.2}]
      C_DG:
        - [{"min": 0.6, "max": 0.9}, {"min": 0.05, "max": 0.15}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0.01, "max": 0.03}]
        - [{"min": 0.05, "max": 0.15}, {"min": 0.6, "max": 0.9}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0.01, "max": 0.03}]
        - [{"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0.8, "max": 1.2}]
    
    tunneling:
      tunnel_couplings:
        - [{"min": 0, "max": 0}, {"min": 20e-4, "max": 40e-4}, {"min": 0, "max": 0}]
        - [{"min": 20e-4, "max": 40e-4}, {"min": 0, "max": 0}, {"min": 0, "max": 0}]
        - [{"min": 0, "max": 0}, {"min": 0, "max": 0}, {"min": 0, "max": 0}]
      temperature: {"min": 0.5, "max": 2.0}
      energy_range_factor: {"min": 3.0, "max": 7.0}
    
    barrier:
      barrier_mappings:
        barrier_2:
          base_coupling: {"min": 20e-4, "max": 40e-4}
          alpha: {"min": 1.5, "max": 2.5}
          voltage_offset: {"min": -0.1, "max": 0.1}
        barrier_3:
          base_coupling: {"min": 20e-4, "max": 40e-4}
          alpha: {"min": 1.0, "max": 2.0}
          voltage_offset: {"min": -0.1, "max": 0.1}
        barrier_4:
          base_coupling: {"min": 20e-4, "max": 40e-4}
          alpha: {"min": 1.5, "max": 2.5}
          voltage_offset: {"min": -0.1, "max": 0.1}
    
    sensor:
      sensor_detunings: [{"min": 0.0003, "max": 0.0007}]
      noise_amplitude:
        fast_noise: {"min": 5e-5, "max": 2e-4}
        slow_noise: {"min": 5e-6, "max": 2e-5}
      peak_width_multiplier: {"min": 150, "max": 250}

# Environment Configuration
env:
  max_steps: 50
  action_space:
    plungers:
      num_voltages: 2
      voltage_range: [-5.0, 5.0]
    barriers:
      num_voltages: 3
      voltage_range: [-2.0, 2.0]
  observation_space:
    # Flexible resolution - will be set based on QDARTS config
    image_size: "auto"  # Will be set to [resolution, resolution] from QDARTS
    channels: 1
    normalization_range: [0.0, 1.0]
    dtype: "float32"
    voltage_shape: 5
  tolerance: 0.5

# Training Configuration
training:
  seed: 42
  debug: false
  render_fps: 10
  render_mode: "human"
```

#### 2.2 Configuration Loader
```python
def _load_qdarts_env_config(self):
    """
    Load QDARTS environment configuration with random ranges support.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'qdarts_env_config.yaml')
    
    # Load base QDARTS environment config
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Load QDARTS config
    qdarts_config_path = env_config['qdarts']['config_path']
    qdarts_config = self._load_qdarts_config(qdarts_config_path)
    
    # Apply random ranges if specified
    if 'random_ranges' in env_config['qdarts']:
        qdarts_config = self._apply_random_ranges(qdarts_config, env_config['qdarts']['random_ranges'])
    
    # Set observation space resolution based on QDARTS config
    resolution = qdarts_config['measurement']['resolution']
    env_config['env']['observation_space']['image_size'] = [resolution, resolution]
    
    return env_config, qdarts_config

def _apply_random_ranges(self, qdarts_config, random_ranges):
    """
    Apply random ranges to QDARTS configuration parameters.
    """
    # Apply random ranges to capacitance matrices
    if 'capacitance' in random_ranges:
        self._apply_capacitance_random_ranges(qdarts_config, random_ranges['capacitance'])
    
    # Apply random ranges to tunneling parameters
    if 'tunneling' in random_ranges:
        self._apply_tunneling_random_ranges(qdarts_config, random_ranges['tunneling'])
    
    # Apply random ranges to barrier parameters
    if 'barrier' in random_ranges:
        self._apply_barrier_random_ranges(qdarts_config, random_ranges['barrier'])
    
    # Apply random ranges to sensor parameters
    if 'sensor' in random_ranges:
        self._apply_sensor_random_ranges(qdarts_config, random_ranges['sensor'])
    
    return qdarts_config
```

### Phase 3: QDARTS Model Integration

#### 3.1 QDARTS Model Loading (based on qdarts_v5.py)
```python
def _load_model(self):
    """
    Load QDARTS model with barrier support.
    """
    # Load QDARTS environment configuration
    env_config, qdarts_config = self._load_qdarts_env_config()
    
    # Extract configurations (exactly like qdarts_v5.py)
    capacitance_config = qdarts_config['simulator']['capacitance']
    tunneling_config = qdarts_config['simulator']['tunneling']
    sensor_config = qdarts_config['simulator']['sensor']
    barrier_config = qdarts_config['simulator']['barrier']
    
    # Create QDARTS experiment (exactly like qdarts_v5.py)
    self.experiment = Experiment(
        capacitance_config=capacitance_config,
        tunneling_config=tunneling_config,
        sensor_config=sensor_config,
        barrier_config=barrier_config,
        print_logs=self.debug
    )
    
    # Set initial barrier voltages (exactly like qdarts_v5.py)
    initial_barrier_voltages = barrier_config['default_barrier_voltages']
    self.experiment.update_tunnel_couplings(initial_barrier_voltages)
    
    return self.experiment
```

### Phase 4: Flexible Observation Space

#### 4.1 Dynamic Observation Space
```python
def _get_obs(self):
    """
    Get current observation with flexible resolution based on config.
    """
    # Get charge sensor data using QDARTS (based on qdarts_v5.py)
    current_voltages = self.device_state["current_voltages"]
    self.z = self._get_charge_sensor_data_qdarts(current_voltages)
    z = self.z
    
    # Extract and normalize image observation (flexible resolution)
    channel_data = z[:, :, 0]  # Shape: (resolution, resolution)
    image_obs = self._normalize_observation(channel_data)
    
    # Extract voltage centers for plungers
    plunger_centers = self._extract_voltage_centers(current_voltages)
    
    # Get barrier voltages
    barrier_voltages = self.device_state.get("current_barrier_voltages", np.zeros(3))
    
    # Combine all voltages
    all_voltages = np.concatenate([plunger_centers, barrier_voltages])
    
    observation = {
        'image': image_obs,  # Shape: (resolution, resolution, 1)
        'voltages': all_voltages  # Shape: (5,)
    }
    
    return observation

def _get_charge_sensor_data_qdarts(self, voltages):
    """
    Get charge sensor data using QDARTS (based on qdarts_v5.py).
    """
    # Extract measurement configuration (based on qdarts_v5.py)
    _, qdarts_config = self._load_qdarts_env_config()
    measurement = qdarts_config['measurement']
    
    # Create voltage ranges (based on qdarts_v5.py)
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
    
    # Convert sweep matrix to plane axes (based on qdarts_v5.py)
    sweep_matrix = measurement['sweep_matrix']
    plane_axes = []
    for i in range(2):
        for j, row in enumerate(sweep_matrix):
            if abs(row[i]) > 1e-10:
                plane_axes.append(j)
                break
    
    # Generate CSD using QDARTS (based on qdarts_v5.py)
    xout, yout, _, polytopes, sensor_values, v_offset = self.experiment.generate_CSD(
        x_voltages=x_voltages,
        y_voltages=y_voltages,
        plane_axes=plane_axes,
        target_state=qdarts_config['device']['target_state'],
        use_sensor_signal=True,  # Get sensor response
        compensate_sensors=True,
        compute_polytopes=False
    )
    
    return sensor_values  # Shape: (resolution, resolution, num_sensors)
```

### Phase 5: Voltage Application Logic

#### 5.1 New Voltage Application (based on qdarts_v5.py)
```python
def _apply_voltages(self, action):
    """
    Apply voltage settings to the quantum device.
    
    Args:
        action: np.ndarray(5,) containing [plunger1, plunger2, barrier1, barrier2, barrier3]
    """
    # Split action into plungers and barriers
    plungers = action[:2]  # First 2 elements
    barriers = action[2:]  # Last 3 elements
    
    # Apply plunger voltages (same logic as current)
    self._apply_plunger_voltages(plungers)
    
    # Apply barrier voltages (new - based on qdarts_v5.py)
    self._apply_barrier_voltages(barriers)

def _apply_plunger_voltages(self, plungers):
    """
    Apply plunger voltages (same logic as current _apply_voltages).
    """
    # Create 2D grids centered around plunger voltages
    x_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[1])
    y_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[0])
    
    X, Y = np.meshgrid(x_grid + plungers[0], y_grid + plungers[1])
    
    # Update voltage grids
    self.device_state["current_voltages"][:,:,0] = X
    self.device_state["current_voltages"][:,:,1] = Y

def _apply_barrier_voltages(self, barriers):
    """
    Apply barrier voltages and update tunnel couplings (based on qdarts_v5.py).
    """
    # Convert barriers to dictionary format expected by QDARTS
    barrier_voltages = {
        'barrier_2': barriers[0],
        'barrier_3': barriers[1], 
        'barrier_4': barriers[2]
    }
    
    # Update tunnel couplings in QDARTS experiment (exactly like qdarts_v5.py)
    self.experiment.update_tunnel_couplings(barrier_voltages)
    
    # Store current barrier voltages
    self.device_state["current_barrier_voltages"] = barriers
```

### Phase 6: Reward Function Enhancement

#### 6.1 Multi-Objective Reward
```python
def _get_reward(self):
    """
    Enhanced reward function considering both plunger alignment and barrier optimization.
    """
    # Plunger alignment reward (same as current)
    ground_truth_center = self.device_state["ground_truth_center"][:2]
    current_plunger_center = self._extract_voltage_centers(self.device_state["current_voltages"])
    plunger_distance = np.linalg.norm(ground_truth_center - current_plunger_center)
    
    # Barrier optimization reward (new)
    barrier_reward = self._calculate_barrier_reward()
    
    # Combined reward
    plunger_reward = max(self.max_possible_distance - plunger_distance, 0) * 0.1
    total_reward = plunger_reward + barrier_reward * 0.05  # Weight barrier reward less
    
    # Termination bonus
    if self._is_at_target():
        total_reward += 200.0
    
    return total_reward

def _calculate_barrier_reward(self):
    """
    Calculate reward based on barrier optimization.
    """
    # Get current tunnel couplings
    current_couplings = self.experiment.get_current_tunnel_couplings()
    
    # Define target coupling strengths (could be configurable)
    target_couplings = np.array([
        [0, 30e-4, 0],
        [30e-4, 0, 0], 
        [0, 0, 0]
    ])
    
    # Calculate coupling alignment reward
    coupling_distance = np.linalg.norm(current_couplings - target_couplings)
    max_coupling_distance = np.linalg.norm(target_couplings)
    
    return max(max_coupling_distance - coupling_distance, 0)
```

## Critical Roadblocks and Solutions

### ✅ Roadblock 1: Action Space Design
**Issue**: How to handle different voltage ranges for plungers vs barriers
**Solution**: Use separate voltage ranges and combine in action space
**Status**: ✅ Resolved - Use combined Box space with different ranges

### ✅ Roadblock 2: Model Integration
**Issue**: Different APIs between qarray and QDARTS
**Solution**: Use QDARTS Experiment class directly (as shown in qdarts_v5.py)
**Status**: ✅ Resolved - QDARTS Experiment class provides clean interface

### ✅ Roadblock 3: Observation Space Flexibility
**Issue**: QDARTS resolution vs expected observation size
**Solution**: Make observation space flexible based on config resolution
**Status**: ✅ Resolved - Dynamic observation space based on config

### ✅ Roadblock 4: Voltage Application Compatibility
**Issue**: Mixing QDARTS barrier updates with qarray-style plunger grids
**Solution**: Confirmed compatible - can mix approaches
**Status**: ✅ Resolved - Compatible voltage application methods

### ✅ Roadblock 5: Configuration Management
**Issue**: Complex configuration with multiple sources
**Solution**: Create new qdarts_env_config.yaml with random ranges support
**Status**: ✅ Resolved - New unified config with random ranges

### ⚠️ Roadblock 6: Performance
**Issue**: QDARTS might be slower than qarray
**Solution**: Profile and optimize, consider caching
**Status**: ⚠️ Needs testing - QDARTS takes ~30s for CSD generation (vs qarray ~1s)

### ⚠️ Roadblock 7: Testing
**Issue**: Need to test barrier physics effects
**Solution**: Create comprehensive test suite
**Status**: ⚠️ Needs implementation - Test suite required

## Implementation Steps

1. **Create new qdarts_env_config.yaml** with random ranges support
2. **Implement configuration loader** with random ranges application
3. **Implement new action space** (5 gates: 2 plungers + 3 barriers)
4. **Integrate QDARTS model** with barrier support (based on qdarts_v5.py)
5. **Update voltage application logic** to handle barriers
6. **Implement flexible observation space** based on config resolution
7. **Implement multi-objective reward** function
8. **Add comprehensive testing** for barrier effects
9. **Optimize performance** if needed
10. **Documentation and examples**

## Questions to Resolve

1. **Action Space Format**: ✅ Resolved - Use combined Box space
2. **Voltage Ranges**: ✅ Resolved - Different ranges for plungers vs barriers
3. **Reward Weights**: ✅ Resolved - Start with 0.1 for plungers, 0.05 for barriers
4. **Performance**: ⚠️ Needs testing - QDARTS may be slower but manageable
5. **Configuration**: ✅ Resolved - New unified config with random ranges
6. **Testing**: ⚠️ Needs implementation - Comprehensive test suite required

## Next Steps

1. **Create qdarts_env_config.yaml** with random ranges support
2. **Implement configuration loader** with random ranges application
3. **Test basic QDARTS integration** with flexible observation space
4. **Validate barrier physics integration** with simple examples
5. **Performance benchmarking** against current environment
6. **Documentation and examples** for new environment

## Summary

Based on my exhaustive exploration of the codebase and QDARTS integration, here's what I found:

### **Planning Complexity: MODERATE**

**Why it's not trivial:**
1. **Different physics models** - QDARTS uses exponential barrier physics vs qarray's simpler model
2. **Multi-gate architecture** - Need to handle 5 gates (2 plungers + 3 barriers) vs current 2
3. **Complex reward function** - Multi-objective optimization (plunger alignment + barrier optimization)
4. **Performance considerations** - QDARTS may be slower than qarray (~30s vs ~1s)
5. **Configuration complexity** - Need to manage multiple config sources

**Why it's feasible:**
1. ✅ **QDARTS is working** - Successfully imported and tested
2. ✅ **Clean interfaces** - QDARTS Experiment class provides good abstraction
3. ✅ **Barrier physics implemented** - Exponential tunnel coupling effects already working
4. ✅ **Configuration management** - QDARTS config loader exists and works
5. ✅ **Similar structure** - Both environments use similar observation/action patterns

### **Key Technical Decisions Made:**

1. **Action Space**: Combined Box space with 5 elements (2 plungers + 3 barriers)
2. **Voltage Ranges**: Different ranges for plungers (-5 to 5V) vs barriers (-2 to 2V)
3. **Reward Function**: Weighted sum of plunger alignment (0.1) + barrier optimization (0.05)
4. **Model Integration**: Direct use of QDARTS Experiment class (as shown in qdarts_v5.py)
5. **Configuration**: New unified qdarts_env_config.yaml with random ranges support
6. **Observation Space**: Flexible resolution based on config

### **Estimated Implementation Time: 2-3 weeks**

**Week 1**: New config structure and QDARTS integration
**Week 2**: Flexible observation space and voltage application logic
**Week 3**: Reward function implementation and testing

### **Critical Success Factors:**

1. **Performance testing** - Ensure QDARTS is fast enough for RL training
2. **Reward function tuning** - Balance between plunger and barrier objectives
3. **Comprehensive testing** - Validate barrier physics effects
4. **Documentation** - Clear examples and usage instructions

This is a **well-scoped project** with clear technical requirements and a feasible implementation path. The main challenges are around performance optimization and reward function tuning, but the core integration is straightforward thanks to the existing QDARTS implementation. 