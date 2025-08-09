# QDARTS Environment Implementation

## Overview

The QDARTS environment (`qdarts_env.py`) is a comprehensive implementation of a quantum device environment using the QDARTS physics simulator. This environment extends the original `qarray_env.py` to support:

- **5 gates**: 2 plungers + 3 barriers
- **QDARTS physics model**: Exponential barrier effects and realistic quantum dot physics
- **Flexible observation space**: Configurable resolution based on QDARTS config
- **Multi-objective reward function**: Combines plunger alignment and barrier optimization
- **Random ranges support**: Configurable parameter randomization for training diversity

## Key Features

### 1. Action Space
- **Combined Box space**: 5-dimensional action space
  - First 2 dimensions: Plunger voltages (range: [-5.0, 5.0]V)
  - Last 3 dimensions: Barrier voltages (range: [-2.0, 2.0]V)
- **Action format**: `[plunger1, plunger2, barrier1, barrier2, barrier3]`

### 2. Observation Space
- **Multi-modal observation**: Dictionary with two components
  - `image`: Charge sensor data (shape: `[resolution, resolution, 1]`, dtype: `uint8`)
  - `voltages`: Current voltage values (shape: `[5]`, dtype: `float32`)
- **Flexible resolution**: Automatically set based on QDARTS config resolution

### 3. Physics Model
- **QDARTS integration**: Uses `qdarts.experiment_with_barriers.Experiment`
- **Barrier physics**: Exponential tunnel coupling effects based on barrier voltages
- **Sensor integration**: Built-in sensor simulation with noise models
- **Dynamic coupling updates**: Real-time tunnel coupling updates via `update_tunnel_couplings()`

### 4. Reward Function
- **Multi-objective**: Combines plunger alignment and barrier optimization
- **Plunger reward**: Distance-based reward for plunger alignment (weight: 0.1)
- **Barrier reward**: Coupling alignment reward for barrier optimization (weight: 0.05)
- **Termination bonus**: +200.0 reward when target is reached

## Configuration

### Environment Configuration (`qdarts_env_config.yaml`)

```yaml
# QDARTS Configuration
qdarts:
  config_path: "qdarts_config_v5.yaml"
  random_ranges:
    capacitance:
      C_DD: # Dot-to-dot capacitance ranges
      C_DG: # Dot-to-gate capacitance ranges
    tunneling:
      tunnel_couplings: # Tunnel coupling ranges
      temperature: # Temperature range
      energy_range_factor: # Energy range factor
    barrier:
      barrier_mappings: # Barrier parameter ranges
    sensor:
      sensor_detunings: # Sensor detuning ranges
      noise_amplitude: # Noise amplitude ranges

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
    image_size: "auto"  # Set from QDARTS config
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

### QDARTS Configuration (`qdarts_config_v5.yaml`)

The environment uses the existing QDARTS configuration with support for:
- **Device topology**: 3 dots, 6 gates (2 plungers + 3 barriers + 1 sensor)
- **Capacitance matrices**: Dot-to-dot and dot-to-gate capacitance matrices
- **Barrier physics**: Exponential tunnel coupling effects
- **Sensor model**: Noise models and sensor detunings
- **Measurement config**: Voltage ranges, resolution, and sweep matrices

## Usage

### Basic Usage

```python
from qdarts_env import QDARTSEnv

# Create environment
env = QDARTSEnv(config_path='qdarts_env_config.yaml')

# Reset environment
observation, info = env.reset()

# Take action
action = env.action_space.sample()  # Random action
observation, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

### Training Integration

```python
import gymnasium as gym
from qdarts_env import QDARTSEnv

# Register environment (if needed)
gym.register(
    id='QDARTS-v0',
    entry_point='qdarts_env:QDARTSEnv',
    max_episode_steps=50
)

# Use in training
env = gym.make('QDARTS-v0')
```

## Implementation Details

### Core Components

1. **Configuration Loading** (`_load_qdarts_env_config`)
   - Loads environment and QDARTS configurations
   - Applies random ranges for parameter diversity
   - Sets observation space resolution dynamically

2. **Model Integration** (`_load_model`)
   - Creates QDARTS Experiment with barrier support
   - Initializes tunnel couplings and sensor models
   - Sets up barrier voltage mappings

3. **Observation Generation** (`_get_obs`)
   - Generates charge sensor data using QDARTS
   - Normalizes image observations
   - Combines plunger and barrier voltages

4. **Voltage Application** (`_apply_voltages`)
   - Applies plunger voltages to voltage grids
   - Updates barrier voltages and tunnel couplings
   - Maintains device state consistency

5. **Reward Calculation** (`_get_reward`)
   - Calculates plunger alignment reward
   - Computes barrier optimization reward
   - Combines rewards with appropriate weights

### Key Methods

- `reset()`: Initialize environment with random center and QDARTS model
- `step(action)`: Apply action and return observation, reward, termination status
- `render()`: Render environment state as RGB array
- `close()`: Clean up resources

## Testing

### Test Script

Run the test script to verify implementation:

```bash
cd DreamerV3
python test_qdarts_env.py
```

### Expected Output

```
Testing QDARTS Environment...
✅ Environment created successfully
✅ Environment reset successfully
   Observation keys: ['image', 'voltages']
   Image shape: (250, 250, 1)
   Voltage shape: (5,)
✅ Action space: Box([-5. -5. -2. -2. -2.], [5. 5. 2. 2. 2.], (5,), float32)
✅ Environment step successful
✅ Environment render successful
✅ Environment closed successfully

🎉 All tests passed!
```

## Performance Considerations

### Computational Cost
- **QDARTS CSD generation**: ~30 seconds per step (vs ~1 second for qarray)
- **Barrier physics**: Real-time tunnel coupling updates
- **Memory usage**: Configurable resolution affects memory requirements

### Optimization Strategies
- **Caching**: Consider caching CSD results for similar voltage configurations
- **Resolution**: Lower resolution for faster training, higher for accuracy
- **Batch processing**: Process multiple voltage configurations simultaneously

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure QDARTS is properly installed and in Python path
2. **Configuration errors**: Check YAML syntax and required fields
3. **Memory issues**: Reduce resolution or batch size
4. **Performance issues**: Consider caching or lower resolution

### Debug Mode

Enable debug mode in configuration:

```yaml
training:
  debug: true
```

This will print detailed logs during environment operation.

## Future Enhancements

1. **Performance optimization**: Implement caching and parallel processing
2. **Additional physics**: Support for more complex quantum effects
3. **Configurable rewards**: Make reward weights configurable
4. **Multi-agent support**: Support for multiple quantum devices
5. **Real-time visualization**: Interactive plotting and monitoring

## Dependencies

- `gymnasium`: RL environment framework
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `yaml`: Configuration file parsing
- `qdarts`: Quantum device physics simulator

## License

This implementation follows the same license as the parent project. 