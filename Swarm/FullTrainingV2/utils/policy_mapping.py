
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec
import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent.parent
swarm_dir = current_dir.parent  # Get Swarm directory
sys.path.append(str(swarm_dir))

try:
    from VoltageAgent.ppo_trainer_recurrent import SingleAgentRecurrentPPOModel
except ImportError:
    # Fallback: try adding VoltageAgent directory directly
    voltage_agent_dir = swarm_dir / "VoltageAgent"
    if str(voltage_agent_dir) not in sys.path:
        sys.path.append(str(voltage_agent_dir))
    from ppo_trainer_recurrent import SingleAgentRecurrentPPOModel


def create_rl_module_spec(env_instance) -> MultiRLModuleSpec:
    """
    Create policy specifications for RLlib with the plunger and barrier policies
    (note there are only TWO policies although each has multiple agent instances)
    
    Args:
        env_instance: Instance of the quantum device environment
        
    Returns:
        MultiRLModuleSpec object
    """
    from gymnasium import spaces
    import numpy as np
    
    # Get full environment spaces from base environment
    # Multi-agent wrapper preserves original spaces as base_observation_space and base_action_space
    if hasattr(env_instance, 'base_observation_space'):
        full_obs_space = env_instance.base_observation_space
        full_action_space = env_instance.base_action_space
    else:
        raise ValueError("Training attempted on a non-wrapped environment")
    
    # Extract dimensions from environment
    image_shape = full_obs_space['image'].shape  # (H, W, channels)
    num_gates = full_action_space['action_gate_voltages'].shape[0]
    num_barriers = full_action_space['action_barrier_voltages'].shape[0]
    
    # Gate voltage ranges
    gate_low = full_action_space['action_gate_voltages'].low[0]
    gate_high = full_action_space['action_gate_voltages'].high[0]
    barrier_low = full_action_space['action_barrier_voltages'].low[0]
    barrier_high = full_action_space['action_barrier_voltages'].high[0]
    
    # Create observation space for gate agents
    # Each gate agent sees: dual-channel image + single voltage value
    gate_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(image_shape[0], image_shape[1], 2),  # Dual channel for gate agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=gate_low, high=gate_high,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    # Create action space for gate agents
    # Each gate agent controls: single gate voltage
    gate_action_space = spaces.Box(
            low=gate_low, high=gate_high,
            shape=(1,),  # Single gate voltage output
            dtype=np.float32
        )
    
    # Create observation space for barrier agents  
    # Each barrier agent sees: single-channel image + single voltage value
    barrier_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(image_shape[0], image_shape[1], 1),  # Single channel for barrier agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=barrier_low, high=barrier_high,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    # Create action space for barrier agents
    # Each barrier agent controls: single barrier voltage
    barrier_action_space = spaces.Box(
            low=barrier_low, high=barrier_high,
            shape=(1,),  # Single barrier voltage output
            dtype=np.float32
        )
    
    # Create model config for custom RLModule
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    # Create single agent RLModule specs
    plunger_spec = RLModuleSpec(
        module_class=SingleAgentRecurrentPPOModel,
        observation_space=gate_obs_space,
        action_space=gate_action_space,
        model_config=model_config,
        learner_only=False  # Allow inference on EnvRunners
    )
    
    barrier_spec = RLModuleSpec(
        module_class=SingleAgentRecurrentPPOModel,
        observation_space=barrier_obs_space,
        action_space=barrier_action_space,
        model_config=model_config,
        learner_only=False  # Allow inference on EnvRunners
    )

    # Create multi-agent RLModule spec
    rl_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "plunger_policy": plunger_spec,
            "barrier_policy": barrier_spec,
        }
    )

    return rl_module_spec

