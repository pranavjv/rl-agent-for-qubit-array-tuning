"""
Policy mapping utilities for multi-agent quantum device environment.
Maps agent IDs to appropriate policies based on agent type (plunger vs barrier).
"""

from typing import Dict, Any, Optional


def get_policy_mapping_fn(num_quantum_dots: int = 8):
    """
    Create a policy mapping function for the quantum device environment.
    
    Args:
        num_quantum_dots: Number of quantum dots (N), typically 8
        
    Returns:
        Function that maps agent IDs to policy IDs
    """
    
    def policy_mapping_fn(agent_id: str, episode: Optional[Any] = None, worker: Optional[Any] = None, **kwargs) -> str:
        """
        Map agent ID to policy ID.
        
        Args:
            agent_id: The agent identifier
            episode: Current episode (unused)
            worker: Current worker (unused)
            
        Returns:
            Policy ID string
        """
        if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
            return "plunger_policy"
        elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
            return "barrier_policy"
        else:
            # Fallback: try to infer from numeric ID if using numeric agent IDs
            # Assuming agents 0 to N-1 are plungers, N to 2N-2 are barriers
            try:
                agent_num = int(agent_id.split("_")[-1]) if "_" in agent_id else int(agent_id)
                if agent_num < num_quantum_dots:
                    return "plunger_policy"
                else:
                    return "barrier_policy"
            except (ValueError, IndexError):
                # Default to plunger if we can't determine type
                return "plunger_policy"
    
    return policy_mapping_fn


def get_policies_to_train():
    """
    Get the list of policies that should be trained.
    
    Returns:
        List of policy IDs to train
    """
    return ["plunger_policy", "barrier_policy"]


def create_policy_specs(env_instance) -> Dict[str, Any]:
    """
    Create policy specifications for RLlib based on environment observation/action spaces.
    
    Args:
        env_instance: Instance of the quantum device environment
        
    Returns:
        Dictionary mapping policy IDs to policy specifications
    """
    # Get observation and action spaces from environment
    # Assuming the environment has methods to get spaces for different agent types
    obs_space = env_instance.observation_space
    action_space = env_instance.action_space
    
    # If the environment returns spaces per agent, we might need to handle that
    if isinstance(obs_space, dict):
        # Multi-agent environment with different spaces per agent
        # We'll use the first plunger and barrier agent spaces as templates
        plunger_obs_space = None
        barrier_obs_space = None
        plunger_action_space = None
        barrier_action_space = None
        
        for agent_id, space in obs_space.items():
            if "plunger" in agent_id.lower() and plunger_obs_space is None:
                plunger_obs_space = space
            elif "barrier" in agent_id.lower() and barrier_obs_space is None:
                barrier_obs_space = space
        
        for agent_id, space in action_space.items():
            if "plunger" in agent_id.lower() and plunger_action_space is None:
                plunger_action_space = space
            elif "barrier" in agent_id.lower() and barrier_action_space is None:
                barrier_action_space = space
    else:
        # Single agent spaces - use for both policies
        plunger_obs_space = obs_space
        barrier_obs_space = obs_space
        plunger_action_space = action_space
        barrier_action_space = action_space
    
    policies = {
        "plunger_policy": {
            "observation_space": plunger_obs_space,
            "action_space": plunger_action_space,
        },
        "barrier_policy": {
            "observation_space": barrier_obs_space,
            "action_space": barrier_action_space,
        }
    }
    
    return policies


def get_agent_ids(num_quantum_dots: int = 8):
    """
    Generate expected agent IDs for the quantum device environment.
    
    Args:
        num_quantum_dots: Number of quantum dots (N)
        
    Returns:
        List of agent IDs
    """
    agent_ids = []
    
    # N plunger agents (one per quantum dot)
    for i in range(num_quantum_dots):
        agent_ids.append(f"plunger_{i}")
    
    # N-1 barrier agents (between quantum dots)
    for i in range(num_quantum_dots - 1):
        agent_ids.append(f"barrier_{i}")
    
    return agent_ids


def validate_agent_assignment(agent_ids, num_quantum_dots: int = 8):
    """
    Validate that agent IDs match expected structure.
    
    Args:
        agent_ids: List of agent IDs from environment
        num_quantum_dots: Expected number of quantum dots
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_plungers = num_quantum_dots
    expected_barriers = num_quantum_dots - 1
    
    plunger_count = sum(1 for aid in agent_ids if "plunger" in aid.lower())
    barrier_count = sum(1 for aid in agent_ids if "barrier" in aid.lower())
    
    if plunger_count != expected_plungers:
        return False, f"Expected {expected_plungers} plunger agents, got {plunger_count}"
    
    if barrier_count != expected_barriers:
        return False, f"Expected {expected_barriers} barrier agents, got {barrier_count}"
    
    return True, "Agent assignment is valid" 