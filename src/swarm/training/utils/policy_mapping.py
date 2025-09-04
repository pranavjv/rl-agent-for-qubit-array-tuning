def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs. Ray 2.49.0 passes agent_id and episode."""
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type. "
            f"Expected format: 'plunger_X' or 'barrier_X' where X is the agent number."
        )

