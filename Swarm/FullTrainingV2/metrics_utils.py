"""
Focused training metrics extraction for losses, rewards, and episode lengths.
"""

def extract_training_metrics(result, iteration_time):
    """Extract focused training metrics: losses, rewards, and episode lengths."""
    metrics = {}
    
    # === BASIC INFO ===
    metrics['iteration_time'] = iteration_time
    
    # === LEARNING LOSSES ===
    learners = result.get('learners', {})
    
    # Plunger policy losses
    plunger_learner = learners.get('plunger_policy', {})
    metrics['plunger_total_loss'] = plunger_learner.get('total_loss', 'N/A')
    metrics['plunger_policy_loss'] = plunger_learner.get('policy_loss', 'N/A')
    metrics['plunger_vf_loss'] = plunger_learner.get('vf_loss', 'N/A')
    
    # Barrier policy losses
    barrier_learner = learners.get('barrier_policy', {})
    metrics['barrier_total_loss'] = barrier_learner.get('total_loss', 'N/A')
    metrics['barrier_policy_loss'] = barrier_learner.get('policy_loss', 'N/A')
    metrics['barrier_vf_loss'] = barrier_learner.get('vf_loss', 'N/A')
    
    # === EPISODE REWARDS AND LENGTHS ===
    env_runners = result.get('env_runners', {})
    
    # Episode rewards (all available reward metrics)
    metrics['episode_reward_mean'] = env_runners.get('episode_reward_mean', 'N/A')
    metrics['episode_reward_min'] = env_runners.get('episode_reward_min', 'N/A') 
    metrics['episode_reward_max'] = env_runners.get('episode_reward_max', 'N/A')
    
    # Episode lengths (rollout lengths - will be shorter if episodes terminate early)
    metrics['episode_len_mean'] = env_runners.get('episode_len_mean', 'N/A')
    metrics['episode_len_min'] = env_runners.get('episode_len_min', 'N/A')
    metrics['episode_len_max'] = env_runners.get('episode_len_max', 'N/A')
    
    # === ENVIRONMENT STEPS (for tracking progress) ===
    metrics['num_env_steps_sampled'] = env_runners.get('num_env_steps_sampled', 0)
    
    # === CREATE SUMMARY STRING ===
    def format_metric(value, decimals=3):
        if value == 'N/A' or value is None:
            return 'N/A'
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    summary_parts = [
        f"env_steps={metrics['num_env_steps_sampled']}",
        f"plunger_loss={format_metric(metrics['plunger_total_loss'])}",
        f"barrier_loss={format_metric(metrics['barrier_total_loss'])}",
        f"episode_reward={format_metric(metrics['episode_reward_mean'])}",
        f"episode_len={format_metric(metrics['episode_len_mean'], 1)}",
        f"time={format_metric(iteration_time, 1)}s"
    ]
    metrics['summary'] = ', '.join(summary_parts)
    
    return metrics


def log_training_metrics(logger, iteration, metrics):
    """Log comprehensive training metrics to logger."""
    logger.info(f"=" * 80)
    logger.info(f"ITERATION {iteration} DETAILED METRICS")
    logger.info(f"=" * 80)
    
    # Timing
    logger.info(f"TIMING:")
    logger.info(f"  Total iteration time: {metrics['iteration_time']:.2f}s")
    logger.info(f"  Environment sampling: {metrics['env_sampling_time']:.2f}s")
    logger.info(f"  Learner updates: {metrics['learner_update_time']:.2f}s")
    logger.info(f"  Weight synchronization: {metrics['synch_weights_time']:.3f}s")
    
    # Sampling
    logger.info(f"SAMPLING:")
    logger.info(f"  Environment steps (current): {metrics['num_env_steps_sampled']}")
    logger.info(f"  Environment steps (lifetime): {metrics['num_env_steps_sampled_lifetime']}")
    logger.info(f"  Agent steps (current): {metrics['total_agent_steps_current']}")
    logger.info(f"  Agent steps (lifetime): {metrics['total_agent_steps_lifetime']}")
    logger.info(f"  Plunger agent steps: {metrics['plunger_agent_steps']}")
    logger.info(f"  Barrier agent steps: {metrics['barrier_agent_steps']}")
    logger.info(f"  Healthy workers: {metrics['num_healthy_workers']}")
    logger.info(f"  Worker restarts: {metrics['num_worker_restarts']}")
    
    # Learning - Plunger Policy
    logger.info(f"PLUNGER POLICY LEARNING:")
    logger.info(f"  Total loss: {metrics['plunger_total_loss']}")
    logger.info(f"  Policy loss: {metrics['plunger_policy_loss']}")
    logger.info(f"  Value function loss: {metrics['plunger_vf_loss']}")
    logger.info(f"  Entropy: {metrics['plunger_entropy']}")
    logger.info(f"  KL divergence: {metrics['plunger_kl_loss']}")
    logger.info(f"  VF explained variance: {metrics['plunger_vf_explained_var']}")
    logger.info(f"  Gradient norm: {metrics['plunger_grad_norm']}")
    logger.info(f"  Learning rate: {metrics['plunger_learning_rate']}")
    logger.info(f"  KL coefficient: {metrics['plunger_kl_coeff']}")
    logger.info(f"  Module steps trained: {metrics['plunger_module_steps_trained']}")
    
    # Learning - Barrier Policy
    logger.info(f"BARRIER POLICY LEARNING:")
    logger.info(f"  Total loss: {metrics['barrier_total_loss']}")
    logger.info(f"  Policy loss: {metrics['barrier_policy_loss']}")
    logger.info(f"  Value function loss: {metrics['barrier_vf_loss']}")
    logger.info(f"  Entropy: {metrics['barrier_entropy']}")
    logger.info(f"  KL divergence: {metrics['barrier_kl_loss']}")
    logger.info(f"  VF explained variance: {metrics['barrier_vf_explained_var']}")
    logger.info(f"  Gradient norm: {metrics['barrier_grad_norm']}")
    logger.info(f"  Learning rate: {metrics['barrier_learning_rate']}")
    logger.info(f"  KL coefficient: {metrics['barrier_kl_coeff']}")
    logger.info(f"  Module steps trained: {metrics['barrier_module_steps_trained']}")
    
    # System Performance
    logger.info(f"SYSTEM PERFORMANCE:")
    logger.info(f"  CPU utilization: {metrics['cpu_util_percent']}%")
    logger.info(f"  RAM utilization: {metrics['ram_util_percent']}%")
    logger.info(f"  Environment throughput: {metrics['env_steps_throughput']:.2f} steps/s")
    logger.info(f"  Module training throughput: {metrics['module_steps_throughput']:.2f} steps/s")
    logger.info(f"  Total trainable parameters: {metrics['total_trainable_params']:,}")
    logger.info(f"  Total module steps trained: {metrics['total_num_module_steps_trained']}")
    logger.info(f"  Total environment steps trained: {metrics['total_num_env_steps_trained']}")
    
    logger.info(f"=" * 80)


def analyze_training_health(metrics, logger, iteration):
    """Analyze training health and provide warnings/recommendations."""
    warnings = []
    recommendations = []
    
    # Check for learning problems
    if metrics['plunger_vf_explained_var'] != 'N/A' and float(metrics['plunger_vf_explained_var']) < -0.5:
        warnings.append("Plunger policy value function is performing poorly (explained variance < -0.5)")
        recommendations.append("Consider adjusting vf_loss_coeff or learning rate for plunger policy")
    
    if metrics['barrier_vf_explained_var'] != 'N/A' and float(metrics['barrier_vf_explained_var']) < -0.5:
        warnings.append("Barrier policy value function is performing poorly (explained variance < -0.5)")
        recommendations.append("Consider adjusting vf_loss_coeff or learning rate for barrier policy")
    
    # Check for gradient issues
    if metrics['plunger_grad_norm'] != 'N/A' and float(metrics['plunger_grad_norm']) > 10.0:
        warnings.append(f"Large plunger policy gradient norm: {metrics['plunger_grad_norm']}")
        recommendations.append("Consider adding gradient clipping or reducing learning rate")
    
    if metrics['barrier_grad_norm'] != 'N/A' and float(metrics['barrier_grad_norm']) > 10.0:
        warnings.append(f"Large barrier policy gradient norm: {metrics['barrier_grad_norm']}")
        recommendations.append("Consider adding gradient clipping or reducing learning rate")
    
    # Check for entropy issues (exploration)
    if metrics['plunger_entropy'] != 'N/A' and float(metrics['plunger_entropy']) < 0.1:
        warnings.append(f"Low plunger policy entropy: {metrics['plunger_entropy']} (may indicate insufficient exploration)")
        recommendations.append("Consider increasing entropy_coeff for more exploration")
    
    if metrics['barrier_entropy'] != 'N/A' and float(metrics['barrier_entropy']) < 0.1:
        warnings.append(f"Low barrier policy entropy: {metrics['barrier_entropy']} (may indicate insufficient exploration)")
        recommendations.append("Consider increasing entropy_coeff for more exploration")
    
    # Check system performance
    if metrics['cpu_util_percent'] != 'N/A' and float(metrics['cpu_util_percent']) > 95:
        warnings.append(f"High CPU utilization: {metrics['cpu_util_percent']}%")
        recommendations.append("Consider reducing num_env_runners or rollout_fragment_length")
    
    if metrics['ram_util_percent'] != 'N/A' and float(metrics['ram_util_percent']) > 90:
        warnings.append(f"High RAM utilization: {metrics['ram_util_percent']}%")
        recommendations.append("Consider reducing batch sizes or max_seq_len")
    
    # Check worker health
    if metrics['num_worker_restarts'] > 0:
        warnings.append(f"Worker restarts detected: {metrics['num_worker_restarts']}")
        recommendations.append("Check for environment or model errors causing worker failures")
    
    # Log warnings and recommendations
    if warnings or recommendations:
        logger.warning(f"TRAINING HEALTH CHECK - Iteration {iteration}")
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")
        for rec in recommendations:
            logger.info(f"  💡 {rec}")
        logger.warning("-" * 60)