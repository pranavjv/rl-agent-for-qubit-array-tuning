"""
Logging utilities for quantum device training with Weights & Biases integration.
"""

import os
import sys
import psutil
import gc
import time
import logging
from datetime import datetime
from pathlib import Path
import wandb


def log_memory_usage_wandb(stage: str, additional_info: str = "", step: int = None):
    """Log detailed memory usage information to Weights & Biases."""
    try:
        # System memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process memory info
        process = psutil.Process()
        process_memory = process.memory_info()
        process_percent = process.memory_percent()
        
        # Disk usage for /tmp (where Ray stores temp files)
        tmp_usage = psutil.disk_usage('/tmp')
        
        # Create memory metrics dictionary
        memory_metrics = {
            f"memory/{stage}/system_ram_percent": memory.percent,
            f"memory/{stage}/system_ram_used_gb": memory.used / 1e9,
            f"memory/{stage}/system_ram_total_gb": memory.total / 1e9,
            f"memory/{stage}/system_ram_available_gb": memory.available / 1e9,
            f"memory/{stage}/process_rss_gb": process_memory.rss / 1e9,
            f"memory/{stage}/process_percent": process_percent,
            f"memory/{stage}/process_vms_gb": process_memory.vms / 1e9,
            f"memory/{stage}/swap_percent": swap.percent,
            f"memory/{stage}/swap_used_gb": swap.used / 1e9,
            f"memory/{stage}/tmp_disk_percent": tmp_usage.percent,
            f"memory/{stage}/tmp_disk_used_gb": tmp_usage.used / 1e9,
            f"memory/{stage}/gc_objects": len(gc.get_objects()),
        }
        
        # Log to wandb
        if step is not None:
            wandb.log(memory_metrics, step=step)
        else:
            wandb.log(memory_metrics)
        
        # Console log for key info
        print(f"MEMORY [{stage}] {additional_info} | "
              f"RAM: {memory.percent:.1f}% | Process: {process_percent:.1f}% | "
              f"Available: {memory.available/1e9:.2f}GB")
        
        # Log warnings to console for high memory usage
        if memory.percent > 90:
            print(f"WARNING: HIGH SYSTEM MEMORY USAGE: {memory.percent:.1f}%")
        if process_percent > 50:
            print(f"WARNING: HIGH PROCESS MEMORY USAGE: {process_percent:.1f}%")
        if tmp_usage.percent > 90:
            print(f"WARNING: HIGH /tmp DISK USAGE: {tmp_usage.percent:.1f}%")
            
    except Exception as e:
        print(f"Error logging memory usage for {stage}: {e}")


def memory_checkpoint_wandb(stage: str, additional_info: str = "", step: int = None):
    """Create a memory usage checkpoint with forced garbage collection and log to wandb."""
    # Force garbage collection before logging
    gc.collect()
    log_memory_usage_wandb(stage, additional_info, step)


def log_training_metrics_wandb(metrics: dict, iteration: int):
    """Log training metrics to Weights & Biases."""
    # Create wandb metrics dictionary with proper formatting
    wandb_metrics = {}

    print("[DEBUG] wandb training metrics: ", metrics)
    
    # Training losses
    if metrics['plunger_total_loss'] != 'N/A':
        wandb_metrics['training/plunger_total_loss'] = float(metrics['plunger_total_loss'])
    if metrics['plunger_policy_loss'] != 'N/A':
        wandb_metrics['training/plunger_policy_loss'] = float(metrics['plunger_policy_loss'])
    if metrics['plunger_vf_loss'] != 'N/A':
        wandb_metrics['training/plunger_vf_loss'] = float(metrics['plunger_vf_loss'])
        
    if metrics['barrier_total_loss'] != 'N/A':
        wandb_metrics['training/barrier_total_loss'] = float(metrics['barrier_total_loss'])
    if metrics['barrier_policy_loss'] != 'N/A':
        wandb_metrics['training/barrier_policy_loss'] = float(metrics['barrier_policy_loss'])
    if metrics['barrier_vf_loss'] != 'N/A':
        wandb_metrics['training/barrier_vf_loss'] = float(metrics['barrier_vf_loss'])
    
    # Episode rewards
    if metrics['episode_reward_mean'] != 'N/A':
        wandb_metrics['episode/reward_mean'] = float(metrics['episode_reward_mean'])
    if metrics['episode_reward_min'] != 'N/A':
        wandb_metrics['episode/reward_min'] = float(metrics['episode_reward_min'])
    if metrics['episode_reward_max'] != 'N/A':
        wandb_metrics['episode/reward_max'] = float(metrics['episode_reward_max'])
    
    # Episode lengths (rollout lengths)
    if metrics['episode_len_mean'] != 'N/A':
        wandb_metrics['episode/length_mean'] = float(metrics['episode_len_mean'])
    if metrics['episode_len_min'] != 'N/A':
        wandb_metrics['episode/length_min'] = float(metrics['episode_len_min'])
    if metrics['episode_len_max'] != 'N/A':
        wandb_metrics['episode/length_max'] = float(metrics['episode_len_max'])
    
    # Environment steps
    wandb_metrics['training/env_steps_sampled'] = metrics['num_env_steps_sampled']
    wandb_metrics['training/iteration_time'] = metrics['iteration_time']
    
    # Log to wandb
    print(f"[DEBUG] Final wandb_metrics to be logged: {wandb_metrics}")
    try:
        wandb.log(wandb_metrics, step=iteration)
        print(f"[DEBUG] Successfully logged to wandb for step {iteration}")
    except Exception as e:
        print(f"[DEBUG] ERROR logging to wandb: {e}")
        raise


def setup_memory_logging():
    """Setup detailed memory usage logging to file (fallback when wandb disabled)."""
    current_dir = Path(__file__).parent.parent
    os.makedirs(current_dir / "logs", exist_ok=True)
    log_filename = f"logs/memory_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = current_dir / log_filename
    
    # Create logger
    logger = logging.getLogger('memory_monitor')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    print(f"Memory logging initialized. Log file: {log_path}")
    return logger


def log_memory_usage(logger, stage: str, additional_info: str = ""):
    """Log detailed memory usage information to file (fallback when wandb disabled)."""
    try:
        import ray
        
        # System memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process memory info
        process = psutil.Process()
        process_memory = process.memory_info()
        process_percent = process.memory_percent()
        
        # Disk usage for /tmp (where Ray stores temp files)
        tmp_usage = psutil.disk_usage('/tmp')
        
        # Ray memory info if available
        ray_memory_info = ""
        if ray.is_initialized():
            try:
                memory_summary = ray._private.internal_api.memory_summary(
                    stats_only=True, 
                    line_wrap_length=10000
                )
                ray_memory_info = f" | Ray Memory: {memory_summary}"
            except:
                ray_memory_info = " | Ray Memory: Unable to get summary"
        
        # Garbage collection stats
        gc_stats = str(gc.get_stats())
        
        log_message = (
            f"MEMORY [{stage}] {additional_info} | "
            f"System RAM: {memory.percent:.1f}% ({memory.used/1e9:.2f}GB/{memory.total/1e9:.2f}GB) | "
            f"Available: {memory.available/1e9:.2f}GB | "
            f"Process: {process_memory.rss/1e9:.2f}GB ({process_percent:.1f}%) | "
            f"Process VMS: {process_memory.vms/1e9:.2f}GB | "
            f"Swap: {swap.percent:.1f}% ({swap.used/1e9:.2f}GB/{swap.total/1e9:.2f}GB) | "
            f"/tmp disk: {tmp_usage.percent:.1f}% ({tmp_usage.used/1e9:.2f}GB/{tmp_usage.total/1e9:.2f}GB)"
            f"{ray_memory_info} | "
            f"GC: {len(gc.get_objects())} objects"
        )
        
        logger.info(log_message)
        
        # Additional check for high memory usage
        if memory.percent > 90:
            logger.warning(f"HIGH SYSTEM MEMORY USAGE: {memory.percent:.1f}%")
        if process_percent > 50:
            logger.warning(f"HIGH PROCESS MEMORY USAGE: {process_percent:.1f}%")
        if tmp_usage.percent > 90:
            logger.warning(f"HIGH /tmp DISK USAGE: {tmp_usage.percent:.1f}%")
            
    except Exception as e:
        logger.error(f"Error logging memory usage for {stage}: {e}")


def memory_checkpoint(logger, stage: str, additional_info: str = ""):
    """Create a memory usage checkpoint with forced garbage collection (fallback when wandb disabled)."""
    # Force garbage collection before logging
    gc.collect()
    log_memory_usage(logger, stage, additional_info)