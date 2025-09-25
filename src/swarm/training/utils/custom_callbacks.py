"""
Callbacks for custom logging during training
see docs at:
https://docs.ray.io/en/latest/rllib/package_ref/callback.html
"""

import time
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict, Any


class CustomCallbacks(RLlibCallback):
    # migrate all callbacks to this class
    def __init__(self, log_images: bool = False):
        super().__init__()
        self.log_images = log_images

    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        print("[DEBUG] log std callback called - on_learn_on_batch")

        print(f"Policy: {policy}")
        print(f"Policy model: {policy.model}")
    
    def on_sample_end(
        self,
        *,
        env_runner=None,
        metrics_logger=None,
        samples,
        worker=None,
        **kwargs
    ):
        print("[DEBUG] log std callback called - on_sample_end")

        print(f"Samples: {samples}")
        print(f"Env runner: {env_runner}")
        print(f"Worker: {worker}")
        
        # Access policies from the env_runner or worker
        active_runner = env_runner or worker
        if active_runner and hasattr(active_runner, 'policy_map'):
            for policy_id, policy in active_runner.policy_map.items():
                print(f"Policy: {policy}")
                print(f"Policy model: {policy.model}")
                
                if hasattr(policy.model, "log_std"):
                    log_std = policy.model.log_std.detach().cpu().numpy()
                    
                    # Store metrics in the samples for later aggregation
                    if not hasattr(samples, 'custom_metrics'):
                        samples.custom_metrics = {}
                    
                    if "plunger" in policy_id:
                        samples.custom_metrics["plunger_log_std_mean"] = float(log_std.mean())
                        samples.custom_metrics["plunger_log_std_min"] = float(log_std.min())
                        samples.custom_metrics["plunger_log_std_max"] = float(log_std.max())
                    elif "barrier" in policy_id:
                        samples.custom_metrics["barrier_log_std_mean"] = float(log_std.mean())
                        samples.custom_metrics["barrier_log_std_min"] = float(log_std.min())
                        samples.custom_metrics["barrier_log_std_max"] = float(log_std.max())

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        print("[DEBUG] ===== MAXIMAL NETWORK OUTPUT EXPLORATION =====")
        
        def exhaustive_network_exploration(obj, name="", max_depth=4, current_depth=0, visited=None):
            """Exhaustively explore network structure and try forward passes"""
            if visited is None:
                visited = set()
            
            if current_depth > max_depth or id(obj) in visited:
                return {}
            
            visited.add(id(obj))
            
            findings = {}
            
            print(f"{'  ' * current_depth}[EXPLORE] {name}: {type(obj)}")
            
            # Try forward pass on any callable object
            if callable(obj) and hasattr(obj, '__call__'):
                try:
                    # Try with different dummy inputs
                    import torch
                    
                    test_inputs = [
                        torch.zeros(1, 2, 100, 100),  # Plunger format
                        torch.zeros(1, 1, 100, 100),  # Barrier format  
                        torch.zeros(1, 256),          # Encoded format
                        torch.zeros(256),             # Single batch format
                        torch.zeros(1, 128),          # Intermediate format
                    ]
                    
                    for i, test_input in enumerate(test_inputs):
                        try:
                            output = obj(test_input)
                            if torch.is_tensor(output):
                                shape = output.shape
                                values = output.detach().cpu().numpy()
                                
                                print(f"{'  ' * current_depth}[FORWARD_SUCCESS] {name} with input_{i} {test_input.shape} -> output {shape}")
                                print(f"{'  ' * current_depth}[OUTPUT_VALUES] {values}")
                                
                                # Check if this could be log_std output
                                if len(shape) >= 1 and shape[-1] == 2:  # [mean, log_std] format
                                    potential_log_std = values[..., 1]  # Second element
                                    findings[f"{name}_forward_pass_{i}"] = {
                                        'input_shape': test_input.shape,
                                        'output_shape': shape,
                                        'potential_log_std': potential_log_std,
                                        'full_output': values
                                    }
                                    print(f"{'  ' * current_depth}[POTENTIAL_LOG_STD] {name} output[1]: {potential_log_std}")
                                    
                                elif len(values.shape) == 1 and len(values) >= 1:  # Single value outputs
                                    findings[f"{name}_single_output_{i}"] = {
                                        'input_shape': test_input.shape,
                                        'output_value': float(values[0]) if len(values) > 0 else None,
                                        'full_output': values
                                    }
                                    
                        except Exception as e:
                            if 'size mismatch' not in str(e) and 'shape' not in str(e).lower():
                                print(f"{'  ' * current_depth}[FORWARD_ERROR] {name} with input_{i}: {e}")
                            
                except Exception as e:
                    print(f"{'  ' * current_depth}[FORWARD_FAILED] {name}: {e}")
            
            # Recursively explore attributes
            if hasattr(obj, '__dict__') or hasattr(obj, 'keys'):
                attrs_to_check = []
                
                # Get attributes from __dict__
                if hasattr(obj, '__dict__'):
                    attrs_to_check.extend([(k, v) for k, v in obj.__dict__.items() if not k.startswith('_')])
                
                # Get keys if dict-like
                if hasattr(obj, 'keys') and callable(obj.keys):
                    try:
                        for key in list(obj.keys())[:10]:  # Limit to prevent spam
                            attrs_to_check.append((key, obj[key]))
                    except:
                        pass
                
                # Check important neural network attributes
                important_attrs = ['encoder', 'pi', 'actor', 'policy_head', 'action_dist_net', 'mlp', 'layers', 'fc', 'linear']
                for attr in important_attrs:
                    if hasattr(obj, attr):
                        try:
                            attrs_to_check.append((attr, getattr(obj, attr)))
                        except:
                            pass
                
                # Explore each attribute
                for attr_name, attr_val in attrs_to_check[:15]:  # Limit depth
                    if attr_val is not None and current_depth < max_depth:
                        sub_findings = exhaustive_network_exploration(
                            attr_val, f"{name}.{attr_name}", max_depth, current_depth + 1, visited
                        )
                        findings.update(sub_findings)
            
            return findings
        
        try:
            print("[DISCOVERY] free_log_std=False - searching ALL network paths for log_std outputs")
            
            if hasattr(algorithm, 'learner_group'):
                learner_group = algorithm.learner_group
                
                # 1. EXHAUSTIVE WEIGHTS ANALYSIS
                print(f"\n[DEBUG] ===== COMPREHENSIVE WEIGHTS ANALYSIS =====")
                if hasattr(learner_group, 'get_weights'):
                    try:
                        weights = learner_group.get_weights()
                        
                        for module_id, module_weights in weights.items():
                            print(f"\n[DEBUG] === ANALYZING ALL WEIGHTS IN {module_id} ===")
                            
                            # Check EVERY weight tensor for potential log_std relevance
                            for weight_name, weight_tensor in module_weights.items():
                                shape = getattr(weight_tensor, 'shape', 'no shape')
                                print(f"[WEIGHT] {weight_name}: shape={shape}")
                                
                                # Look for output layers (anything that outputs to size 2 could be [mean, log_std])
                                if hasattr(weight_tensor, 'shape') and len(weight_tensor.shape) >= 2:
                                    if weight_tensor.shape[0] == 2:  # Output dimension is 2
                                        print(f"[POTENTIAL_OUTPUT] {weight_name} outputs 2 values - could be [mean, log_std]")
                                        
                                        if 'bias' in weight_name:
                                            bias_vals = weight_tensor
                                            print(f"[BIAS_VALUES] {weight_name}: {bias_vals}")
                                            if len(bias_vals) == 2:
                                                log_std_bias = float(bias_vals[1])
                                                print(f"[LOG_STD_CANDIDATE] {module_id}.{weight_name}[1] = {log_std_bias:.4f}")
                                                
                                                # Store this finding
                                                if "plunger" in module_id:
                                                    result[f"plunger_log_std_from_{weight_name}"] = log_std_bias
                                                elif "barrier" in module_id:
                                                    result[f"barrier_log_std_from_{weight_name}"] = log_std_bias
                                        
                                        elif 'weight' in weight_name:
                                            weight_vals = weight_tensor[1] if len(weight_tensor) > 1 else weight_tensor
                                            weight_stats = {
                                                'mean': float(weight_vals.mean()),
                                                'std': float(weight_vals.std()), 
                                                'min': float(weight_vals.min()),
                                                'max': float(weight_vals.max())
                                            }
                                            print(f"[WEIGHT_STATS] {weight_name}[1] (log_std weights): {weight_stats}")
                                            
                                            if "plunger" in module_id:
                                                result[f"plunger_weights_{weight_name.replace('.', '_')}_stats"] = weight_stats['mean']
                                            elif "barrier" in module_id:
                                                result[f"barrier_weights_{weight_name.replace('.', '_')}_stats"] = weight_stats['mean']
                                            
                    except Exception as e:
                        print(f"[ERROR] comprehensive weights analysis: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 2. EXHAUSTIVE NETWORK FORWARD PASS EXPLORATION
                print(f"\n[DEBUG] ===== MAXIMAL FORWARD PASS EXPLORATION =====")
                if hasattr(learner_group, 'foreach_learner'):
                    try:
                        def comprehensive_network_exploration(learner):
                            import torch
                            exploration_results = {
                                'learner_type': str(type(learner)),
                                'network_findings': {},
                                'forward_pass_results': {}
                            }
                            
                            if hasattr(learner, '_rl_module'):
                                rl_module = learner._rl_module
                                
                                for mod_id in rl_module.keys():
                                    mod = rl_module[mod_id]
                                    print(f"[EXPLORING] Module {mod_id}: {type(mod)}")
                                    
                                    # Comprehensive network exploration
                                    findings = exhaustive_network_exploration(mod, f"module_{mod_id}")
                                    exploration_results['network_findings'][mod_id] = findings
                                    
                                    # Try multiple forward pass strategies
                                    forward_results = {}
                                    
                                    # Strategy 1: Full forward pass
                                    try:
                                        if 'plunger' in mod_id:
                                            dummy_obs = torch.zeros(1, 2, 100, 100)
                                        else:
                                            dummy_obs = torch.zeros(1, 1, 100, 100)
                                        
                                        # Try different forward pass methods
                                        if hasattr(mod, 'forward'):
                                            full_output = mod.forward(dummy_obs)
                                            if torch.is_tensor(full_output):
                                                vals = full_output.detach().cpu().numpy()
                                                forward_results['full_forward'] = {
                                                    'shape': full_output.shape,
                                                    'values': vals.tolist()
                                                }
                                        
                                        # Try step-by-step forward pass
                                        if hasattr(mod, 'encoder') and hasattr(mod, 'pi'):
                                            encoded = mod.encoder(dummy_obs)
                                            pi_out = mod.pi(encoded)
                                            
                                            if torch.is_tensor(pi_out):
                                                pi_vals = pi_out.detach().cpu().numpy()
                                                forward_results['stepwise_forward'] = {
                                                    'encoded_shape': encoded.shape,
                                                    'pi_shape': pi_out.shape,
                                                    'pi_values': pi_vals.tolist()
                                                }
                                                
                                                # Extract log_std if shape suggests [mean, log_std]
                                                if len(pi_vals.shape) >= 1 and pi_vals.shape[-1] == 2:
                                                    log_std_val = pi_vals[0, 1] if len(pi_vals.shape) > 1 else pi_vals[1]
                                                    forward_results['extracted_log_std'] = float(log_std_val)
                                        
                                        # Try action distribution creation
                                        if hasattr(mod, '_action_dist_class'):
                                            dist_class = mod._action_dist_class
                                            if hasattr(dist_class, 'from_logits'):
                                                action_dist = dist_class.from_logits(pi_out)
                                                if hasattr(action_dist, 'scale'):
                                                    scale_val = action_dist.scale.detach().cpu().numpy()
                                                    forward_results['action_dist_scale'] = scale_val.tolist()
                                                if hasattr(action_dist, 'concentration'):
                                                    conc_val = action_dist.concentration.detach().cpu().numpy() 
                                                    forward_results['action_dist_concentration'] = conc_val.tolist()
                                                    
                                    except Exception as e:
                                        forward_results['error'] = str(e)
                                    
                                    exploration_results['forward_pass_results'][mod_id] = forward_results
                            
                            return exploration_results
                        
                        exploration_results = learner_group.foreach_learner(comprehensive_network_exploration)
                        
                        # Process comprehensive results
                        for i, learner_result in enumerate(exploration_results):
                            # Handle CallResult objects
                            if hasattr(learner_result, 'get'):
                                actual_result = learner_result.get()
                            else:
                                actual_result = learner_result
                            
                            print(f"\n[COMPREHENSIVE_RESULTS] Learner {i}:")
                            
                            # Process forward pass results
                            for mod_id, forward_data in actual_result.get('forward_pass_results', {}).items():
                                print(f"[MODULE_RESULTS] {mod_id}: {forward_data}")
                                
                                # Extract any log_std values found
                                if 'extracted_log_std' in forward_data:
                                    log_std_val = forward_data['extracted_log_std']
                                    print(f"[SUCCESS] Extracted log_std from {mod_id}: {log_std_val:.4f}")
                                    
                                    if "plunger" in mod_id:
                                        result["plunger_log_std_network_output"] = float(log_std_val)
                                    elif "barrier" in mod_id:
                                        result["barrier_log_std_network_output"] = float(log_std_val)
                                
                                # Check action distribution outputs
                                for key, value in forward_data.items():
                                    if 'scale' in key or 'concentration' in key:
                                        print(f"[ACTION_DIST] {mod_id}.{key}: {value}")
                                        
                                        if "plunger" in mod_id:
                                            result[f"plunger_{key}"] = value
                                        elif "barrier" in mod_id:
                                            result[f"barrier_{key}"] = value
                        
                    except Exception as e:
                        print(f"[ERROR] comprehensive network exploration: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 3. FINAL COMPREHENSIVE SUMMARY
                print(f"\n[DEBUG] ===== COMPREHENSIVE EXTRACTION SUMMARY =====")
                log_std_metrics = [k for k in result.keys() if 'log_std' in k or 'scale' in k or 'concentration' in k]
                
                if log_std_metrics:
                    print(f"[SUCCESS] ✅ Found {len(log_std_metrics)} potential log_std/variance metrics:")
                    for metric in log_std_metrics:
                        print(f"[SUCCESS]   {metric}: {result[metric]}")
                else:
                    print(f"[CONCLUSION] ❌ No log_std metrics found after exhaustive search")
                    print(f"[RECOMMENDATION] Consider enabling free_log_std=True for explicit log_std parameters")
                        
            else:
                print(f"[ERROR] Algorithm has no learner_group attribute")
                
        except Exception as e:
            print(f"[FATAL_ERROR] Comprehensive exploration failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("[DEBUG] ===== END MAXIMAL NETWORK EXPLORATION =====")

    def on_episode_end(
        self,
        *,
        episode,
        prev_episode_chunks=None,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs
    ):
        print('[DEBUG] log std callback called - on_episode_end')
        print(f'[DEBUG] Episode: {episode}')
        print(f'[DEBUG] Env runner: {env_runner}')
        print(f'[DEBUG] Worker: {worker}')
        print(f'[DEBUG] RL module: {rl_module}')
        print(f'[DEBUG] Policies: {policies}')
        
        # Try to extract log std from policies (old API stack compatibility)
        if policies:
            for policy_id, policy in policies.items():
                if hasattr(policy, 'model') and hasattr(policy.model, "log_std"):
                    log_std = policy.model.log_std.detach().cpu().numpy()
                    print(f'[SUCCESS] Found log_std in policy {policy_id}: mean={log_std.mean():.4f}')
                    
        # Try to extract from rl_module (new API stack)
        if rl_module:
            print(f'[DEBUG] RL module type: {type(rl_module)}')
            print(f'[DEBUG] RL module attributes: {dir(rl_module)}')
            
            # Check if it's a multi-agent RL module
            if hasattr(rl_module, 'keys'):
                for module_id in rl_module.keys():
                    sub_module = rl_module[module_id]
                    print(f'[DEBUG] Sub-module {module_id}: {type(sub_module)}')
                    
                    if hasattr(sub_module, 'log_std'):
                        log_std = sub_module.log_std.detach().cpu().numpy()
                        print(f'[SUCCESS] Found log_std in RL module {module_id}: mean={log_std.mean():.4f}')
            elif hasattr(rl_module, 'log_std'):
                log_std = rl_module.log_std.detach().cpu().numpy()
                print(f'[SUCCESS] Found log_std in RL module: mean={log_std.mean():.4f}')


class ScanLoggingCallback(RLlibCallback):
    """
    Custom callback that extracts scan images and adds them to result dict.
    
    The metrics logger will handle the actual wandb logging.
    """
    
    def __init__(self, scan_log_interval: int):
        """
        Initialize the scan logging callback.
        
        Args:
            scan_log_interval: Time interval in seconds between scan logs
        """
        super().__init__()
        self.scan_log_interval = scan_log_interval
        self.last_log_time = time.time()
        self.episode_count = 0
    
    def on_episode_end(self, *, episode, **kwargs):
        """
        Extract scan images and add to result dict for wandb logging.
        """
        self.episode_count += 1
        current_time = time.time()
        
        # Check if it's time to log scans
        if (current_time - self.last_log_time) >= self.scan_log_interval:
            try:
                print('[DEBUG] scan logging episode')
                print(episode)
                # Get observations from the episode
                if hasattr(episode, 'get_observations'):
                    agent_ids = getattr(episode, 'agent_ids', [])
                    if agent_ids:
                        # Get the latest observation from the first agent
                        agent_id = list(agent_ids)[0]
                        agent_obs = episode.get_observations(agent_id=agent_id, indices=-1)
                        
                        # Store scan data in episode custom metrics for wandb logging
                        if isinstance(agent_obs, np.ndarray) and len(agent_obs.shape) == 3:
                            episode.custom_metrics["scan_images"] = agent_obs
                            episode.custom_metrics["scan_episode_count"] = self.episode_count
                            self.last_log_time = current_time
                        
            except Exception as e:
                print(f"Error in scan logging callback: {e}")
