from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomLoggingCallback(BaseCallback):
    """
    Custom callback to add additional metrics to the training logs for stable_baselines3
    """
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # This method is called at every step
        # You can add custom metrics here that get logged
        
        # Example: Log custom metrics every few steps
        if self.n_calls % 1000 == 0:
            # Add custom fields to the logger
            self.logger.record("custom/step_count", self.n_calls)
            self.logger.record("custom/learning_progress", self.n_calls / self.model.num_timesteps * 100)
            
            # Log accuracy from environment
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get accuracy from all environments and average them
                    accuracies = self.training_env.get_attr('get_accuracy')
                    if accuracies:
                        # Call the get_accuracy method for each environment
                        accuracy_values = [env_accuracy() if callable(env_accuracy) else env_accuracy for env_accuracy in accuracies]
                        avg_accuracy = np.mean(accuracy_values)
                        self.logger.record("custom/accuracy_percent", avg_accuracy)
                except:
                    pass
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout
        """
        # Add rollout-specific metrics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            
            if ep_lengths:
                self.logger.record("custom/mean_ep_length", np.mean(ep_lengths))
                self.logger.record("custom/max_ep_length", np.max(ep_lengths))
                
            if ep_rewards:
                self.logger.record("custom/mean_ep_reward", np.mean(ep_rewards))
                self.logger.record("custom/std_ep_reward", np.std(ep_rewards))
