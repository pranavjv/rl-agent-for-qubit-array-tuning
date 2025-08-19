import torch

#from sb3_contrib import RecurrentPPO
from policy import CustomAgentPolicy
from policy import CustomRecurrentPPO

from envs.qarray_2dot_env import QuantumDeviceEnv
from model.voltage_agent import Agent


def main():
    env = QuantumDeviceEnv()
    
    # model = RecurrentPPO(
    #     "CustomAgentPolicy", 
    #     env, 
    #     agent_class=Agent,
    #     agent_kwargs={
    #         'input_channels': 1, 
    #         'action_dim': 2, 
    #         'num_input_voltages': 2,
    #     },
    #     n_steps=128, # rollouts
    #     batch_size=64,
    #     use_wandb=False,
    # )

    model = CustomRecurrentPPO(
        agent_class=Agent,
        agent_kwargs={
            'input_channels': 1,
            'action_dim': 2,
            'num_input_voltages': 2,
        },
        policy=CustomAgentPolicy,
        env=env,
        use_wandb=True,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=1e-3,
        vf_coef=5e-4,
        gae_lambda=0.95,
    )
    
    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
    )

    torch.save({
        "model": model.policy.agent.state_dict(),
        "optimizer": model.policy.optimizer.state_dict(),
    }, "ppo_ckpt_0.pth")


if __name__ == '__main__':
    main()