import torch

#from policy import RecurrentPPO
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
        #n_steps=2048,
    )
    
    print('Learning ...')
    model.learn(
        total_timesteps=1000,
        progress_bar=True,
    )

if __name__ == '__main__':
    main()