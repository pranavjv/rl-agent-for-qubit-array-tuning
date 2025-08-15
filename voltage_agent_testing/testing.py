import torch

from policy import PPO
from envs.qarray_2dot_env import QuantumDeviceEnv
from model.voltage_agent import Agent


def main():
    env = QuantumDeviceEnv()
    
    model = PPO(
        "CustomAgentPolicy", 
        env, 
        agent_class=Agent,
        agent_kwargs={
            'input_channels': 1, 
            'action_dim': 2, 
            'num_input_voltages': 2
        },
        n_steps=128,
    )
    
    print('Training ...')
    model.learn(1000)

if __name__ == '__main__':
    main()