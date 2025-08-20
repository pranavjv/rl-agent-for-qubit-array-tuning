import torch

#from sb3_contrib import RecurrentPPO
#from policy import CustomRecurrentPPO
from ppo_recurrent import RecurrentPPO
from envs.qarray_2dot_env import QuantumDeviceEnv
from model.voltage_agent import Agent
from policy import CustomAgentPolicy

from stable_baselines3.common.env_util import make_vec_env


def main0():
    env = QuantumDeviceEnv()

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
    

def main():
    env = make_vec_env(QuantumDeviceEnv, n_envs=1)
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        use_wandb=True,
        vf_coef=1e-5,
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("recurrent_ppo_v0")


if __name__ == '__main__':
    main()