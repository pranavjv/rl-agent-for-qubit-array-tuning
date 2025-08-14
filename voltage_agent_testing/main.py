import torch

from envs.qarray_ndot_env import QuantumDeviceEnv
from model.voltage_agent import Agent


def main():
    num_dots = 8
    env = QuantumDeviceEnv(ndots=num_dots)
    plunger_agent = Agent(input_channels=1)

    obs, _ = env.reset()
    obs = obs['image']
    actions = []
    for i in range(num_dots-1):
        frame = obs[:,:,i:i+1]
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        action, _ = plunger_agent(frame)
        action = action.detach().numpy()
        actions.append(action)

if __name__ == '__main__':
    main()