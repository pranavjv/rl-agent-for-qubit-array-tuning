from envs.qarray_ndot_env import QuantumDeviceEnv
from models.voltage_agent import VoltageAgent



def main():
    env = QuantumDeviceEnv()
    agent = VoltageAgent(env.observation_space, env.action_space)


if __name__ == '__main__':
    main()