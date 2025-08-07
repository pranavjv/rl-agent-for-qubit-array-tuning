import numpy as np

# read in the data
def read_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    voltages = data['voltages']
    states = data['states']
    return voltages, states

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read CSD data from a file")
    parser.add_argument("--file_path", type=str, help="Path to the .npz file containing voltages and states")
    args = parser.parse_args()

    voltages, states = read_data(args.file_path)
    print(voltages.shape)
    print(states.shape)
    print("Voltages:", voltages[0])
    print("States:", states[0])