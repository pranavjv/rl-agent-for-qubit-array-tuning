import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import io
import fcntl
import json
from torch.nn.functional import sigmoid

# For block diagonal matrix construction
from scipy.linalg import block_diag

"""
todo

fix n-dot random param setup

add barrier voltages in _get_obs
"""

class QarrayBaseClass:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_dots, config_path='qarray_config.yaml', randomise_actions=True, debug=False, **kwargs):

        print(f'Initialising qarray env with {num_dots} dots ...')

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.debug = self.config['init']['debug'] or debug
        self.seed = self.config['init']['seed']

        self.num_dots = num_dots
        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1
        
        self.obs_image_size = self.config['simulator']['measurement']['resolution']
        self.obs_channels = self.num_dots - 1

        optimal_center_dots = self.config['simulator']['measurement']['optimal_VG_center']['dots']
        optimal_center_sensor = self.config['simulator']['measurement']['optimal_VG_center']['sensor']
        self.optimal_VG_center = [optimal_center_dots] * num_dots + [optimal_center_sensor] # must always call model.optimal_Vg on this

        # --- Initialize Model ---
        self.model = self._load_model()


    def _get_charge_sensor_data(self, voltage1, voltage2, gate1, gate2):
        """
        Get charge sensor data for given voltages.
        
        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """

        z, _ = self.model.do2d_open(
            gate1, voltage1 + self.obs_voltage_min, voltage1 + self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
            gate2, voltage2 + self.obs_voltage_min, voltage2 + self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        )
        return z


    def _get_obs(self, gate_voltages, barrier_voltages):
        """
        Helper method to get the current observation of the environment.
        
        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """
        # TODO we are currently not using the barrier voltages

        assert len(gate_voltages) == self.num_dots, f"Incorrect gate voltage shape, expected {self.num_dots}, got {len(gate_voltages)}"
        assert len(barrier_voltages) == self.num_dots - 1, f"Incorrect barrier voltage shape, expected {self.num_dots - 1}, got {len(barrier_voltages)}"

        allgates = list(range(1, self.num_dots+1))
        all_z = []
        for (gate1, gate2) in zip(allgates[:-1], allgates[1:]):
            voltage1 = gate_voltages[gate1]
            voltage2 = gate_voltages[gate2]
            z = self._get_charge_sensor_data(voltage1, voltage2, gate1, gate2)
            all_z.append(z)


        all_images = []

        for z in all_z:
            # Extract first channel
            channel_data = z[:, :, 0]  # Shape: (height, width)
            
            all_images.append(image_obs)

        all_images = np.concatenate(all_images, axis=-1)
            
        # Validate observation structure
        expected_image_shape = (self.obs_image_size, self.obs_image_size, self.obs_channels)

        if all_images.shape != expected_image_shape:
            raise ValueError(f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}")

        return {
            "image": all_images, # unnormalised image
            "obs_gate_voltages": gate_voltages,
            "obs_barrier_voltages": barrier_voltages
        }
    

    def _gen_random_qarray_params(self, rng: np.random.Generator = None):
        """
        Generate random parameters for the quantum device.
        """
        model_config = self.config['simulator']['model']
        measurement_config = self.config['simulator']['measurement']

        if rng is None:
            rng = np.random.default_rng()

        rb = {
            'Cdd': model_config['Cdd'],
            'Cgd': model_config['Cgd'],
            'Cds': model_config['Cds'],
            'Cgs': model_config['Cgs'],
            'white_noise_amplitude': model_config['white_noise_amplitude'],
            'telegraph_noise_parameters': model_config['telegraph_noise_parameters'],
            'latching_model_parameters': model_config['latching_model_parameters'],
            'T': model_config['T'],
            'coulomb_peak_width': model_config['coulomb_peak_width'],
            'algorithm': model_config['algorithm'],
            'implementation': model_config['implementation'],
            'max_charge_carriers': model_config['max_charge_carriers'],
            'sensor_gate_voltage': measurement_config['sensor_gate_voltage'],
            'optimal_VG_center': measurement_config['optimal_VG_center']  # Add this for optimal voltage calculation
        }

        latching = True

        cdd_diag = rb["Cdd"]["diagonal"]
        cdd_nn = rng.uniform(rb["Cdd"]["nearest_neighbor"]["min"], 
                            rb["Cdd"]["nearest_neighbor"]["max"])
        cdd_next = rng.uniform(rb["Cdd"]["next_nearest"]["min"], 
                              rb["Cdd"]["next_nearest"]["max"])  
        cdd_far = rng.uniform(rb["Cdd"]["furthest"]["min"], 
                             rb["Cdd"]["furthest"]["max"])
        

        # Create the 4x4 Cdd block
        Cdd_block = [
            [cdd_diag, cdd_nn, cdd_next, cdd_far],
            [cdd_nn, cdd_diag, cdd_nn, cdd_next],
            [cdd_next, cdd_nn, cdd_diag, cdd_nn],
            [cdd_far, cdd_next, cdd_nn, cdd_diag]
        ]

        # Make Cdd a block diagonal matrix of shape (self.num_dots, self.num_dots)
        num_blocks = self.num_dots // 4
        Cdd = block_diag(*([Cdd_block] * num_blocks))


        # Create the 4x5 Cgd block
        Cgd_block = [[0.0 for _ in range(5)] for _ in range(4)]
        # Fill diagonal (primary couplings)
        for i in range(4):
            Cgd_block[i][i] = rng.uniform(rb["Cgd"][i][i]["min"], rb["Cgd"][i][i]["max"])
        # Fill symmetric cross-couplings for plunger gates (gates 0-3)
        for i in range(4):
            for j in range(4):
                if i < j:  # Only fill upper triangle, then mirror
                    coupling = rng.uniform(rb["Cgd"][i][j]["min"], rb["Cgd"][i][j]["max"])
                    Cgd_block[i][j] = coupling
                    Cgd_block[j][i] = coupling
        # Fill sensor gate couplings (gate 4)
        for i in range(4):
            Cgd_block[i][4] = rng.uniform(rb["Cgd"][i][4]["min"], rb["Cgd"][i][4]["max"])

        # Make Cgd a block diagonal matrix of shape (self.num_dots, self.num_dots+1)
        num_blocks = self.num_dots // 4
        # block_diag pads with zeros to the right/bottom if blocks are not square, so we need to trim the result
        Cgd_full = block_diag(*([Cgd_block] * num_blocks))
        # Cgd_full will be (self.num_dots, num_blocks*5), but we want (self.num_dots, self.num_dots+1)
        Cgd = Cgd_full[:, :self.num_dots+1]

        Cds = [[rng.uniform(rb["Cds"]["dots"]["min"], rb["Cds"]["dots"]["max"]) for i in range(self.num_dots)]]
        Cgs = [[rng.uniform(rb["Cgs"]["dots"]["min"], rb["Cgs"]["dots"]["max"]) for i in range(self.num_dots)] + [rng.uniform(rb["Cgs"]["sensor"]["min"], rb["Cgs"]["sensor"]["max"])]]

        # Generate 4x4 p_inter matrix for latching model (must be symmetric)
        p_inter = [[0.0 for _ in range(4)] for _ in range(4)]  # Initialize with zeros
        
        # Fill upper triangle and mirror to lower triangle for symmetry
        for i in range(4):
            for j in range(i+1, 4):  # Only fill upper triangle
                coupling = rng.uniform(rb["latching_model_parameters"]["p_inter"]["min"],
                                     rb["latching_model_parameters"]["p_inter"]["max"])
                p_inter[i][j] = coupling
                p_inter[j][i] = coupling  # Ensure symmetry
        # Diagonal elements remain 0.0 (no self-interaction)
        
        # Generate 4-element p_leads array
        p_leads = [rng.uniform(rb["latching_model_parameters"]["p_leads"]["min"],
                              rb["latching_model_parameters"]["p_leads"]["max"]) 
                   for _ in range(4)]
        
        p01 = rng.uniform(rb["telegraph_noise_parameters"]["p01"]["min"], 
                         rb["telegraph_noise_parameters"]["p01"]["max"])
        

        # # Generate fixed gate voltages for outer gates (these will be replaced by optimal voltages later)
        # fixed_gate_voltages = {}
        # for gate_idx in rb["fixed_gates"]:
        #     if gate_idx == 4:  # Sensor gate (will be replaced with optimal)
        #         fixed_gate_voltages[gate_idx] = rng.uniform(
        #             rb["sensor_gate_voltage"]["min"], 
        #             rb["sensor_gate_voltage"]["max"]
        #         )
        #     else:  # Plunger gates (will be replaced with optimal)
        #         fixed_gate_voltages[gate_idx] = rng.uniform(
        #             rb["fixed_gate_voltages"]["min"], 
        #             rb["fixed_gate_voltages"]["max"]
        #         )
        

        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "white_noise_amplitude": rng.uniform(rb["white_noise_amplitude"]["min"], 
                                               rb["white_noise_amplitude"]["max"]),
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": rng.uniform(rb["telegraph_noise_parameters"]["p10_factor"]["min"], 
                                 rb["telegraph_noise_parameters"]["p10_factor"]["max"]) * p01,
                "amplitude": rng.uniform(rb["telegraph_noise_parameters"]["amplitude"]["min"], 
                                       rb["telegraph_noise_parameters"]["amplitude"]["max"]),
            },
            "latching_model_parameters": {
                "Exists": latching,
                "n_dots": 4,
                "p_leads": p_leads,
                "p_inter": p_inter,
            },
            "T": rng.uniform(rb["T"]["min"], rb["T"]["max"]),
            "coulomb_peak_width": rng.uniform(rb["coulomb_peak_width"]["min"], 
                                            rb["coulomb_peak_width"]["max"]),
            "algorithm": rb["algorithm"],
            "implementation": rb["implementation"],
            "max_charge_carriers": rb["max_charge_carriers"],
            "optimal_VG_center": rb["optimal_VG_center"]
        }
        
        return model_params


    def _load_model(self):
        """
        Load the model from the config file.
        """

        model_params = self._gen_random_qarray_params()

        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None

        Cdd_base = model_params['Cdd']
        Cgd_base = model_params['Cgd']
        Cds_base = model_params['Cds']
        Cgs_base = model_params['Cgs']

        model_mats = []
        for mat in [Cdd_base, Cgd_base]:
            block_size = np.array(mat).shape[0]
            num_blocks = self.num_dots // block_size
            out_mat = block_diag(*([mat]*num_blocks))
            model_mats.append(out_mat)

        Cdd, Cgd = model_mats
        Cds = [np.array(Cds_base).flatten().tolist() * (self.num_dots // 4)]
        Cgs = [np.array(Cgs_base).flatten().tolist() * (self.num_dots // 4)]

        # print(np.array(Cdd).shape)
        # print(np.array(Cgd).shape)
        # print(np.array(Cds).shape)
        # print(np.array(Cgs).shape)

        model = ChargeSensedDotArray(
            Cdd=Cdd,
            Cgd=Cgd,
            Cds=Cds,
            Cgs=Cgs,
            coulomb_peak_width=model_params['coulomb_peak_width'],
            T=model_params['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params['algorithm'],
            implementation=model_params['implementation'],
            max_charge_carriers=model_params['max_charge_carriers'],
        )

        # model.gate_voltage_composer.virtual_gate_matrix = self.config['simulator']['virtual_gate_matrix']
        # TODO update virtual gate matrix

        return model


    def _update_virtual_gate_matrix(self, vgm):
        self.model.virtual_gate_matrix = vgm


    def _render_frame(self, gate1):
        """
        Internal method to create the render image.

        Returns the CSD scan between gate1 and the next gate
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        assert gate1 in range(self.dots - 1)

        z = self.z

        vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)

        plt.figure(figsize=(5, 5))
        plt.imshow(z, extent=[vmin, vmax, vmin, vmax], origin='lower', aspect='auto', cmap='viridis')
        plt.xlabel("$\Delta$PL (V)")
        plt.ylabel("$\Delta$PR (V)")
        plt.title("Normalized $|S_{11}|$ (Agent Observation)")
        plt.axis('equal')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        from PIL import Image
        img = Image.open(buf)
        img = np.array(img)
        return img
 

    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config


if __name__ == "__main__":
    import sys
    env = QarrayBaseClass(num_dots=4, randomise_actions=False)
    env.reset()

    sys.exit(0)

    gt = env.model.optimal_Vg(env.optimal_VG_center)
    print(gt)
    print(env.device_state['voltage_centers'])
    env.step(gt[1:3])
    print(env.device_state['voltage_centers'])
    frame = env._render_frame(inference_plot=True)
    path = "quantum_dot_plot.png"
    plt.imsave(path, frame, cmap='viridis')
    env.close()