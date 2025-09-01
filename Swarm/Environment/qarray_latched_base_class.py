"""
This file contains the QarrayBaseClass which is used to create a quantum dot array simulator.
"""

import numpy as np
import yaml
import os
from qarray_latched.DotArrays.TunnelCoupledChargeSensed import TunnelCoupledChargeSensed

from qarray import WhiteNoise, TelegraphNoise, LatchingModel
from qarray_latched.DotArrays.barrier_voltage_model import BarrierVoltageModel
from qarray_latched.DotArrays.voltage_dependent_capacitance import create_linear_capacitance_model
import jax.numpy as jnp


# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#NOTE: gates are zero indexed but qarray is one indexed



class QarrayBaseClass:
 
    def __init__(self, num_dots, config_path='qarray_latched_config.yaml', obs_voltage_min=-0.5, obs_voltage_max=0.5, obs_image_size=128, debug=False, **kwargs):

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.num_dots = num_dots
        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1
        
        self.obs_image_size = obs_image_size
        self.obs_channels = self.num_dots - 1
        
        # Set voltage scanning range for observations
        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max

        self.optimal_tc = self.config['simulator']['measurement']['tc']
        optimal_center_dots = self.config['simulator']['measurement']['optimal_VG_center']['dots']
        optimal_center_sensor = self.config['simulator']['measurement']['optimal_VG_center']['sensor']
        self.optimal_VG_center = [optimal_center_dots] * num_dots + [optimal_center_sensor] # must always call model.optimal_Vg on this


        self.barrier_alpha = None
        self.barrier_tc_base = None
        # --- Initialize Model ---
        self.model = self._load_model()


    def _get_charge_sensor_data(self, voltage1, voltage2, gate1, gate2, barrier_voltages):
        """
        Get charge sensor data for given voltages using TunnelCoupledChargeSensed interface.
        
        Args:
            voltage1 (float): Center voltage for first gate
            voltage2 (float): Center voltage for second gate  
            gate1 (int): First gate index (1-indexed for qarray)
            gate2 (int): Second gate index (1-indexed for qarray)
            barrier_voltages (np.ndarray): Barrier voltage array
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width)
        """
        
        # Generate 2D virtual gate grid using gate_voltage_composer
        vg = self.model.gate_voltage_composer.do2d(
            f'P{gate1}', voltage1 + self.obs_voltage_min, voltage1 + self.obs_voltage_max, self.obs_image_size,
            f'P{gate2}', voltage2 + self.obs_voltage_min, voltage2 + self.obs_voltage_max, self.obs_image_size
        )
        
        # Flatten the voltage grid to shape (obs_image_size^2, n_gates)
        vg_flat = vg.reshape(-1, vg.shape[-1])
        
        # Create barrier voltage array (same shape as flattened gate voltage array)
        vb = jnp.full((vg_flat.shape[0], self.num_barrier_voltages), jnp.array(barrier_voltages))
        
        # Get charge sensor data using the new interface
        z, _ = self.model.charge_sensor_open(vg_flat, vb)
        
        # Reshape to 2D image
        z = z.reshape(self.obs_image_size, self.obs_image_size)
        
        return z


    def _get_obs(self, gate_voltages, barrier_voltages):
        """
        Helper method to get the current observation of the environment.
        
        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """

        assert len(gate_voltages) == self.num_dots, f"Incorrect gate voltage shape, expected {self.num_dots}, got {len(gate_voltages)}"
        assert len(barrier_voltages) == self.num_dots - 1, f"Incorrect barrier voltage shape, expected {self.num_dots - 1}, got {len(barrier_voltages)}"

        allgates = list(range(1, self.num_dots+1))  # Gate numbers for qarray (1-indexed)
        all_z = []
        for i, (gate1, gate2) in enumerate(zip(allgates[:-1], allgates[1:])):
            voltage1 = gate_voltages[i]      # Use 0-based indexing for gate_voltages array
            voltage2 = gate_voltages[i+1]    # Use 0-based indexing for gate_voltages array
            z = self._get_charge_sensor_data(voltage1, voltage2, gate1, gate2, barrier_voltages)
            all_z.append(z)  # z is now 2D, no need to index [:, :, 0]

        # Stack images along the channel dimension
        all_images = np.stack(all_z, axis=-1)
            
        # Validate observation structure
        expected_image_shape = (self.obs_image_size, self.obs_image_size, self.obs_channels)

        if all_images.shape != expected_image_shape:
            raise ValueError(f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}")

        return {
            "image": all_images, # unnormalised image
            "obs_gate_voltages": gate_voltages,
            "obs_barrier_voltages": barrier_voltages
        }
    

    def _sample_from_range(self, range_config: dict, rng: np.random.Generator) -> float:
        """Sample a random value from a min-max range configuration."""
        return rng.uniform(range_config["min"], range_config["max"])
    
    def _get_coupling_by_distance(self, coupling_config: dict, distance: int) -> dict:
        """Get coupling range configuration based on distance between elements."""
        if distance == 1:
            return coupling_config[1]
        elif distance == 2:
            return coupling_config[2]
        else:
            return coupling_config["3_plus"]
    
    def _create_symmetric_matrix(self, size: int, fill_func) -> np.ndarray:
        """Create a symmetric matrix by filling upper triangle and mirroring."""
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                value = fill_func(i, j)
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix
    
    def _generate_cdd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate dot-to-dot capacitance matrix with distance-based coupling."""
        cdd_config = config_ranges["Cdd"]
        diagonal_val = cdd_config["diagonal"]
        distance_coupling = cdd_config["distance_coupling"]
        
        def fill_cdd(i: int, j: int) -> float:
            distance = abs(i - j)
            if distance == 0:
                return diagonal_val
            else:
                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                return self._sample_from_range(coupling_range, rng)
        
        return self._create_symmetric_matrix(self.num_dots, fill_cdd)
    
    def _generate_cgd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate gate-to-dot capacitance matrix with distance-based coupling."""
        cgd_config = config_ranges["Cgd"]
        num_gates = self.num_dots + 1  # plunger gates + sensor gate
        
        Cgd = np.zeros((self.num_dots, num_gates))
        
        # Fill plunger gate couplings
        for dot_i in range(self.num_dots):
            for gate_j in range(self.num_dots):
                distance = abs(dot_i - gate_j)
                if distance == 0:
                    # Primary coupling
                    coupling_range = cgd_config["primary_coupling"]
                else:
                    # Cross coupling
                    coupling_range = self._get_coupling_by_distance(
                        cgd_config["cross_coupling"], distance
                    )
                Cgd[dot_i, gate_j] = self._sample_from_range(coupling_range, rng)
        
        # Make plunger gate submatrix symmetric
        for i in range(self.num_dots):
            for j in range(i + 1, self.num_dots):
                avg_coupling = (Cgd[i, j] + Cgd[j, i]) / 2
                Cgd[i, j] = avg_coupling
                Cgd[j, i] = avg_coupling
        
        # Fill sensor gate couplings (last column)
        sensor_coupling = cgd_config["sensor_coupling"]
        for dot_i in range(self.num_dots):
            Cgd[dot_i, self.num_dots] = self._sample_from_range(sensor_coupling, rng)
        
        return Cgd
    
    def _generate_cbd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-dot capacitance matrix with distance-based coupling."""
        cbd_config = config_ranges["Cbd"]
        distance_coupling = cbd_config["distance_coupling"]
        
        # Cbd is (n_dot, n_barrier) matrix
        Cbd = np.zeros((self.num_dots, self.num_barrier_voltages))
        
        for dot_i in range(self.num_dots):
            for barrier_j in range(self.num_barrier_voltages):
                # Calculate distance between dot and barrier
                # Barriers are typically between adjacent dots
                barrier_center = barrier_j + 0.5  # Barrier j is between dots j and j+1
                distance = int(abs(dot_i - barrier_center))
                distance = max(1, distance)  # Minimum distance is 1
                
                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                Cbd[dot_i, barrier_j] = self._sample_from_range(coupling_range, rng)
        
        return Cbd
    
    def _generate_cbg_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-gate capacitance matrix with distance-based coupling."""
        cbg_config = config_ranges["Cbg"]
        distance_coupling = cbg_config["distance_coupling"]
        num_gates = self.num_dots + 1  # plunger gates + sensor gate
        
        # Cbg is (n_barrier, n_gate) matrix
        Cbg = np.zeros((self.num_barrier_voltages, num_gates))
        
        for barrier_i in range(self.num_barrier_voltages):
            for gate_j in range(num_gates):
                if gate_j < self.num_dots:  # Plunger gates
                    # Calculate distance between barrier and gate
                    barrier_center = barrier_i + 0.5  # Barrier i is between dots i and i+1
                    distance = int(abs(barrier_center - gate_j))
                    distance = max(1, distance)  # Minimum distance is 1
                    
                    coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                    Cbg[barrier_i, gate_j] = self._sample_from_range(coupling_range, rng)
                else:  # Sensor gate
                    # Use distance 2 coupling for sensor gate
                    coupling_range = self._get_coupling_by_distance(distance_coupling, 2)
                    Cbg[barrier_i, gate_j] = self._sample_from_range(coupling_range, rng)
        
        return Cbg
    
    def _generate_cbs_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-sensor capacitance matrix."""
        cbs_config = config_ranges["Cbs"]
        coupling_range = cbs_config["coupling"]
        
        # Cbs is (n_sensor, n_barrier) matrix - typically (1, n_barrier)
        Cbs = np.zeros((1, self.num_barrier_voltages))
        
        for barrier_j in range(self.num_barrier_voltages):
            Cbs[0, barrier_j] = self._sample_from_range(coupling_range, rng)
        
        return Cbs
    
    def _generate_cbb_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-barrier capacitance matrix with distance-based coupling."""
        cbb_config = config_ranges["Cbb"]
        diagonal_val = cbb_config["diagonal"]
        distance_coupling = cbb_config["distance_coupling"]
        
        def fill_cbb(i: int, j: int) -> float:
            distance = abs(i - j)
            if distance == 0:
                return diagonal_val
            else:
                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                return self._sample_from_range(coupling_range, rng)
        
        return self._create_symmetric_matrix(self.num_barrier_voltages, fill_cbb)
    
    def _generate_sensor_capacitances(self, config_ranges: dict, rng: np.random.Generator) -> tuple:
        """Generate dot-to-sensor (Cds) and gate-to-sensor (Cgs) capacitances."""
        # Cds: dot-to-sensor capacitances
        cds_config = config_ranges["Cds"]["dots"]
        Cds = [[self._sample_from_range(cds_config, rng) for _ in range(self.num_dots)]]
        
        # Cgs: gate-to-sensor capacitances
        cgs_plunger = config_ranges["Cgs"]["plunger_gates"]
        cgs_sensor = config_ranges["Cgs"]["sensor_gate"]
        
        cgs_values = [self._sample_from_range(cgs_plunger, rng) for _ in range(self.num_dots)]
        cgs_values.append(self._sample_from_range(cgs_sensor, rng))
        Cgs = [cgs_values]
        
        return Cds, Cgs
    
    def _generate_noise_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate white noise and telegraph noise parameters."""
        # White noise
        white_noise_amp = self._sample_from_range(config_ranges["white_noise_amplitude"], rng)
        
        # Telegraph noise
        telegraph_config = config_ranges["telegraph_noise_parameters"]
        p01 = self._sample_from_range(telegraph_config["p01"], rng)
        p10_factor = self._sample_from_range(telegraph_config["p10_factor"], rng)
        amplitude = self._sample_from_range(telegraph_config["amplitude"], rng)
        
        return {
            "white_noise_amplitude": white_noise_amp,
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": p10_factor * p01,
                "amplitude": amplitude,
            }
        }
    
    def _generate_latching_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate symmetric latching model parameters."""
        latching_config = config_ranges["latching_model_parameters"]
        
        # Generate p_inter matrix (symmetric, zero diagonal)
        p_inter_range = latching_config["p_inter"]
        
        def fill_p_inter(i: int, j: int) -> float:
            if i == j:
                return 0.0  # No self-interaction
            else:
                return self._sample_from_range(p_inter_range, rng)
        
        p_inter = self._create_symmetric_matrix(self.num_dots, fill_p_inter)
        
        # Generate p_leads array
        p_leads_range = latching_config["p_leads"]
        p_leads = [self._sample_from_range(p_leads_range, rng) for _ in range(self.num_dots)]
        
        return {
            "Exists": True,
            "n_dots": self.num_dots,
            "p_leads": p_leads,
            "p_inter": p_inter,
        }
    
    def _generate_barrier_model_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate barrier voltage model parameters."""
        barrier_config = config_ranges["barrier_model"]
        
        tc_base = self._sample_from_range(barrier_config["tc_base"], rng)
        # Generate alpha array with one value per barrier
        alpha = [self._sample_from_range(barrier_config["alpha_per_barrier"], rng) 
                for _ in range(self.num_barrier_voltages)]
        
        return {
            "tc_base": tc_base,
            "alpha": alpha
        }
    
    def _generate_voltage_capacitance_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate voltage-dependent capacitance model parameters."""
        vc_config = config_ranges["voltage_capacitance_model"]
        
        return {
            "alpha": self._sample_from_range(vc_config["alpha"], rng),
            "beta": self._sample_from_range(vc_config["beta"], rng)
        }

    def _gen_random_qarray_params(self, rng: np.random.Generator = None) -> dict:
        """
        Generate random parameters for the quantum device using distance-based coupling rules.
        
        Returns:
            dict: Complete model parameters for qarray simulator
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract configuration ranges
        model_config = self.config['simulator']['model']
        measurement_config = self.config['simulator']['measurement']
        
        config_ranges = {
            'Cdd': model_config['Cdd'],
            'Cgd': model_config['Cgd'], 
            'Cds': model_config['Cds'],
            'Cgs': model_config['Cgs'],
            'Cbd': model_config['Cbd'],
            'Cbg': model_config['Cbg'],
            'Cbs': model_config['Cbs'],
            'Cbb': model_config['Cbb'],
            'white_noise_amplitude': model_config['white_noise_amplitude'],
            'telegraph_noise_parameters': model_config['telegraph_noise_parameters'],
            'latching_model_parameters': model_config['latching_model_parameters'],
            'barrier_model': model_config['barrier_model'],
            'voltage_capacitance_model': model_config['voltage_capacitance_model'],
            'T': model_config['T'],
            'coulomb_peak_width': model_config['coulomb_peak_width'],
            'tc': model_config['tc']
        }

        # Generate all matrix components using helper methods
        Cdd = self._generate_cdd_matrix(config_ranges, rng)
        Cgd = self._generate_cgd_matrix(config_ranges, rng)
        Cds, Cgs = self._generate_sensor_capacitances(config_ranges, rng)
        
        # Generate barrier matrices
        Cbd = self._generate_cbd_matrix(config_ranges, rng)
        Cbg = self._generate_cbg_matrix(config_ranges, rng)
        Cbs = self._generate_cbs_matrix(config_ranges, rng)
        Cbb = self._generate_cbb_matrix(config_ranges, rng)
        
        # Generate model parameters
        noise_params = self._generate_noise_parameters(config_ranges, rng)
        latching_params = self._generate_latching_parameters(config_ranges, rng)
        barrier_model_params = self._generate_barrier_model_parameters(config_ranges, rng)
        voltage_capacitance_params = self._generate_voltage_capacitance_parameters(config_ranges, rng)

        # Assemble final model parameters
        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "Cbd": Cbd,
            "Cbg": Cbg,
            "Cbs": Cbs,
            "Cbb": Cbb,
            "white_noise_amplitude": noise_params["white_noise_amplitude"],
            "telegraph_noise_parameters": noise_params["telegraph_noise_parameters"],
            "latching_model_parameters": latching_params,
            "barrier_model_parameters": barrier_model_params,
            "voltage_capacitance_parameters": voltage_capacitance_params,
            "T": self._sample_from_range(config_ranges['T'], rng),
            "coulomb_peak_width": self._sample_from_range(config_ranges['coulomb_peak_width'], rng),
            "tc": self._sample_from_range(config_ranges['tc'], rng),
            "algorithm": model_config['algorithm'],
            "implementation": model_config['implementation'],
            "max_charge_carriers": model_config['max_charge_carriers'],
            "optimal_VG_center": measurement_config['optimal_VG_center']
        }
        
        return model_params


    def _load_model(self):
        """
        Load the TunnelCoupledChargeSensed model from the config file.
        """

        model_params = self._gen_random_qarray_params()

        # Extract noise models
        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        
        # Extract latching model
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None

        # Extract capacitance matrices
        Cdd = model_params['Cdd']
        Cgd = model_params['Cgd']
        Cds = model_params['Cds']
        Cgs = model_params['Cgs']
        Cbd = model_params['Cbd']
        Cbg = model_params['Cbg']
        Cbs = model_params['Cbs']
        Cbb = model_params['Cbb']

        # Create barrier voltage model
        barrier_params = model_params['barrier_model_parameters']
        barrier_model = BarrierVoltageModel(
            n_barrier=self.num_barrier_voltages, 
            n_dot=self.num_dots,
            tc_base=barrier_params['tc_base'], 
            alpha=barrier_params['alpha']
        )

        self.barrier_alpha = barrier_params['alpha']
        self.barrier_tc_base = barrier_params['tc_base']

        # Create TunnelCoupledChargeSensed model
        model = TunnelCoupledChargeSensed(
            Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
            Cbd=Cbd, Cbg=Cbg, Cbs=Cbs, Cbb=Cbb,
            barrier_model=barrier_model,
            coulomb_peak_width=model_params['coulomb_peak_width'], 
            T=model_params['T'], 
            max_charge_carriers=model_params['max_charge_carriers'], 
            tc=model_params['tc'],
            noise_model=noise_model, 
            latching_model=latching_model,
            voltage_capacitance_model=None  # Will be set below
        )

        # Create and set voltage-dependent capacitance model
        vc_params = model_params['voltage_capacitance_parameters']
        voltage_capacitance_model = create_linear_capacitance_model(
            cdd_0=jnp.array(model.cdd), 
            cgd_0=jnp.array(model.cgd),
            alpha=vc_params['alpha'],
            beta=vc_params['beta']
        )
        model.voltage_capacitance_model = voltage_capacitance_model

        return model


    def _update_virtual_gate_matrix(self, cgd_estimate):
        vgm = -np.linalg.pinv(np.linalg.inv(self.model.Cdd) @ cgd_estimate)  
    
        self.model.gate_voltage_composer.virtual_gate_matrix = vgm


    def _render_frame(self, image):
        """
        Internal method to create the render image.

        Returns the CSD scan between gate1 and the next gate
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        z = image

        vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)

        plt.figure(figsize=(5, 5))
        plt.imshow(z, extent=[vmin, vmax, vmin, vmax], origin='lower', aspect='auto', cmap='viridis')
        plt.xlabel("$\\Delta$PL (V)")
        plt.ylabel("$\\Delta$PR (V)")
        plt.title("$|S_{11}|$ (Charge Stability Diagram)")
        plt.axis('equal')

        plt.savefig("quantum_dot_plot.png")
        plt.show()
        plt.close()


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
    
    def calculate_ground_truth(self):
        """
        Get the ground truth for the quantum dot array.
        """
        ndot = self.num_dots
        nbarrier = self.num_barrier_voltages
        cgd = self.model.cgd_full
        cdd = self.model.cdd_full
        cdd_inv = self.model.cdd_inv_full

        numsensor = 1


        #Assumes Vb = 0
        vg_optimal_physical = self.model.optimal_Vg(self.optimal_VG_center)

        #Assumes perfectly virtualised barriers (this is not the case but since ground truth for barriers is approximate we leave this)
        tc_ratio = float(self.optimal_tc) / self.barrier_tc_base

        print(tc_ratio)
        vb_mag = -np.log(tc_ratio) / self.model.barrier_model.alpha
        vb_optimal = np.full(self.num_barrier_voltages, vb_mag)

        #calculate change in dot potential due to barriers
        dot_potential = cgd[:ndot+numsensor, -nbarrier:] @ vb_optimal

        delta_vg = np.linalg.inv(cgd[:ndot+numsensor, :ndot+numsensor]) @ dot_potential

        vg_optimal_physical -= delta_vg
        sensor_optimal_physical = vg_optimal_physical[-numsensor:]
        plunger_optimal_physical = vg_optimal_physical[:-numsensor]


        #note ignoring virtual gate origins here
        plunger_optimal_virtual = -cdd_inv[:ndot, :ndot] @ cgd[:ndot, :ndot] @ plunger_optimal_physical

        return plunger_optimal_virtual, vb_optimal, sensor_optimal_physical




if __name__ == "__main__":
    experiment = QarrayBaseClass(num_dots=3, obs_image_size=50)

    # Test optimal voltage calculation
    gt_vg, gt_vb, gt_vs = experiment.calculate_ground_truth()
    print(f"Optimal voltages: {gt_vg}, {gt_vs}, {gt_vb}")
    
    # Test getting observations
    gate_voltages = [0.0] * experiment.num_dots
    barrier_voltages = [4.0] * experiment.num_barrier_voltages

    obs = experiment._get_obs(gt_vg, gt_vb)
    print("Observation shape:", obs["image"].shape)
    print("Gate voltages shape:", len(obs["obs_gate_voltages"]))
    print("Barrier voltages shape:", len(obs["obs_barrier_voltages"]))

    print(obs["image"].shape)
    experiment._render_frame(obs["image"][:,:,0])
