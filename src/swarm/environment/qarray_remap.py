import numpy as np
import yaml
import os
import copy

class QarrayRemapper:

    def __init__(
        self,
        model,
        num_dots,
        has_barriers,
        optimal_VG_center,
        obs_voltage_min,
        obs_voltage_max,
        obs_image_size=128,
        config_path="./remap_config.yaml",
        **kwargs
    ):

        self.model = model # inherited from qarray base class, do not modify
        self.has_barriers = has_barriers
        self.num_dots = num_dots

        self.config = self._load_config(config_path)

        reference_mode = self.config["model_config"]["reference_mode"]
        if reference_mode == "zeros":
            self.reference_coords = np.zeros(self.num_dots)
        elif reference_mode == "optimal":
            self.reference_coords = self.model.optimal_Vg(optimal_VG_center)[:-1] # ignore sensor voltage

        self.voltage_bounds = self.reference_coords - 10. # placeholder for now, may depend on the base model params

        self.window_sizes = self.reference_coords - self.voltage_bounds

        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max
        self.obs_image_size = obs_image_size


    def get_remapped_scan(self, gate1, gate2, gate_voltage1, gate_voltage2, barrier_voltage=None):
        """
        Gets a single scan, and returns its corresponding remapped ((gate1, gate2), barrier) voltages
        """
        if barrier_voltage is None:
            assert not self.has_barriers, "Cannot provide barrier voltage, environment is not configured for barriers"
            return self._get_data_no_barrier(gate1, gate2, gate_voltage1, gate_voltage2)
        else:
            assert self.has_barriers, "Expected barrier voltage but received None"
            raise NotImplementedError

    
    def _get_data_no_barrier(self, gate1, gate2, gate_voltage1, gate_voltage2):
        # note gates are 1-indexed
        
        debug = self.config["model_config"]["debug"]

        if (gate_voltage1 > self.voltage_bounds[gate1-1] and gate_voltage2 > self.voltage_bounds[gate2-1]):
            # or (gate_voltage1 > self.reference_coords[gate1-1] or gate_voltage2 > self.reference_coords[gate2-1]):
            # don't remap if either voltage is in the single line regime, or both are within
            if debug:
                print('No remap needed, calling original base model ...')
            z = self._get_charge_sensor_data(self.model, gate_voltage1, gate_voltage2, gate1, gate2)
            return z, (gate_voltage1, gate_voltage2), 0.


        delta1 = self.voltage_bounds[gate1-1] - gate_voltage1
        delta2 = self.voltage_bounds[gate2-1] - gate_voltage2

        if debug:
            print('delta1, delta2: ', delta1, delta2)

        map_number1 = delta1 // self.window_sizes[gate1-1] + 1 if delta1 > 0 else 0
        map_number2 = delta2 // self.window_sizes[gate2-1] + 1 if delta2 > 0 else 0

        if debug:
            print('map_number1, map_number2: ', map_number1, map_number2)

        residual1 = delta1 % self.window_sizes[gate1-1]
        residual2 = delta2 % self.window_sizes[gate2-1]

        if debug:
            print('residual1, residual2: ', residual1, residual2)

        new_gate_voltage1 = self.reference_coords[gate1-1] - residual1 if delta1 > 0 else gate_voltage1
        new_gate_voltage2 = self.reference_coords[gate2-1] - residual2 if delta2 > 0 else gate_voltage2

        if debug:
            print('new_gate_voltage1, new_gate_voltage2: ', new_gate_voltage1, new_gate_voltage2)

        model = self._get_local_qarray(map_number1, map_number2)

        z = self._get_charge_sensor_data(model, new_gate_voltage1, new_gate_voltage2, gate1, gate2)

        return z, (new_gate_voltage1, new_gate_voltage2), 0.


    def _get_local_qarray(self, map_number1, map_number2):
        """
        Creates a local model based on the distance from the ground truth
        adds conductance and diagonal transition lines
        """
        base_Cdd = self.model.Cdd.copy()
        base_Cgd = self.model.Cgd.copy()
        # can schedule other matrices as well
        if self.has_barriers:
            base_Cbd = self.model.base_Cbd.copy()
            base_Cbg = self.model.base_Cbg.copy()

        # increase noise, etc.

        local_model = copy.deepcopy(self.model)

        radial_map_number = np.sqrt(map_number1**2 + map_number2**2)

        Cdd = base_Cdd * (1 + radial_map_number * self.config["capacitance_params"]["cdd_increase_per_map"])
        Cgd = base_Cgd * (1 + radial_map_number * self.config["capacitance_params"]["cgd_increase_per_map"])

        local_model.Cdd = Cdd
        local_model.Cgd = Cgd

        if self.has_barriers:
            Cbd = base_Cbd * (1 + radial_map_number * self.config["capacitance_params"]["cbd_increase_per_map"])
            Cbg = base_Cbg * (1 + radial_map_number * self.config["capacitance_params"]["cbg_increase_per_map"])

            local_model.Cbd = Cbd
            local_model.Cbg = Cbg

        return local_model


    def _get_charge_sensor_data(self, model, voltage1, voltage2, gate1, gate2):

        z, _ = model.do2d_open(
            gate1,
            voltage1 + self.obs_voltage_min,
            voltage1 + self.obs_voltage_max,
            self.obs_image_size,
            gate2,
            voltage2 + self.obs_voltage_min,
            voltage2 + self.obs_voltage_max,
            self.obs_image_size,
        )
        return z


    def _load_config(self, config_path):
        if not os.path.isabs(config_path):
            # If relative path, look in the same directory as this file
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        try:
            self._validate_config(config)
        except Exception as e:
            raise ValueError(f"Invalid config: {e}")
        return config


    def _validate_config(self, config):
        debug = config["model_config"]["debug"]
        assert isinstance(debug, bool)

        reference_mode = config["model_config"]["reference_mode"]
        assert reference_mode in ["zeros", "optimal"]

        # etc.


