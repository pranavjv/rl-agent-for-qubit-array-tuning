import numpy as np
import yaml
import os


class QarrayPatched:

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
        debug=False,
        **kwargs
    ):

        self.model = model # inherited from qarray base class, do not modify
        self.has_barriers = has_barriers

        self.voltage_bounds = np.zeros((self.num_dots)) # placeholder for now
        self.ground_truth_coords = self.model.optimal_vg(optimal_VG_center)

        self.window_sizes = self.ground_truth_coords - self.voltage_bounds

        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max
        self.obs_image_size = obs_image_size

        self.debug = debug

        self.config = self._load_config(config_path)


    def get_remapped_scan(self, gate1, gate2, gate_voltage1, gate_voltage2, barrier_voltage=None):
        if barrier_voltage is None:
            assert not self.has_barriers, "Cannot provide barrier voltage, environment is not configured for barriers"
            return self._get_data_no_barrier(gate1, gate2, gate_voltage1, gate_voltage2)
        else:
            assert self.has_barriers, "Expected barrier voltage but received None"
            raise NotImplementedError

    
    def _get_data_no_barrier(self, gate1, gate2, gate_voltage1, gate_voltage2):
        if gate_voltage1 < self.voltage_bounds[gate1-1] and gate_voltage2 < self.voltage_bounds[gate2-1]:
            # nothing to remap
            return self._get_charge_sensor_data(self.model, gate_voltage1, gate_voltage2, gate1, gate2)


        delta1 = self.voltage_bounds[gate1-1] - gate_voltage1
        delta2 = self.voltage_bounds[gate2-1] - gate_voltage2

        map_number1 = delta1 // self.window_sizes[gate1-1] if delta1 > 0 else 0
        map_number2 = delta2 // self.window_sizes[gate2-1] if delta2 > 0 else 0

        residual1 = delta1 % self.window_sizes[gate1-1] if delta1 > 0 else gate_voltage1
        residual2 = delta2 % self.window_sizes[gate2-1] if delta2 > 0 else gate_voltage2

        model = self._get_local_qarray(map_number1, map_number2)

        z = self._get_charge_sensor_data(model, residual1, residual2, gate1, gate2)

        return z, residual1, 0.0


    def _get_local_qarray(self, map_number1, map_number2):
        base_Cdd = self.model.Cdd
        base_Cgd = self.model.Cgd
        # can schedule other matrices as well

        # increase noise, etc.

        local_model = self.model.copy()

        radial_map_number = np.sqrt(map_number1**2 + map_number2**2)

        Cdd = base_Cdd * (1 + radial_map_number * self.config["capacitance_params"]["cdd_increase_per_map"])
        Cgd = base_Cgd * (1 + radial_map_number * self.config["capacitance_params"]["cgd_increase_per_map"])

        local_model.Cdd = Cdd
        local_model.Cgd = Cgd

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
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

