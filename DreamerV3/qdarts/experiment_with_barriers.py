from .simulator import *
from .tunneling_simulator import *
from .util_functions import compensated_simulator
from .noise_processes import OU_process

#for the algorithm
import numpy as np

#for plotting
from matplotlib import pyplot as plt
from .plotting import get_CSD_data, get_polytopes

#for measuring runtime
import time
plt.rcParams.update({'font.size': 18})

# SIMULATION CONSTANTS
SLOW_NOISE = {"tc": 250, 
              "samples": 5, 
              "virtual_samples": 200}



class Experiment(): #TODO: change name to the simulator name
    def __init__(self, capacitance_config, tunneling_config = None, sensor_config = None, barrier_config = None, print_logs = True):
        '''
        capacitance_config: dictionary containing the capacitance model parameters
        tunneling_config: dictionary containing the tunneling model parameters
        sensor_config: dictionary containing the sensor model parameters
        barrier_config: dictionary containing the barrier control parameters
        print_logs: bool, whether to print logs
        '''
        # Store configurations
        self.capacitance_config = capacitance_config
        self.tunneling_config = tunneling_config
        self.sensor_config = sensor_config
        self.barrier_config = barrier_config
        self.print_logs = print_logs

        print("EXPERIMENT WITH BARRIERS INITIALIZED")
        print("-------------------------------------")

        # Deploy simulators
        self.capacitance_sim = self.deploy_capacitance_sim(capacitance_config)
        self.has_sensors = False
        
        # Store base tunnel couplings for dynamic updates
        if tunneling_config is not None:
            self.base_tunnel_couplings = tunneling_config["tunnel_couplings"].copy()
        
        # Check requirements for sensor and tunneling configurations
        if tunneling_config != None and sensor_config == None:
            raise ValueError("Specifying a tunneling configuration also requires a sensor configuration.")
        elif (tunneling_config == None and sensor_config != None):
            raise ValueError("Specifying a sensor configuration also requires a tunneling configuration.")        
        elif (tunneling_config != None and sensor_config != None):
            self.sensor_model = self.deploy_sensor_model(sensor_config)
            self.tunneling_sim = self.deploy_tunneling_sim(self.capacitance_sim, tunneling_config)
            self.has_sensors = True


# DEPLOYMENT FUNCTIONS
#--------------------------
    def deploy_capacitance_sim(self,config):
        '''
        Function that deploys a capacitance simulator with full capacitance matrices.
        ----------------
        Arguments:
        config: dictionary containing the capacitance model parameters
        ----------------
        Returns:
        sim: CapacitiveDeviceSimulator object
        '''
        # Handle both C_Dg (Experiment format) and C_DG (full matrix format)
        if "C_Dg" in config:
            # Original Experiment format - use as is
            capacitance_model = ModelWithBarriers(config["C_Dg"], config["C_DD"], -1, ks= config["ks"])
            self.N = len(config["C_Dg"])  # number of dots
        elif "C_DG" in config:
            # Full capacitance matrix format - use full matrix
            capacitance_model = ModelWithBarriers(config["C_DG"], config["C_DD"], -1, ks= config["ks"])
            self.N = len(config["C_DG"][0])  # number of gates (not dots)
        else:
            raise ValueError("Capacitance config must contain either 'C_Dg' or 'C_DG'")
        
        sim = CapacitiveDeviceSimulator(capacitance_model)     
        
        # CORRECT: Store proper dimensions
        self.num_dots = sim.num_dots  # Store number of dots separately
        self.num_gates = sim.num_inputs  # Store number of gates separately
        
        #Save the parameters
        self.inner_dots = list(np.arange(self.num_dots))  #indces of the dots. NOTE: no sensor at this point   
        
        if self.print_logs:
            if config["ks"] == None:
                # Print log of capacitance parameters
                log = """
                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                {}
                Dot-gate capacitances: 
                {}
                Size of Coulomb peaks V[n] scale as {}/(n + {})
                """.format(config["C_DD"], config.get("C_DG", config.get("C_Dg")), np.round((1-0.137*3.6/(config["ks"]+2.6)*config["ks"]),3), 2+config["ks"])
                print(log)
            else:
                 # Print log of capacitance parameters
                log = """
                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                {}
                Dot-gate capacitances: 
                {}
                Size of Coulomb peaks V[n] is constant
                """.format(config["C_DD"], config.get("C_DG", config.get("C_Dg")))
                print(log)

        
        return sim
    

    def deploy_tunneling_sim(self, capacitance_sim, tunneling_config):
        '''
        Function that deploys a tunneling simulator.
        ----------------
        Arguments:
        capacitance_sim: CapacitiveDeviceSimulator object
        tunneling_config: dictionary containing the tunneling model parameters
        ----------------
        Returns:
        tunneling_sim: ApproximateTunnelingSimulator object
        '''
        tunneling_matrix = tunneling_config["tunnel_couplings"]
        if np.max(tunneling_matrix)==0:
            tunneling_matrix = tunneling_matrix + 1e-20
        tunneling_sim = ApproximateTunnelingSimulator(capacitance_sim, 
                                             tunneling_matrix,   
                                             tunneling_config["temperature"],
                                             self.sensor_model)
        
        # explore the neighboring polytopes
        slack = tunneling_config["energy_range_factor"]*np.maximum(1.0/tunneling_sim.beta,np.max( tunneling_matrix))
        tunneling_sim.poly_sim.set_maximum_polytope_slack(slack)
        tunneling_sim.num_additional_neighbours[self.sensor_model.sensor_dot_ids] = 2 
    

        if self.print_logs:
            # Print log of tunneling parameters
            log = """
            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            {}
            Temperature: {} K
            Energy range factor: {}
            """.format(tunneling_matrix, tunneling_config["temperature"], tunneling_config["energy_range_factor"])
            print(log)


        return tunneling_sim


    def deploy_sensor_model(self, sensor_config):
        '''
        Function that deploys a sensor model.
        ----------------
        Arguments:
        config: dictionary containing the sensor model parameters
        ----------------
        Returns:
        sensor_sim: NoisySensorDot object
        '''
        # Separate between inner and sensor dots
        self.inner_dots = list(set(self.inner_dots) - set(sensor_config["sensor_dot_indices"]))
        
        # Define slow-noise generator
        slow_noise_gen = OU_process(sig = sensor_config["noise_amplitude"]["slow_noise"], 
                                tc = SLOW_NOISE["tc"], 
                                dt = 1,  # the unit of time is the single measurment
                                num_points=SLOW_NOISE["samples"], )  #TODO: LATER: implement 1/f noise
       
        # Deploy sensor model
        sensor_sim = NoisySensorDot(sensor_config["sensor_dot_indices"])

        # Configure sensor model
        sensor_sim.config_noise(sigma = sensor_config["noise_amplitude"]["fast_noise"], 
                            n_virtual_samples = SLOW_NOISE["virtual_samples"], 
                            slow_noise_gen = slow_noise_gen)
        
        sensor_sim.config_peak(g_max = 1.0, 
                                peak_width_multiplier = sensor_config["peak_width_multiplier"]) 


        # Pring log of sensor parameters
        if self.print_logs:
            log = """
            Sensor model deployed with the following parameters:   
            Sensor dot indices: {}
            Sensor detunings: {} meV
            Coulomb peak width: {} meV
            Slow noise amplitude: {} ueV
            Fast noise amplitude: {} ueV
            """.format(sensor_config["sensor_dot_indices"], np.array(sensor_config["sensor_detunings"])*1e3, 
                    np.round(2*sensor_config["peak_width_multiplier"]*self.tunneling_config["temperature"]*86*1e-3/0.631,2),
                    np.round(sensor_config["noise_amplitude"]["slow_noise"]*1e6,4),np.round(sensor_config["noise_amplitude"]["fast_noise"]*1e6,4))
            print(log)
        return sensor_sim


# BARRIER CONTROL FUNCTIONS
#--------------------------
    def update_tunnel_couplings(self, barrier_voltages):
        """
        Update tunnel couplings based on barrier voltages using exponential physics.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages (e.g., {'barrier_2': 0.5})
        """
        if self.barrier_config is None:
            raise ValueError("No barrier configuration available")
        
        if not self.has_sensors:
            raise ValueError("Tunneling simulator not initialized")
        
        # Calculate new tunnel couplings
        new_couplings = self._calculate_barrier_tunnel_couplings(barrier_voltages)
        
        # Update tunnel couplings directly (no reinitialization needed!)
        self.tunneling_sim.tunnel_matrix = new_couplings
        
        if self.print_logs:
            print(f"Barrier voltages updated: {barrier_voltages}")
            print(f"New tunnel couplings:\n{new_couplings}")


    def _calculate_barrier_tunnel_couplings(self, barrier_voltages):
        """
        Calculate tunnel couplings based on barrier voltages using exponential physics.
        
        Args:
            barrier_voltages: Dictionary of barrier voltages
            
        Returns:
            Updated tunnel coupling matrix
        """
        # Start with base tunnel couplings
        new_couplings = self.base_tunnel_couplings.copy()
        barrier_mappings = self.barrier_config['barrier_mappings']
        
        # Apply barrier voltage effects
        for barrier_name, voltage in barrier_voltages.items():
            if barrier_name not in barrier_mappings:
                continue
                
            mapping = barrier_mappings[barrier_name]
            coupling_type = mapping['coupling_type']
            alpha = mapping['alpha']
            base_coupling = mapping['base_coupling']
            voltage_offset = mapping['voltage_offset']
            
            # Calculate effective barrier voltage
            effective_voltage = voltage - voltage_offset
            
            # Calculate coupling strength using exponential physics
            coupling_strength = base_coupling * np.exp(-alpha * effective_voltage)
            
            if coupling_type == "dot_to_dot":
                # Barrier controls coupling between two dots
                target_dots = mapping['target_dots']
                dot1, dot2 = target_dots[0], target_dots[1]
                
                # Update both directions (symmetric matrix)
                new_couplings[dot1, dot2] = coupling_strength
                new_couplings[dot2, dot1] = coupling_strength
                
            elif coupling_type in ["reservoir_to_dot", "dot_to_reservoir"]:
                # Barrier controls coupling between reservoir and dot
                target_dot = mapping['target_dot']
                
                # For reservoir coupling, we modify the diagonal element
                # This represents the coupling strength to the reservoir
                new_couplings[target_dot, target_dot] = coupling_strength
        
        return new_couplings


    def get_current_tunnel_couplings(self):
        """
        Get current tunnel coupling matrix.
        
        Returns:
            Current tunnel coupling matrix
        """
        if hasattr(self, 'tunneling_sim'):
            return self.tunneling_sim.tunnel_matrix
        else:
            return self.base_tunnel_couplings


    def get_barrier_voltage_effect(self, barrier_name, voltage_range, resolution=50):
        """
        Calculate the effect of a barrier voltage on tunnel couplings.
        
        Args:
            barrier_name: Name of the barrier (e.g., 'barrier_2')
            voltage_range: [min_voltage, max_voltage]
            resolution: Number of voltage points
            
        Returns:
            Dictionary with voltage points and corresponding coupling strengths
        """
        if self.barrier_config is None:
            raise ValueError("No barrier configuration available")
        
        if barrier_name not in self.barrier_config['barrier_mappings']:
            raise ValueError(f"Unknown barrier: {barrier_name}")
        
        mapping = self.barrier_config['barrier_mappings'][barrier_name]
        coupling_type = mapping['coupling_type']
        base_coupling = mapping['base_coupling']
        alpha = mapping['alpha']
        voltage_offset = mapping['voltage_offset']
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], resolution)
        couplings = np.zeros(resolution)
        
        for i, voltage in enumerate(voltages):
            effective_voltage = voltage - voltage_offset
            coupling_strength = base_coupling * np.exp(-alpha * effective_voltage)
            couplings[i] = coupling_strength
        
        return {
            'voltages': voltages,
            'couplings': couplings,
            'coupling_type': coupling_type,
            'base_coupling': base_coupling,
            'alpha': alpha
        }
        
    
    def center_transition(self, csimulator, target_state, target_transition, plane_axes, use_virtual_gates = False, compensate_sensors = False):
        '''
        Function that center the CSD at a given facet (transition) of the polytope (occupation state).
        ----------------
        Arguments:
        csimulator: CapacitanceSimulator object
        target_state: int, the state at which the transition happens, e.g. [2,2] 
        target_transition: list of integers, the transition point e.g. [1,-1] would be the transition from [2,2] to [1,1]
        plane_axes: 2xN array, the axes of the transition which span the plane
        use_virtual_gates: bool, whether to use virtual gates
        compensate_sensors: bool, whether to compensate the sensors
        ----------------
        Returns:
        plane_axes: 2xN array, the axes spanning the cut through volage plane
        transition_sim: CapacitanceSimulator object, the transition simulator
        '''
        transition = np.array(target_transition).T
        if compensate_sensors:
            # fix the sensor gate if we compensate
            csimulator = fix_gates(csimulator,self.sensor_config["sensor_dot_indices"], np.zeros(len(self.sensor_config["sensor_dot_indices"])))
            # reduce dimension of the voltag space
            plane_axes = plane_axes[:,self.inner_dots]
            #get_labels of the transitions
            poly = csimulator.boundaries(target_state)
            inner_labels = poly.labels[:,self.inner_dots]
            #Find the index of the transition point
            transition = np.array(target_transition)[self.inner_dots].T
        else:
            poly = csimulator.boundaries(target_state)
            transition = np.array(target_transition).T
            inner_labels = poly.labels

        #find boundaries of the selected polytope
 
        try:
            idx_multidot_transition = [find_label(inner_labels, transition)[0]] 
        except ValueError:
            raise "The transition point is not in the polytope. Please choose a different transition"
        
        #Find the offset of V
        v_transition = find_point_on_transitions(poly,idx_multidot_transition)  
        print("v_offset found:",v_transition) 
        
        
        if use_virtual_gates:
            # Compute the normals
            pair_transitions=np.array(
                [np.array(transition).T for transition in plane_axes],dtype=int)  

            idxs = [find_label(inner_labels,t)[0] for t in pair_transitions]  
            normals = -poly.A[idxs]
            normals /= np.linalg.norm(normals,axis=1)[:,None]
            P_transition = normals.T
            transition_sim = axis_align_transitions(
                            csimulator.slice(P_transition, v_transition),
                            target_state,poly.labels[idxs],[0,1])
        else:
            transition_sim = csimulator.slice(plane_axes.T, v_transition)

        return np.eye(2), transition_sim     

# GETTERS
#--------------------------
            
    def get_virtualised_csim(self, csimulator, target_state):
        '''
        Function that takes a capacitance simulator and virtualises the gates specified by inner_dots.
        ----------------
        Arguments:
        csimulator: CapacitanceSimulator object
        target_state: int, the initial corner state guess
        ----------------
        Returns:
        csimulator: CapacitanceSimulator object, the virtualised simulator
        '''
        gate_transitions = np.eye(self.N,dtype=int)[self.inner_dots]
        #TODO: Default target state is lower left corner state (initial guess). In future user could specify!
        self.target_state = target_state
        csimulator = axis_align_transitions(csimulator,  self.target_state, gate_transitions, self.inner_dots)
        return csimulator
    

    def get_compensated_csim(self,csimulator, target_state):
        '''
        Function that takes a capacitance simulator and compensates the sensors.
        ----------------
        Arguments:
        csimulator: CapacitanceSimulator object
        target_stater: int, the state at which sensor compensation happens
        ----------------
        Returns:
        csimulator: CapacitanceSimulator object, the compensated simulator
        '''
        if not self.has_sensors:
                raise ValueError("Compensating sensors requires a sensor model.")
            
            # TODO: Do we need to specify compensation gates *and* sensor ids? Only if we want to compensate a subset of sensors.
        else:
            csimulator = compensated_simulator(csimulator,
                                        target_state=target_state,
                                        compensation_gates=self.sensor_config["sensor_dot_indices"],
                                        sensor_ids = self.sensor_config["sensor_dot_indices"], 
                                        sensor_detunings = self.sensor_config["sensor_detunings"])
        return csimulator
    
    def get_plot_args(self, x_voltages, y_voltages, plane_axes, v_offset = None):
        '''
        Function that returns the arguments for plotting the CSD.
        ----------------
        Arguments:
        x_voltages: list of floats, the x-axis voltages
        y_voltages: list of floats, the y-axis voltages
        plane_axes: 2xN array, the axes of the plane in which the CSD is to be rendered
        v_offset: Nx1 array, the offset voltage of all of the gates, which defines the origin of the plot
        ----------------
        Returns:
        v_offset: Nx1 array, the offset voltage of all of the gates
        minV: 2x1 array, the minimum voltage of selected axes
        maxV: 2x1 array, the maximum voltage of selected axes
        resolution: list of integers, the resolution of the plot
        '''

        if v_offset is None:
            v_offset = np.zeros(self.N)
        else:
            v_offset = np.array(v_offset,dtype=float)

        xout = x_voltages + v_offset[plane_axes[0]]
        yout = y_voltages + v_offset[plane_axes[1]]


        minV = np.array([x_voltages[0], y_voltages[0]])
        maxV = np.array([x_voltages[-1], y_voltages[-1]])
        
        resolution = [len(x_voltages), len(y_voltages)]

        return v_offset, minV, maxV, resolution, xout, yout


# RENDER FUNCTIONS
#--------------------------

    def generate_CSD(self, x_voltages, y_voltages, plane_axes, target_state = None, 
                               target_transition = None, use_virtual_gates = False, 
                               compensate_sensors = False, compute_polytopes = False,
                               use_sensor_signal = False, v_offset = None):
        '''
        Function that renders the capacitance CSD for a given set of voltages and axes.
        ----------------
        Arguments:
        x_voltages: list of floats, the x-axis voltages
        y_voltages: list of floats, the y-axis voltages
        plane_axes: 2xN array, the axes of the plane in which the CSD is to be rendered
        target_state: int, the guess state or the state at which the transition happens
        target_transition: list of integers, the transition point e.g. [1,-1] would be the transition from [2,2] to [1,1]
        use_virtual_gates: bool, whether to use virtual gates
        compensate_sensors: bool, whether to compensate the sensors
        compute_polytopes: bool, whether to compute the polytopes
        use_sensor_signal: bool, whether to use the sensor signal
        v_offset: Nx1 array, the offset voltage of all of the gates, which defines the origin of the plot
        ----------------
        Returns:
        xout, yout: list of floats, the x and y voltages
        CSD_data: 2D array, the CSD data
        polytopes: dictionary, the polytopes of the CSD. None if compute_polytopes is False
        sensor_values: 3D array, the sensor signal [size(xout),size(yout),num_sensors]. None if use_sensor_signal is False
        v_offset: Nx1 array, the offset voltage of all of the gates
        '''
        sensor_values = None
        CSD_data = None
        polytopes = None

        # check required parameters
        if target_state is None:
            #if not target state use [0,0,0] for number of dots
            target_state = [0]*self.num_dots
        
        plane_axes = np.array(plane_axes)
        # prepare plot
        v_offset, minV, maxV, resolution, xout, yout = self.get_plot_args(x_voltages, y_voltages, plane_axes, v_offset) 
        
        # prepare the simulator
        csimulator = self.capacitance_sim
        
        if compensate_sensors:
            csimulator = self.get_compensated_csim(csimulator,target_state= target_state)
     
        if target_transition is not None:
            plane_axes, csimulator = self.center_transition(csimulator, target_state, target_transition, 
                                                            plane_axes, use_virtual_gates, compensate_sensors)
            v_offset = np.zeros(2)  #TODO: how to do it nicer?
        

        elif use_virtual_gates:
            csimulator = self.get_virtualised_csim(csimulator, target_state)
        
        # Construct proper sweep matrix P from plane_axes
        # Use self.num_gates (number of gates) instead of self.N
        P = np.zeros((self.num_gates, 2))
        P[plane_axes[0], 0] = 1  # x-axis direction
        P[plane_axes[1], 1] = 1  # y-axis direction
        
        if use_sensor_signal:
            sensor_values = self.tunneling_sim.sensor_scan_2D(v_offset, P, minV, maxV, resolution, target_state)
            # When using sensor signal, return sensor values as the main data
            return xout, yout, sensor_values, polytopes, sensor_values, v_offset
        else:
            # Part for the electrostatic CSD:
            if not use_sensor_signal or compute_polytopes:
                backend, CSD_data, states =  get_CSD_data(csimulator, v_offset, P, minV, maxV, resolution, target_state)
                if compute_polytopes:
                    v_offset_polytopes = [np.dot(v_offset,plane_axes[0]), np.dot(v_offset,plane_axes[1])]
                    polytopes = get_polytopes(states, backend, minV, maxV,  v_offset_polytopes)
                
                if not use_sensor_signal:
                    return xout,yout, CSD_data.T, polytopes, sensor_values, v_offset
            
            # Part for the sensor signal:
            self.print_logs = False
            simulator = self.deploy_tunneling_sim(csimulator, self.tunneling_config)
            sensor_values = simulator.sensor_scan_2D(v_offset, P, minV, maxV, resolution, target_state)

            if compute_polytopes:
                backend, CSD_data, states =  get_CSD_data(csimulator, v_offset, P, minV, maxV, resolution,
                                                           target_state)
                V_offset_polytopes = [np.dot(v_offset,plane_axes[0]), np.dot(v_offset,plane_axes[1])]
                polytopes = get_polytopes(states, backend, minV, maxV,   V_offset_polytopes)
        return xout, yout, CSD_data.T, polytopes, sensor_values, v_offset
        
        
