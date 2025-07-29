import numpy as np
import matplotlib.pyplot as plt

from qdarts.plotting import get_polytopes, plot_polytopes
from qdarts.capacitance_model import CapacitanceModel
from qdarts.simulator import CapacitiveDeviceSimulator

import time
start_time = time.time()

N_dots = 3
N_gates = 6 # 2 dot plungers + 1 barrier gate + 1 sensor gate
inner_dots = [0,1]
sensor_dots = [2]

dot_plungers = [0,1] # plunger gates
barrier_plungers = [2,3,4] # barrier gate
sensor_plungers = [5] # sensor gate

target_state = [1, 1, 5]

C_DD = np.array([[1, 0.2, 0.08],
                  [0.2, 1, 0.08],
                  [0.08, 0.08, 1]], dtype=np.float32)
# # inner dot 1, inner dot 2, sensor dot

C_DG = np.array([ [0.75,  0.1, 0, 0, 0, 0.02],
                  [ 0.1, 0.75, 0, 0, 0, 0.02],
                  [   0,    0, 0, 0, 0,    1]], dtype=np.float32)
# # plunger gate 1, plunger gate 2, barrier gate, sensor gate



# Minimum voltages for each plunger gate
bounds_limits = -2.0 * np.ones(N_gates)

# Deviation from the constant interaction model
ks = 4 * np.ones(N_dots)

print("Creating capacitance model...")
capacitance_model = CapacitanceModel(C_DG, C_DD, bounds_limits, ks=ks)

print("Creating device simulator...")
capacitive_sim = CapacitiveDeviceSimulator(capacitance_model)

print(f"Computing boundaries for target state: {target_state}")
m = capacitive_sim.boundaries(target_state).point_inside

P = np.zeros((N_gates, 2))
P[0, 0] = 1  # First dot plunger along x-axis
P[1, 1] = 1  # Barrier gate along y-axis

from qdarts.plotting import get_CSD_data

# minV = np.array([-0.03,-0.15])
# maxV = np.array([ 0.03, 0.15])
minV = np.array([-0.2,-0.2])
maxV = np.array([ 0.4, 0.4])

resolution = 250

print(f"Computing CSD data with {resolution}x{resolution} = {resolution**2} points...")
sliced_csim, CSD_data, states =  get_CSD_data(capacitive_sim, m, P, minV, maxV, resolution, target_state)

xs = np.linspace(minV[0],maxV[0],resolution)
ys = np.linspace(minV[1],maxV[1],resolution)

plt.pcolormesh(xs,ys,CSD_data.T)
V_offset_2D = P.T @ m
plt.xlim(minV[0],maxV[0])
plt.ylim(minV[1],maxV[1])
plt.savefig('qdarts_plot.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")


tunnel_couplings = np.zeros((N_dots, N_dots))
tunnel_couplings[0, 1] = 30e-6
tunnel_couplings[1, 0] = 30e-6
tunnel_couplings[0, 2] = 30e-6
tunnel_couplings[1, 2] = 30e-6
tunnel_couplings[2, 1] = 30e-6
tunnel_couplings[2, 0] = 30e-6

#tunnel_couplings = tunnel_couplings+ tunnel_couplings.T

temperature = 0.2 # 200mK


from qdarts.tunneling_simulator import NoisySensorDot, ApproximateTunnelingSimulator

print(f"Creating sensor model, time elapsed = {time.time() - start_time:.2f}s")

# Update sensor model initialization
sensor_model = NoisySensorDot(sensor_dots) # Define sensor dots for the 2-dot setup
sensor_model.config_peak(g_max = 1.0, peak_width_multiplier = 200) #make the sensor peak broader
tunneling_sim = ApproximateTunnelingSimulator(capacitive_sim, #the underlying polytope simulation
                                             tunnel_couplings,  #constant tunnel couplings
                                             temperature, #electron temperature, should be <=200mK
                                             sensor_model) #our sensor model simulation

capacitive_sim.set_maximum_polytope_slack(5/tunneling_sim.beta) #adding slack to keep more states that are likely to affect the hamiltonian
tunneling_sim.num_additional_neighbours[sensor_dots] = 2 #adding additional states for the sensor dots

print(f"Creating tunneling simulator, time elapsed = {time.time() - start_time:.2f}s")
state=tunneling_sim.poly_sim.find_state_of_voltage(m, [0,0,2])  
sensor_values = tunneling_sim.sensor_scan_2D(m, P, minV, maxV, resolution, state)

print(f"Sensor scan completed, time elapsed = {time.time() - start_time:.2f}s")

plt.pcolormesh(xs, ys, sensor_values[:,:,0].T)
plt.xlim(minV[0],maxV[0])
plt.ylim(minV[1],maxV[1])
print(f"Generating plot, time elapsed = {time.time() - start_time:.2f}s")

plt.savefig('qdarts_plot_noised.png', dpi=300, bbox_inches='tight')

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")