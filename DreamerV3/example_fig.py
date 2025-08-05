
# add Code folder, if the package is not installed.
import sys
sys.path.append('../src')
import pathlib

# import main class and plotting function
from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

# import standard libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define the system

#All capacitances are given in aF
N = 6 #number of dots   
C_DD=20* np.eye((N))/2 #The self-capacitance of each dot, NOTE: factor of 2 due to symmetrization
C_DD[0,1] = 10 #capacitance between dot 0 and dot 1 (Left double dot) 
C_DD[2,3] = 7 #capacitance between dot 3 and dot 4 (Right double dot)

C_DD[0,4] = 1.6 #capacitance between sensor dot 4 and dot 0
C_DD[1,4] = 1.4 #capacitance between sensor dot 4 and dot 1
C_DD[2,5] = 1.4 #capacitance between sensor dot 5 and dot 2
C_DD[3,5] = 2 #capacitance between sensor dot 5 and dot 3
C_DD[1,2] = 6 #capacitance between the middle dots 2 and dot 3
C_DD = C_DD + C_DD.T

C_DG=11*np.eye(N) #dot-to-gate capacitances 
#cross-capacitances
C_DG[0,1] = 1.5 #dot 0 from dot 1
C_DG[1,0] = 1.2 #dot 1 from dot 0
C_DG[2,3] = 1.3 #dot 2 from dot 3
C_DG[3,2] = 1.4 #dot 3 from dot 3

# Definition of the tunnel couplings in eV 
# NOTE: we use the convention that tc is the energy gap at avoided crossing H = tc/2 sx
tunnel_couplings = np.zeros((N,N))
tunnel_couplings[0,1] = 50*1e-6
tunnel_couplings[1,0] = 50*1e-6
tunnel_couplings[2,3] = 60*1e-6
tunnel_couplings[3,2] = 60*1e-6
capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : 4,       #distortion of Coulomb peaks. NOTE: If None -> constant size of Coublomb peak 
}

tunneling_config = {
        "tunnel_couplings": tunnel_couplings, #tunnel coupling matrix
        "temperature": 0.1,                   #temperature in Kelvin
        "energy_range_factor": 5,  #energy scale for the Hamiltonian generation. NOTE: Smaller -> faster but less accurate computation 
}
sensor_config = {
        "sensor_dot_indices": [4,5],  #Indices of the sensor dots
        "sensor_detunings": [0.0005,0.0005],  #Detuning of the sensor dots
        "noise_amplitude": {"fast_noise": 0.5*1e-6, "slow_noise": 1e-8}, #Noise amplitude for the sensor dots in eV
        "peak_width_multiplier": 15,  #Width of the sensor peaks in the units of thermal broadening m *kB*T/0.61.
}
# Create the experiment object from the configuration files
experiment = Experiment(capacitance_config, tunneling_config, sensor_config)