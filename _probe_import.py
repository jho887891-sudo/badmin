import sys
sys.path.insert(0, 'src')
import numpy as np
from trajectory.shuttle_aerodynamics import rk4_step, acceleration
print('NS-IMPORT-OK', rk4_step(np.zeros(3), np.array([1.,0,0]), 0.01, k_per_m=1/6.5)[1])
