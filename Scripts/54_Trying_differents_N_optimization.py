import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import cupy as cp

current_dir =  os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

gpu = True
if gpu:
    print("GPU mode")
    from frank2d_gpu.frank2d_gpu import Frank2D
    from frank2d_gpu.geometry import Geometry
    from frank2d_gpu.fourier2d_gpu import FourierTransform2D
    from frank2d_gpu.preprocess_vis_gpu import Gridding
    from frank2d_gpu.plot_gpu import Plot
    from frank2d_gpu.constants import rad_to_arcsec, deg_to_rad
    from frank2d_gpu.posterior_optimization_gpu import MAPEstimator

else:
    from frank2d import Frank2D
    from frank2d.geometry import Geometry
    from frank2d.plot import Plot
    from frank2d.posterior_optimization import *
    from frank2d.constants import rad_to_arcsec, deg_to_rad
    from frank2d.posterior_optimization import MAPEstimator

os.environ["OMP_NUM_THREADS"] = "1"

# Huang 2018 
AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2*2, 'disk_name': 'AS209'}
Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88*2, 'disk_name': 'Elias27'}

best_Elias27 = {'m': -3.297430763696686, 'c': 316115113583956.4, 'l': 44039.6957302716}
best_AS209 = {'m': -1.3620915465005585, 'c': 4681.0681578390195, 'l': 72454.15132114716}

data =  Elias27
print("Disk name: ", data['disk_name'])

dir = "./../../data/"
disk_name = data['disk_name']
data_file = dir +"uvtable_" + disk_name + "_continuum.npz"

inc = data['inc']
pa = data['pa']
dra = data['dra']
ddec = data['ddec']
rout = data['rout']

# Load data
if gpu:
    file = cp.load(data_file)
else:
    file = np.load(data_file)
u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
Vis = Re + Im*1j

# Frank Parameters
N = 300
uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

geom = Geometry(inc, pa, dra, ddec)
frank2d = Frank2D(N, rout, geom)

print("Creating gridded vis..")
frank2d.process_vis(uvtable)

print("Searching for optimal values of m, c and l..")
N_opt = 100
print("Trying N_opt = ", str(N_opt))
ME = MAPEstimator(frank2d._Rmax,  frank2d._Geometry, N = N_opt)
frank2d.set_MAP_estimator(ME)

start_time = time.time()
frank2d.search_MAP()

end_time = time.time()
print("Time taken for optimization in minutes: ", (end_time - start_time)/60.0)

print("Best value for N_opt ", str(N_opt), " ->",  frank2d.MAP)