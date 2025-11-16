

import os
import sys

current_dir =  os.getcwd()
# replace this following line in the path your frank2d files are.
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import time

from frank2d import Frank2D
from frank2d.geometry import Geometry
from frank2d.plot import Plot
from frank2d.posterior_optimization import *
from frank2d.constants import rad_to_arcsec, deg_to_rad
from frank2d.posterior_optimization import MAPEstimator



dir = "./../../data/"
data_file = dir + "uvtable_nonsym_disc_blob_noisy.npz"
inc = 0
pa = 0
dra = 0
ddec = 0
rout = 1*2

# Load data
file = np.load(data_file)
u, v, Vis, Weights = file['u'], file['v'], file['vis'], file['weights']
uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

geom = Geometry(inc, pa, dra, ddec)
N = 300
rout = 1*2

N_opt = [50, 60, 70, 80, 100]
maps = []
frank2d_objects = []
times = []
Qmax = []

for i in N_opt:
    start_time = time.time()
    print("Processing for N = {}".format(i))
    frank2d = Frank2D(N, rout)
    frank2d.process_vis(uvtable)
    initial_guess = {'m': -2, 'logl': 4}
    frank2d.search_MAP(initial_guess= initial_guess, N = i)

    end_time = time.time()
    total_time = (end_time - start_time)/60
    print("     ---> N = {}, Time taken: {:.2f} seconds".format(i, total_time))
    frank2d_objects.append(frank2d)
    times.append(total_time)

    best = frank2d.MAP
    maps.append(best)
    print("     ---> MAP: "+ str(best))

    MAPEs = frank2d.MAPEstimator
    GP =  MAPEs.GaussianModel
    DFT = GP.DFT2
    qmax = np.max(DFT.Qmax)
    Qmax.append(qmax)

    print("     ---> Qmax: " + str(qmax))
    print("x-------------------------------x")

print(N_opt)
print("Times taken: ", times)
print("MAPs: ", maps)
print("Qmax: ", Qmax)