import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from frank2d import Frank2D
from frank2d.geometry import Geometry
from frank2d.plot import Plot
from frank2d.posterior_optimization import *
from frank2d.constants import rad_to_arcsec, deg_to_rad
from frank2d.posterior_optimization import MAPEstimator

current_dir =  os.getcwd()
# replace this following line in the path your frank2d files are.
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

os.environ["OMP_NUM_THREADS"] = "1"

# Huang 2018 
AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2*2, 'disk_name': 'AS209'}
Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88*2, 'disk_name': 'Elias27'}

best_Elias27 = {'m': -3.297430763696686, 'c': 316115113583956.4, 'l': 44039.6957302716}
best_AS209 = {'m': -1.3620915465005585, 'c': 4681.0681578390195, 'l': 72454.15132114716}

data =  AS209


dir = "./../../data/"
disk_name = data['disk_name']
data_file = dir +"uvtable_" + disk_name + "_continuum.npz"

inc = data['inc']
pa = data['pa']
dra = data['dra']
ddec = data['ddec']
rout = data['rout']

# Frank Parameters
N = 300

# Load data
file = np.load(data_file)
u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
Vis = Re + Im*1j

uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

dir_fits =  "/Users/mariajmelladot/Desktop/Frank2D/data/fits/"
disk_fits = disk_name + '_continuum.fits'
fits_file = dir_fits + disk_fits

geom = Geometry(inc, pa, dra, ddec)
frank2d = Frank2D(N, rout, geom)

frank2d.process_vis(uvtable)
frank2d.fit(kernel_params = best_params, rtol = 1e-8)

ME = MAPEstimator(frank2d._Rmax,  frank2d._Geometry, N = 100)
frank2d.set_MAP_estimator(ME)
frank2d.search_MAP()
frank2d.MAP