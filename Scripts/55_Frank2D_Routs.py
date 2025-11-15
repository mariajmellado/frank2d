import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

current_dir =  os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# frank2d
from frank2d import Frank2D
from frank2d.geometry import Geometry
from frank2d.plot import Plot
from frank2d.posterior_optimization import *
from frank2d.constants import rad_to_arcsec, deg_to_rad
from frank2d.posterior_optimization import MAPEstimator

AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2*2, 'disk_name': 'AS209'}
Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88*2, 'disk_name': 'Elias27'}

best_Elias27 = {'m': -3.297430763696686, 'c': 316115113583956.4, 'l': 44039.6957302716}
best_AS209 = {'m': -1.3620915465005585, 'c': 4681.0681578390195, 'l': 72454.15132114716}

data =  AS209
print("Disk: ", data['disk_name'])


dir = "./../../data/"
disk_name = data['disk_name']
data_file = dir +"uvtable_" + disk_name + "_continuum.npz"

inc = data['inc']
pa = data['pa']
dra = data['dra']
ddec = data['ddec']
rout = data['rout']

# Load data
file = np.load(data_file)
u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
Vis = Re + Im*1j
uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

# Frank Parameters
N = 300
params = best_AS209

r_outs = [4.5]
print("Rout tests: ", r_outs)
r_outs_name = ["4p5"]
frank2d_objects = []

for i in range(len(r_outs)):
    rout = r_outs[i]
    print("Fitting Frank2D with Rout = ", rout, " arcsec")

    geom = Geometry(inc, pa, dra, ddec)
    frank2d = Frank2D(N, rout, geom)
    
    frank2d.fit(data = uvtable, kernel_params = params,
                rtol = 1e-8, maxiter =100000)
    frank2d_objects.append(frank2d)

# Frank 1D parameters
n_pts = 300
alpha = 1.05
w_smooth = 1e-4
r_out = rout

sol_f1d = frank2d.frank1d(alpha = alpha, w_smooth = w_smooth, n_pts = n_pts, rout = r_out)

#Fits file
dir_fits =  dir
disk_fits = disk_name + '_continuum.fits'
fits_file = dir_fits + disk_fits


for i in range(len(frank2d_objects)):
    frank2d = frank2d_objects[i]
    print("rout = ", r_outs[i])
    print("     --> cellsize = ", (2*frank2d.Rmax)/frank2d._N)

    u, v = frank2d.u_grid, frank2d.v_grid
    vis = frank2d.visibility_model

    index = np.where((u == 0) & (v == 0))
    V_00 = vis[index].real

    x, y = frank2d.x_grid, frank2d.y_grid
    I = frank2d.intensity_model

    dx, dy = frank2d.FT._dx, frank2d.FT._dy

    domega = dx * dy
    sum_I = I.sum() * domega                   
    print(V_00, " vs ",  sum_I)

    Plot_ = Plot(frank2d)
    Plot_.intensity()

    Plot_.set_f1d_solution(sol_f1d)
    Plot_.set_fits_file(fits_file)

    Plot_.intensity_profile(
    clean = True, frank1d = True,
    resol_f2d = 0.023, resol_f1d = 0.0147,
    x_lims = (0, 1.5), y_lims = (1e-7, 3e-3),
    save_fig = True,
    )

    dict_frank2d = {
        'visibility_model': frank2d.visibility_model,
        'intensity_model': frank2d.intensity_model,
        'u_grid': frank2d.u_grid,
        'v_grid': frank2d.v_grid,
        'x_grid': frank2d.x_grid*rad_to_arcsec,
        'y_grid': frank2d.y_grid*rad_to_arcsec,
    }
    with open("f2d_rout" + str(r_outs_name[i])+".pkl", "wb") as file:
        pickle.dump(dict_frank2d, file)