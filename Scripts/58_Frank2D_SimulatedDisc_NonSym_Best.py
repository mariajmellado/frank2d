import os
import sys

current_dir =  os.getcwd()
# replace this following line in the path your frank2d files are.
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import time

#frank2d
from frank2d import Frank2D
from frank2d.geometry import Geometry
from frank2d.plot import Plot
from frank2d.posterior_optimization import *
from frank2d.constants import rad_to_arcsec, deg_to_rad
from frank2d.posterior_optimization import MAPEstimator

# Huang 2018 
AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2*2, 'disk_name': 'AS209'}
IMLup = {'inc': 47.5, 'pa': 144.5, 'dra':-1.5e-3, 'ddec':1e-3, 'rout': 1.71*2, 'disk_name': 'IMLup'}
Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88*2, 'disk_name': 'Elias27'}
Elias24 = {'inc': 29, 'pa': 45.7, 'dra':110.8e-3, 'ddec':-386.8e-3, 'rout': 1.03*2, 'disk_name': 'Elias24'}
HD163296 = {'inc':46.7, 'pa':133.33, 'dra':-2.8e-3, 'ddec':7.7e-3, 'rout':1.7*2, 'disk_name': 'HD163296' }
HD143006 = {'inc': 18.6, 'pa': 169, 'dra':-5.9e-3, 'ddec':21.7e-3, 'rout': 0.518*3, 'disk_name': 'HD143006'}

best_Elias27 = {'m': -3.297430763696686, 'c': 316115113583956.4, 'l': 44039.6957302716}
best_AS209 = {'m': -1.3620915465005585, 'c': 4681.0681578390195, 'l': 72454.15132114716}

data =  Elias27

dir = "./../../data/"
disk_name = data['disk_name']
data_file = dir +"uvtable_" + disk_name + "_continuum.npz"

inc = data['inc']
pa = data['pa']
dra = data['dra']
ddec = data['ddec']
rout = data['rout']

# Frank Parameters
N = 500

# Load data
data = np.load(data_file)
u, v, Re, Imag, Weights = data["u"], data["v"], data["Re"], data["Im"], data["w"]
Vis = Re + Imag*1j

uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

print("Running Frank2D on simulated asymmetric disk data")

geom = Geometry(inc, pa, dra, ddec)
frank2d = Frank2D(N, rout, geom)

import numpy as np
from scipy.special import j0, j1, jn

def synthetic_uvtable_asymmetric(u, v,
                                    sigma_core=0.04,
                                    R_rings=[0.4, 0.6, 0.8],
                                    width_rings=[0.05, 0.02, 0.05],
                                    amp_core=0.1,
                                    amp_rings=[0.05, 0.04, 0.03],
                                    m_modes=[1, 0, 0],        # orden de asimetría por anillo
                                    theta0=[np.pi, 0, 0],  # ángulo central de cada asimetría
                                    amp_asym=[0.5, 0.3, 0.3],        # fuerza relativa de asimetría
                                    rad_to_arcsec=206265.0,
                                    weights=None):
    """
    Generate synthetic uvtable with asymmetric Gaussian-width rings (Fourier-domain model).
    Asymmetry is represented as a low-order azimuthal modulation (m-th mode).
    """
    baseline = np.sqrt(u**2 + v**2)
    phi_uv = np.arctan2(v, u)  # ángulo en el plano uv
    
    # --- Core (centrally symmetric Gaussian) ---
    V_core = amp_core * np.exp(-2 * (np.pi * (sigma_core / rad_to_arcsec) * baseline)**2)
    
    # --- Rings with azimuthal asymmetry ---
    V_rings = np.zeros_like(baseline, dtype=complex)
    
    for A, R, w, m, th0, Aasym in zip(amp_rings, R_rings, width_rings, m_modes, theta0, amp_asym):
        arg = 2 * np.pi * (R / rad_to_arcsec) * baseline
        blur = np.exp(-2 * (np.pi * (w / rad_to_arcsec) * baseline)**2)
        
        # componente axisimétrica (m=0)
        V0 = A * j0(arg) * blur
        
        # componente asimétrica (modo m)
        Vm = Aasym * A * jn(m, arg) * blur * np.exp(1j * m * (phi_uv - th0))
        
        V_rings += V0 + Vm
    
    # --- Combine core + rings ---
    V_total = V_core + V_rings
    Vis = V_total  # complejo
    
    if weights is None:
        weights = np.ones_like(u)
    
    uvtable = {
        'u': u,
        'v': v,
        'vis': Vis,
        'weights': weights
    }
    
    return uvtable


frank2d.process_vis(uvtable)

u, v = frank2d.gridded_data['u'], frank2d.gridded_data['v']

synthetic_data = synthetic_uvtable_asymmetric(u, v)

# To GPU

from frank2d_gpu.frank2d_gpu import Frank2D
from frank2d_gpu.geometry import Geometry
from frank2d_gpu.fourier2d_gpu import FourierTransform2D
from frank2d_gpu.preprocess_vis_gpu import Gridding
from frank2d_gpu.plot_gpu import Plot
from frank2d_gpu.constants import rad_to_arcsec, deg_to_rad
from frank2d_gpu.posterior_optimization_gpu import MAPEstimator

import cupy as cp

synthetic_data_gpu = {
    'u': cp.asarray(synthetic_data['u']),
    'v': cp.asarray(synthetic_data['v']),
    'vis': cp.asarray(synthetic_data['vis']),
    'weights': cp.asarray(synthetic_data['weights'])

}

geom = Geometry(0, 0, 0, 0)

frank2d2 = Frank2D(N, rout, geom)
frank2d2.process_vis(synthetic_data_gpu)

start_time = time.time()

initial_guess = {'m': -0.8, 'logl': 4}
N_opt = 80
print("Starting MAP search with initial guess: ", initial_guess, " and N = ", N_opt)
frank2d2.search_MAP(initial_guess = initial_guess, N = N_opt)

end_time = time.time()
print("MAP search time: ", (end_time - start_time)/60, " minutes")

print("Best value: ", frank2d2.MAP)




