

import os
import sys
import cupy as cp
current_dir =  os.getcwd()
# replace this following line in the path your frank2d files are.
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import time

#frank2d
from frank2d_gpu.frank2d_gpu import Frank2D
from frank2d_gpu.geometry import Geometry
from frank2d_gpu.fourier2d_gpu import FourierTransform2D
from frank2d_gpu.process_vis_gpu import Gridding
from frank2d_gpu.plot_gpu import Plot
from frank2d_gpu.constants import rad_to_arcsec, deg_to_rad
from frank2d_gpu.posterior_optimization_gpu import MAPEstimator

disc = 'Simulated'
print("-----> Disc type: {}".format(disc))

dir = "./../../data/"
if disc == 'Simulated':
    data_file = dir + "uvtable_nonsym_disc_blob_noisy.npz"
    inc = 0
    pa = 0
    dra = 0
    ddec = 0
    rout = 1*2

    file = cp.load(data_file)
    u, v, Vis, Weights = file['u'], file['v'], file['vis'], file['weights']
    # [50, 60, 70, 80]
    # Times taken:  [23.821291848023733, 83.7020370999972, 96.87331301768621, 208.44252157211304]
    # MAPs:  [{'m': -2.6928767634567086, 'c': 5565918924985.133, 'l': 85344.21491570085}, {'m': -3.1733216083914595, 'c': 4781178693980001.0, 'l': 83515.64050517132}, {'m': -3.5833382724264506, 'c': 1.4774087424414195e+18, 'l': 81076.42094700658}, {'m': -4.572459366888535, 'c': 1.0086866109221515e+24, 'l': 68720.91547274287}]
    # Qmax:  [array(1823140.54021814), array(2187768.64826177), array(2552396.7563054), array(2917024.86434902)]
if disc == 'AS209':
    AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2*2, 'disk_name': 'AS209'}
    data_file = dir +"uvtable_" + disc + "_continuum.npz"

    inc = AS209['inc']
    pa = AS209['pa']
    dra = AS209['dra']
    ddec = AS209['ddec']
    rout = AS209['rout']

    file = cp.load(data_file)
    u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
    Vis = Re + Im*1j

    # gpu ------
    # N_opt = [50, 60, 70, 80]
    #Times taken:  [30.174559326966605, 87.17014620701472, 70.0317242105802, 94.86205356518427]
    #MAPs:  [{'m': -1.3620943031521517, 'c': 4681.257567443593, 'l': 72454.22528223816}, {'m': -1.1589975280679916, 'c': 318.83329458711563, 'l': 67271.58875567082}, {'m': -0.8815982469048975, 'c': 8.290675382849015, 'l': 65295.905434193555}, {'m': -0.5965380764780932, 'c': 0.24061018817178012, 'l': 65963.98525534407}]
    #Qmax:  [array(1519283.78351512), array(1823140.54021814), array(2126997.29692116), array(2430854.05362419)]

if disc =='Elias27':
    Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88*2, 'disk_name': 'Elias27'}
    data_file = dir +"uvtable_" + disc + "_continuum.npz"
    inc = Elias27['inc']
    pa = Elias27['pa']
    dra = Elias27['dra']
    ddec = Elias27['ddec']
    rout = Elias27['rout']

    file = cp.load(data_file)
    u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
    Vis = Re + Im*1j
    # gpu ------
    # N_opt = [50, 60, 70, 80]
    # Times taken:  [19.343274784088134, 59.44294565518697, 75.24612582524618, 86.83972671429316]
    # MAPs:  [{'m': -3.2974334115230772, 'c': 316125239602758.7, 'l': 44039.627767028585}, {'m': -2.621078189858711, 'c': 54325279042.69005, 'l': 38993.94179547343}, {'m': -2.320894222847415, 'c': 1157395657.8725452, 'l': 35922.326961874816}, {'m': -2.0397167285521345, 'c': 33723204.88832001, 'l': 34841.2089126212}]
    # Qmax:  [array(969755.60649901), array(1163706.72779881), array(1357657.84909861), array(1551608.97039842)]
            
# Load data
uvtable = {'u': u, 'v': v, 'vis': Vis, 'weights': Weights }

geom = Geometry(inc, pa, dra, ddec)
N = 300

N_opt = [50, 60, 70, 80]
maps = []
frank2d_objects = []
times = []
Qmax = []

for i in N_opt:
    start_time = time.time()
    print("---> Processing for N = {}".format(i))
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

    print("     ---> Saving  frank2d object...")
    import pickle
    with open(dir+ disc+'_opt_Nopt_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(frank2d, f)

    GP =  MAPEs.GaussianModel
    DFT = GP.DFT2
    qmax = cp.max(DFT.Qmax)
    Qmax.append(qmax)

    print("     ---> Qmax: " + str(qmax))
    print("x-------------------------------------------------x")

print(N_opt)
print("Times taken: ", times)
print("MAPs: ", maps)
print("Qmax: ", Qmax)