

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

disc = 'Elias27'
print("-----> Disc type: {}".format(disc))

dir = "./../../data/uvtable/"
if disc == 'SimulatedBlob':
    data_file = dir + "uvtable_SimulatedBlob.txt"
    inc = 0
    pa = 0
    dra = 0
    ddec = 0
    rout = 1
    Rmax = 3*rout

    file = cp.loadtxt(data_file, unpack=True)
    u, v, Re, Im, Weights = file
    Vis = Re + Im*1j

    #[50, 60, 70, 80, 90]
    #Times taken:  [7.870652735233307, 37.83131151596705, 56.695450735092166, 66.12822169860205, 140.7839174469312]
    #MAPs:  [{'m': -1.7118115146053943, 'c': 1107386.7020548182, 'l': 21833.50491835495}, {'m': -1.7900507315194405, 'c': 4208615.485734088, 'l': 22746.607839035685}, {'m': -2.205758019904303, 'c': 1360404752.1630862, 'l': 22817.1219726599}, {'m': -2.9537030067142087, 'c': 40546491180778.625, 'l': 22781.422417760932}, {'m': -3.7041542492704664, 'c': 1.3882769163865464e+18, 'l': 22906.177349190206}]
    #Qmax:  [array(1215427.02681209), array(1458512.43217451), array(1701597.83753693), array(1944683.24289935), array(2187768.64826177)]
if disc == 'AS209':
    AS209 = {'inc': 34.97, 'pa': 85.76, 'dra':1.9e-3, 'ddec':-2.5e-3, 'rout': 1.2, 'disk_name': 'AS209'}
    data_file = dir +"uvtable_" + disc + "_continuum.npz"

    inc = AS209['inc']
    pa = AS209['pa']
    dra = AS209['dra']
    ddec = AS209['ddec']
    rout = AS209['rout']
    Rmax = 3*rout

    file = cp.load(data_file)
    u, v, Re, Im, Weights = file['u'], file['v'], file['Re'], file['Im'], file['w']
    Vis = Re + Im*1j

    # gpu ------
    #[100, 50, 70, 60, 80]
    #Times taken:  [373.7676817178726, 13.976889828840891, 68.03289994398753, 25.795325084527335, 123.44866792360942]
    #MAPs:  [{'m': -1.027405913813779, 'c': 55.22531124827761, 'l': 35674.21039835561}, {'m': -2.2478888378636794, 'c': 241548327.67926273, 'l': 37853.96840857266}, {'m': -1.5813930105203955, 'c': 63111.391583057986, 'l': 38055.23839286482}, {'m': -1.7999547305871808, 'c': 950820.5934736129, 'l': 38844.40616729767}, {'m': -1.3385364980306098, 'c': 2972.974797969014, 'l': 38691.341836030646}]
    #Qmax:  [array(2025711.71135349), array(1012855.85567674), array(1417998.19794744), array(1215427.02681209), array(1620569.36908279)]

if disc =='Elias27':
    Elias27 = {'inc': 56.2, 'pa': 118.8, 'dra':-5e-3, 'ddec':-8e-3, 'rout': 1.88, 'disk_name': 'Elias27'}
    data_file = dir +"uvtable_" + disc + "_continuum.npz"
    inc = Elias27['inc']
    pa = Elias27['pa']
    dra = Elias27['dra']
    ddec = Elias27['ddec']
    rout = Elias27['rout']
    Rmax = 3*rout

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
Qmax = max(cp.hypot(u,v))

import math 
N = math.floor(5* Rmax/rad_to_arcsec * Qmax)

N_opt = [50, 70, 60, 80, 100]
maps = []
frank2d_objects = []
times = []
Qmaxs = []
print("x-------------------------------------------------x")
print("For N_opt array: ", N_opt)
print("Running with Rmax = {} arcsec".format(Rmax))

for i in N_opt:
    start_time = time.time()
    print("---> Processing for N = {}".format(i))
    frank2d = Frank2D(N, Rmax)
    frank2d.process_vis(uvtable)
    initial_guess = {'m': -2, 'logl': 4, 'p': -2}
    frank2d.search_MAP(initial_guess= initial_guess, N = i)

    end_time = time.time()
    total_time = (end_time - start_time)/60

    frank2d_objects.append(frank2d)
    times.append(total_time)

    best = frank2d.MAP
    maps.append(best)
    print("     + MAP: "+ str(best))

    MAPEs = frank2d.MAPEstimator

    print("     + Saving  frank2d object...")
    import pickle
    dir_ = "../../data/exp/N_opt/"
    with open(dir_+ disc+ '_optGPU_Nopt{}.pkl'.format(i), 'wb') as f:
        pickle.dump(frank2d, f)

    GP =  MAPEs.GaussianModel
    DFT = GP.DFT2
    qmax = cp.max(DFT.Qmax)
    Qmaxs.append(qmax)

    print("     + Qmax: " + str(qmax))
    print("         + Time taken: {:.2f} mins".format(i, total_time))
    print("x-------------------------------------------------x")

print(N_opt)
print("Times taken: ", times)
print("MAPs: ", maps)
print("Qmaxs: ", Qmaxs)