from constants import rad_to_arcsec, deg_to_rad
from fourier2d import FourierTransform2D
from geometry import Geometry
from preprocess_vis import Gridding
from gaussian_process import SquareExponential, Wendland
from fitting import IterativeSolverMethod

from frank.radial_fitters import FrankFitter
from frank.geometry import SourceGeometry
from frank.plot import sweep_profile

import numpy as np
import matplotlib.pyplot as plt
import time


class Frank2D(object):
    def __init__(self, N, Rmax, geom):
        self._Nx = self._Ny =  N
        self._N2 = self._Nx*self._Ny
        self._Rmax = Rmax/rad_to_arcsec
        self._Geometry = geom
        self._FT = FourierTransform2D(self._Rmax, self._Nx, self._Geometry)

        self._gridded_data = None
        self._kernel = None
        self._x0 = None
        self._method = None
        self._frank1d_guess = None

        self._set_guess = False
        self._set_kernel = False
        self._set_gridded_data = False

        self.sol_visibility = None
        self.sol_intensity = None

    """ Setters """
    def set_kernel(self, type_kernel, kernel_params):
        print("Setting kernel: " + type_kernel + '...')
        if type_kernel == 'SquareExponential':
            self._kernel =  SquareExponential(self._Nx, self._Rmax, kernel_params, self._FT).kernel_polar
        elif type_kernel == 'Wendland':
            self._kernel = Wendland(self._Nx, self._Rmax, kernel_params, self._FT).kernel
        
        self._set_kernel = True

    def set_guess(self, Vis):
        print("Setting guess...")
        self._x0 = Vis
        self._set_guess = True

    def set_fit_method(self, type_kernel, kernel_params, method, rtol):
        if not self._set_kernel:
            self.set_kernel(type_kernel = type_kernel, kernel_params = kernel_params)

        if not self._set_guess:
            print("Setting guess... Visibilities gridded")
            self._x0 = self._gridded_data["vis"]
        
        self._method = IterativeSolverMethod(self._gridded_data["vis"], self._gridded_data["weights"],
                                             self._kernel, self._FT,
                                             method = method, x0 = self._x0, rtol = rtol)
                                            
    def set_gridded_data(self, u, v, Vis, Weights):
        print("Setting gridded data...")
        self._gridded_data = {"u": u, "v": v, "vis": Vis, "weights": Weights}
        self._set_gridded_data = True

    """ Getters """
    def preprocess_vis(self, u, v, Vis, Weights, hermitian = True, vis_component = "all"):
        if not self._set_gridded_data:
            start_time = time.time()
            grid = Gridding(self._Nx, self._Rmax, self._FT, self._Geometry)
            u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights,
                                                                          hermitian = hermitian)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

            if vis_component == "real":
                vis_gridded = vis_gridded.real
            elif vis_component == "imag":
                vis_gridded = vis_gridded.imag*1j

            self.set_gridded_data(u_gridded, v_gridded, vis_gridded, weights_gridded)

    def fit(self, u, v, Vis, Weights, type_kernel = 'SquareExponential', kernel_params = [-5, 60, 1e4],
            rtol = 1e-7, method = 'cg', frank1d_guess = False,
            hermitian = True, vis_component = "all"):

        # Visibility model.
        print("Gridding...")
        self.preprocess_vis(u, v, Vis, Weights, hermitian = hermitian, vis_component = vis_component)

        # Frank1D guess.
        if frank1d_guess:
            print("Setting guess with Frank1D ...")
            self.frank1d(u, v, Vis, Weights)
            self.set_guess(self._frank1d_guess)

        print("Setting fit with " + method + " ...")
        self.set_fit_method(type_kernel, kernel_params, method, rtol)
        
        print("Fitting...")
        vis_model = self._method.solve()
        self.sol_visibility = vis_model 
    
    def fft(self, transform = "2fft"):
        # Image model.
        vis_model = self.sol_visibility
        I_model = None
        print("Inverting with " + transform + " ...")
        if transform == "2fft":
            start_time = time.time()
            I_model =  self._FT.fast_transform(vis_model, direction = 'backward')
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
        elif transform == "2dft":
            start_time = time.time()
            I_model =  self._FT.transform(vis_model, direction = 'backward')
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        self.sol_intensity = I_model.real.reshape(self._Nx, self._Ny)

    def frank1d(self, u, v, Vis, Weights, alpha = 1.3, w_smooth = 1e-3, n_pts = 300):
        geom = self._Geometry
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        Rout = self._Rmax*rad_to_arcsec
        geom_f1d = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)
        FF = FrankFitter(Rout, n_pts, geom_f1d, alpha = alpha, weights_smooth = w_smooth)
        sol = FF.fit(u, v, Vis, Weights)

        """
        u_gridded, v_gridded = None, None

        if geom._deproject: # Frank1D needs deprojected Vis.
            Geometry_ = Geometry(inc, pa, dra, ddec, deproject = False)
            FT_ = FourierTransform2D(self._Rmax, self._Nx, Geometry_)
            Grid_ = Gridding(self._Nx, self._Rmax, FT_, Geometry_)
            u_gridded, v_gridded, _, _ = Grid_.run(u, v, Vis, Weights)
        else:
            u_gridded, v_gridded = self._gridded_data["u"], self._gridded_data["v"]

        u, v = u_gridded, v_gridded
        vis_fit_1d = sol.predict(u, v, sol.mean, geometry = geom_f1d)
        self._frank1d_guess = vis_fit_1d
        """
        return sol