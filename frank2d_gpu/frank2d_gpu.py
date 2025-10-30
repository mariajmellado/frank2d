from .constants import rad_to_arcsec, deg_to_rad
from .fourier2d_gpu import FourierTransform2D
from .geometry import Geometry
from .preprocess_vis_gpu import Gridding
from .fitting_gpu import IterativeSolverMethod
from .gaussian_process_gpu import Wendland
from .posterior_optimization_gpu import MAPEstimator


from frank.radial_fitters import FrankFitter
from frank.geometry import SourceGeometry
from frank.plot import sweep_profile

import cupy as cp
import time

"""
Frank2D: A class to perform 2D visibility fitting using Gaussian Processes.
This is the main module of the Frank2D package. 
"""

class Frank2D(object):
    def __init__(self, N, Rmax, geom):
        """
        Initialize the Frank2D class (CuPy version).
        N : int
        Rmax : float (arcseconds)
        geom : Geometry object
        """
        self._N = N
        self._Nx = N
        self._Ny = N
        self._N2 = self._N * self._N
        self._Rmax = Rmax / rad_to_arcsec
        self._Geometry = geom
        self._FT = FourierTransform2D(self._Rmax, self._N, self._Geometry)

        self._set_guess = False
        self._set_kernel = False
        self._set_gridded_data = False
        self._set_fit_method = False
        self._set_MAP_estimator = False

        self._sol_visibility = None
        self._sol_intensity = None
        self._sol_visibility_weighted = None
        self._sol_f1d = None
        self._MAP = None
        self._MAP_estimator = None

    def set_kernel(self, type_kernel, kernel_params):
        print("Setting GP Kernel " + type_kernel + "...")
        kernel_types_allowed = ['SquareExponential', 'Wendland']

        if type_kernel not in kernel_types_allowed:
            raise ValueError("Unknown kernel type. Use 'SquareExponential' or 'Wendland'.")
        else:
            self._kernel_info = {"type": type_kernel, "params": kernel_params}

        self._set_kernel = True

    def get_kernel(self, kernel_info, u, v, u2=None, v2=None):
        type_kernel = kernel_info["type"]
        params = kernel_info["params"]

        if type_kernel == 'SquareExponential':
            return SquaredExponential(params, u, v, u2=u2, v2=v2)
        elif type_kernel == 'Wendland':
            return Wendland(params, u, v, u2=u2, v2=v2)
        else:
            raise ValueError("Unknown kernel type")

    def set_guess(self, guess):
        if guess is not None:
            print("Setting guess...")
            index = self.gridded_data_postprocess["index_weighted"]
            # ensure guess is cupy and select indices
            guess_cp = cp.asarray(guess)
            self._x0 = guess_cp[index].flatten()
        else:
            self._x0 = None

        self._set_guess = True

    def set_fit_method(self, method, x0, maxiter, rtol):
        if not self._set_kernel:
            raise ValueError("Set kernel before setting fit method.")

        data_weighted = self._gridded_data_postprocess["weighted"]
        u = data_weighted["u"]
        v = data_weighted["v"]
        Vis = data_weighted["vis"]
        Weights = data_weighted["weights"]

        # Create covariance matrix operator.
        Kernel = self.get_kernel(self._kernel_info, u, v)

        self._solver = IterativeSolverMethod(u, v, Vis, Weights,
                                             Kernel,
                                             method=method,
                                             x0=self._x0,
                                             maxiter=maxiter,
                                             rtol=rtol)
        self._set_fit_method = True

    def set_gridded_data(self, u, v, Vis, Weights):
        print("Setting gridded data...")
        # Convert to cupy arrays (if they are numpy)
        self._gridded_data = {
            "u": u,
            "v": v,
            "vis": Vis,
            "weights": Weights
        }
        self._set_gridded_data = True

    def set_MAP_estimator(self, MAP_estimator):
        print("Setting MAP estimator...")
        if not isinstance(MAP_estimator, MAPEstimator):
            raise ValueError("MAP_estimator must be an instance of MAPEstimator class.")
        self._MAP_estimator = MAP_estimator
        self._set_MAP_estimator = True

    def process_vis(self, data, hermitian=True):
        if not self._set_gridded_data:
            grid = Gridding(self._Rmax, self._FT, self._Geometry)
            try:
                u = data["u"]
                v = data["v"]
                Vis = data["vis"]
                Weights = data["weights"]
            except KeyError:
                raise ValueError("data dictionary must contain 'u', 'v', 'vis' and 'weights' keys.")

            u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights,
                                                                          hermitian=hermitian)
            # grid.run may return numpy arrays; convert to cupy
            self.set_gridded_data(u_gridded, v_gridded, vis_gridded, weights_gridded)

        self.process_gridded_vis()

    def process_gridded_vis(self):
        u_gridded = self._gridded_data['u']
        v_gridded = self._gridded_data['v']
        vis_gridded = self._gridded_data['vis']
        weights_gridded = self._gridded_data['weights']

        W = weights_gridded.reshape(self._Nx, self._Ny)
        mask = (W != 0)

        r = mask.ravel(order="C")
        index_w = cp.where(r)[0]
        index_uw = cp.where(~r)[0]

        # data with weights != 0.
        u_weighted = u_gridded[index_w]
        v_weighted = v_gridded[index_w]
        vis_weighted = vis_gridded[index_w]
        weights_weighted = weights_gridded[index_w]

        # data with weights == 0.
        u_unweighted = u_gridded[index_uw]
        v_unweighted = v_gridded[index_uw]
        vis_unweighted = vis_gridded[index_uw]
        weights_unweighted = weights_gridded[index_uw]

        data_w = {"u": u_weighted, "v": v_weighted, "vis": vis_weighted, "weights": weights_weighted}
        data_uw = {"u": u_unweighted, "v": v_unweighted, "vis": vis_unweighted, "weights": weights_unweighted}

        self._gridded_data_postprocess = {
            "weighted": data_w,
            "unweighted": data_uw,
            "index_weighted": index_w,
            "index_unweighted": index_uw
        }

    def build_full_visibility_model(self):
        print("Building full visibility model...")
        start = time.time()

        index_w = self._gridded_data_postprocess["index_weighted"]
        data_w = self._gridded_data_postprocess["weighted"]
        u_w = data_w["u"]
        v_w = data_w["v"]
        vis_w = data_w["vis"]
        weights_w = data_w["weights"]

        Kernel1 = self.get_kernel(self._kernel_info, u_w, v_w)
        S11 = Kernel1.sparse()

        index_uw = self._gridded_data_postprocess["index_unweighted"]
        data_uw = self._gridded_data_postprocess["unweighted"]
        u_uw = data_uw["u"]
        v_uw = data_uw["v"]
        vis_uw = data_uw["vis"]
        weights_uw = data_uw["weights"]
        Kernel2 = self.get_kernel(self._kernel_info, u_w, v_w, u2=u_uw, v2=v_uw)
        S_12_T = Kernel2.sparse()

        # Build full visibility model.
        V1 = S11.matvec(self._sol_visibility_weighted)
        V2 = S_12_T.matvec(self._sol_visibility_weighted)

        V_full = cp.zeros((self._Nx, self._Ny), dtype=cp.complex128)

        data_coords_w = cp.unravel_index(index_w, (self._Nx, self._Ny))
        data_coords_uw = cp.unravel_index(index_uw, (self._Nx, self._Ny))

        V_full[data_coords_w] = V1
        V_full[data_coords_uw] = V2

        end = time.time()
        execution_time = end - start
        print(f'--> time build full visibility = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        return V_full

    def fit(self, data=None,
            type_kernel='Wendland', kernel_params={'m': -2, 'c': 1e8, 'l': 5e4},
            method='bicgstab', x0=None, maxiter=50000, rtol=1e-9,
            hermitian=True, run_from_scratch=True):

        if not self._set_gridded_data:
            if not data:
                raise ValueError("If gridded data is not set, u, v, Vis and Weights must be provided.")
            try:
                u = data["u"]
                v = data["v"]
                Vis = data["vis"]
                Weights = data["weights"]
            except KeyError:
                raise ValueError("data dictionary must contain 'u', 'v', 'vis' and 'weights' keys.")
            self.process_vis(data, hermitian=hermitian)

        if run_from_scratch:
            self._set_guess = False
            self._set_kernel = False
            self._set_fit_method = False

        if not self._set_guess:
            self.set_guess(x0)

        if not self._set_kernel:
            self.set_kernel(type_kernel, kernel_params)

        if not self._set_fit_method:
            self.set_fit_method(method, x0, maxiter, rtol)

        self._solver.run()
        self._sol_visibility_weighted = self._solver.sol
        self._sol_visibility = self.build_full_visibility_model()

        self._sol_intensity = self.transform(self._sol_visibility)

    def search_MAP(self, data=None,
                   initial_guess={'m': -2, 'logl': 4},
                   N=50):
        if data is  None:
            print("Using existing visibility data...")
            if not self._set_gridded_data:
                raise ValueError("Gridded data is not set, u, v, Vis and Weights must be provided.")
            data = self._gridded_data

        if not self._set_MAP_estimator:
            self._MAP_estimator = MAPEstimator(self._Rmax * rad_to_arcsec, self._Geometry, N=N)
            self._set_MAP_estimator = True

        self._MAP_estimator.optimize(data, initial_guess)
        self._MAP = self._MAP_estimator.MAP

    def transform(self, vis, direction="backward"):
        return self._FT.fast_transform(vis, direction=direction)

    def frank1d(self, data=None,
                alpha=1.05, w_smooth=1e-3, n_pts=300,
                rout=None, geom=None):
        print('Performing 1D Frank fit...\n')
        print(r'$\alpha$ = ', str(alpha), r' and $w_{smooth}$ = ', str(w_smooth) + '\n')
        print('N = ', str(n_pts), r' and $R_{max}$ = ', str(rout))
        if geom is None:
            geom = self._Geometry
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        if rout is None:
            rout = self._Rmax * rad_to_arcsec
        geom_f1d = SourceGeometry(inc=inc, PA=pa, dRA=dra, dDec=ddec)
        FF = FrankFitter(rout, n_pts, geom_f1d, alpha=alpha, weights_smooth=w_smooth)

        if data is None:
            if not self._set_gridded_data:
                raise ValueError("No gridded data. Provide data or run process_vis first.")
            u, v = self._gridded_data["u"], self._gridded_data["v"]
            vis = self._gridded_data["vis"]
            weights = self._gridded_data["weights"]
        else:
            u, v = data["u"], data["v"]
            vis = data["vis"]
            weights = data["weights"]

        self._sol_f1d = FF.fit(u, v, vis, weights)
        return self._sol_f1d

    @property
    def gridded_data(self):
        return self._gridded_data

    @property
    def u(self):
        return self._FT.u

    @property
    def u_grid(self):
        return self._FT._Un.reshape((self._Nx, self._Ny), order='C')

    @property
    def v(self):
        return self._FT.v

    @property
    def v_grid(self):
        return self._FT._Vn.reshape((self._Nx, self._Ny), order='C')

    @property
    def x(self):
        return self._FT._x

    @property
    def x_grid(self):
        return self._FT._Xn.reshape((self._Nx, self._Ny), order='C')

    @property
    def y(self):
        return self._FT._y

    @property
    def y_grid(self):
        return self._FT._Yn.reshape((self._Nx, self._Ny), order='C')

    @property
    def visibility_model(self):
        return self._sol_visibility.reshape((self._Nx, self._Ny), order='C')

    @property
    def Rmax(self):
        return self._Rmax * rad_to_arcsec

    @property
    def intensity_model(self):
        return self._sol_intensity.reshape((self._Nx, self._Ny), order='C').real

    @property
    def FT(self):
        return self._FT

    @property
    def Geometry(self):
        return self._Geometry

    @property
    def MAPEstimator(self):
        if self._MAP_estimator is None:
            raise ValueError("No MAP estimator found. Run search_MAP first.")
        return self._MAP_estimator

    @property
    def MAP(self):
        if self._MAP is None:
            raise ValueError("No MAP found. Run search_MAP first.")
        return self._MAP

    @property
    def N(self):
        return self._N

    @property
    def sol_f1d(self):
        if self._sol_f1d is None:
            raise ValueError("No 1D Frank solution found. Run frank1d first.")
        return self._sol_f1d
