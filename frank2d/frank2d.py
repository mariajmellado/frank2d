from .constants import rad_to_arcsec, deg_to_rad
from .fourier2d import FourierTransform2D
from .geometry import Geometry
from .preprocess_vis import Gridding
from .fitting import IterativeSolverMethod
from .gaussian_process import SquaredExponential, Wendland

from frank.radial_fitters import FrankFitter
from frank.geometry import SourceGeometry
from frank.plot import sweep_profile

import numpy as np
import matplotlib.pyplot as plt
import time

"""
Frank2D: A class to perform 2D visibility fitting using Gaussian Processes.
This is the main module of the Frank2D package. 
"""

class Frank2D(object):
    def __init__(self, N, Rmax, geom):
        """
        Initialize the Frank2D class.
        Parameters:
        -----------
        N : int
            Number of collocation points in each dimension (image will be NxN).
        Rmax : float
            Radius of the image in arcseconds.
        geom : Geometry object
            Geometry object containing source geometry parameters.
        """
        self._N =  N
        self._N2 = self._N*self._N
        self._Rmax = Rmax/rad_to_arcsec
        self._Geometry = geom
        self._FT = FourierTransform2D(self._Rmax, self._N, self._Geometry)

        self._gridded_data = None
        self._gridded_data_postprocess = None
        self._kernel_info = None
        self._x0 = None
        self._solver = None
        self._linear_system = None

        self._set_guess = False
        self._set_kernel = False
        self._set_gridded_data = False
        self._set_fit_method = False

        self.sol_visibility = None
        self.sol_intensity = None

    def set_kernel(self, type_kernel, kernel_params):
        """
        Setter of the Gaussian Process kernel.
        Parameters:
        -----------
        type_kernel : str
            Type of kernel to use ('SquareExponential' or 'Wendland').
        kernel_params : list
            Parameters for the kernel, amplitude (given by m and c) and length scale (l).
        """
        print("Setting GP Kernel " + type_kernel + "...")

        kernel_types_allowed = ['SquareExponential', 'Wendland']
        
        if type_kernel not in kernel_types_allowed:
            raise ValueError("Unknown kernel type. Use 'SquareExponential' or 'Wendland'.")
        else:
            self._kernel_info = {"type": type_kernel, "params": kernel_params}

        self._set_kernel = True

    def get_kernel(self, kernel_info, u, v, u2 = None, v2 = None):
        """
        Getter of the Gaussian Process kernel.
        Parameters:
        -----------
        kernel_info : dict
            Dictionary containing kernel type and parameters.
        u : 1D array
            u coordinates of the visibility data.
        v : 1D array
            v coordinates of the visibility data.
        u2 : 1D array
            Optional second set of u coordinates to compute correlation.
        v2 : 1D array
            Optional second set of v coordinates to compute correlation.

        Returns:
        --------
        Kernel : CovarianceMatrix object
            Correlation matrix operator.
        """
        type_kernel = kernel_info["type"]
        params = kernel_info["params"]

        types_allowed = ['SquaredExponential', 'Wendland']

        if type_kernel == 'SquareExponential':
            return SquaredExponential(params, u, v, u2 = u2, v2 = v2)
        elif type_kernel == 'Wendland':
            return Wendland(params, u, v, u2 = u2, v2 = v2)

    def set_guess(self, guess):
        """
        Setter of the initial guess for the visibility fitting.
        Parameters:
        -----------
        guess : array
            Initial guess for the visibility model.
        """

        if guess is not None: # If user provides an initial guess.
            print("Setting guess...")
            index = self.gridded_data_postprocess["index_weighted"]
            self._x0 = guess[index].flatten()
        else:
            self._x0 = guess

        self._set_guess = True

    def set_fit_method(self, method, x0, maxiter, rtol):
        """
        Setter of the fitting method for the iterative solver.
        Parameters:
        -----------
        method : str
            Iterative solver method to use ('cg', 'bicgstab', etc.).
        x0 : array
            Initial guess for the iterative solver.
        maxiter : int
            Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved.
        rtol : float
            Relative tolerance for the iterative solver.
        """
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
                                             method = method, 
                                             x0 = self._x0,
                                             maxiter = maxiter,
                                             rtol = rtol)
        self._set_fit_method = True
                                            
    def set_gridded_data(self, u, v, Vis, Weights):
        """
        Setter of the gridded visibility data.
        Parameters:
        -----------
        u : array, size (N2, 1), unit: lambda
            Gridded u coordinates.
        v : array,  size (N2, 1), unit: lambda
            Gridded v coordinates.
        Vis : array, size (N2, 1), unit: Jy
            Gridded visibility data.
        Weights : array, size (N2, 1), unit: 1/Jy^2
            Gridded weights.
        """
        print("Setting gridded data...")

        self._gridded_data = {"u": u, "v": v, "vis": Vis, "weights": Weights}

        self._set_gridded_data = True

    def process_vis(self, u, v, Vis, Weights, hermitian = True):
        """
        Process the visibility data by gridding and dividing weighted data and non-weighted data.
        Parameters:
        -----------
        u : 1D array, size relative to UV table, unit: lambda
            u coordinates of the visibility data.
        v : 1D array, size relative to UV table, unit: lambda
            v coordinates of the visibility data.
        Vis : 1D array, size relative to UV table, unit: Jy
            Visibility data.
        Weights : 1D array, size relative to UV table, unit: 1/Jy^2
            Weights of the visibility data.
        hermitian : bool
            Whether to enforce Hermitian symmetry.

        Returns:
        None
        """
        if not self._set_gridded_data:
            grid = Gridding(self._N, self._Rmax, self._FT, self._Geometry)
            u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights,
                                                                          hermitian = hermitian)
            self.set_gridded_data(u_gridded, v_gridded, vis_gridded, weights_gridded)
        
        self.process_gridded_vis()
        
    def process_gridded_vis(self):
        """
        Postprocess the gridded visibility data by separating weighted and non-weighted data.
        """
        u_gridded = self._gridded_data['u']
        v_gridded = self._gridded_data['v']
        vis_gridded = self._gridded_data['vis']
        weights_gridded = self._gridded_data['weights']

        index_w = np.argwhere(weights_gridded != 0).flatten()
        index_uw = np.argwhere(weights_gridded == 0).flatten()

        # data with weights != 0.
        u_weighted = u_gridded[index_w]
        v_weighted = v_gridded[index_w]
        vis_weighted  = vis_gridded[index_w]
        weights_weighted = weights_gridded[index_w]

        # data with weights == 0.
        u_unweighted = u_gridded[index_uw]
        v_unweighted = v_gridded[index_uw]
        vis_unweighted  = vis_gridded[index_uw]
        weights_unweighted = weights_gridded[index_uw]

        data_w = {"u": u_weighted, "v": v_weighted, "vis": vis_weighted, "weights": weights_weighted}
        data_uw = {"u": u_unweighted, "v": v_unweighted, "vis": vis_unweighted, "weights": weights_unweighted}
    
        self._gridded_data_postprocess = {"weighted": data_w, "unweighted": data_uw,
                                           "index_weighted": index_w, "index_unweighted": index_uw }
      
    def build_full_visibility_model(self):
        """
        Build the full visibility model from the weighted and non-weighted solution.
        """
        print("Building full visibility model...")

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
        Kernel2 = self.get_kernel(self._kernel_info, u_w, v_w, u2 = u_uw, v2 = v_uw) 
        S_12_T = Kernel2.sparse()

        # Build full visibility model.
        V1 = S11.matvec(self._sol_visibility_weighted)
        V2 = S_12_T.matvec(self._sol_visibility_weighted)

        V_full = np.zeros((self._N, self._N), dtype="c16")

        data_coords_w = np.unravel_index(index_w, (self._N, self._N))
        data_coords_uw = np.unravel_index(index_uw, (self._N, self._N))

        V_full[data_coords_w] = V1
        V_full[data_coords_uw] = V2

        return V_full

    def fit(self, u = None, v = None, Vis = None, Weights = None,
            type_kernel = 'Wendland', kernel_params = {'m': -2, 'c': 1e8, 'l': 5e4},
            method = 'bicgstab', x0 = None, maxiter = 50000, rtol = 1e-9,
            hermitian = True):
        """
        Fit the visibility data using Gaussian Processes.
        Parameters:
        -----------
        u : 1D array, unit: lambda
            u coordinates of the visibility data.
        v : 1D array, unit: lambda
            v coordinates of the visibility data.
        Vis : 1D array,  unit: Jy
            Visibility data.
        Weights : 1D array, unit: 1/Jy^2
            Weights of the visibility data.
        type_kernel : str
            Type of kernel to use ('SquareExponential' or 'Wendland').
        kernel_params : dict
            Parameters for the kernel, amplitude (given by m and c) and length scale (l).
        method : str
            Iterative solver method to use ('cg', 'bicgstab', etc.).
        x0 : array
            Initial guess for the iterative solver.
        maxiter : int
            Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved.
        rtol : float
            Relative tolerance for the iterative solver.
        hermitian : bool
            Whether to enforce Hermitian symmetry.
        """

        if not self._set_gridded_data:
            if (u is None) or (v is None) or (Vis is None) or (Weights is None):
                raise ValueError("If gridded data is not set, u, v, Vis and Weights must be provided.")
            self.process_vis(u, v, Vis, Weights, hermitian = hermitian)

        if not self._set_guess:
            self.set_guess(x0)
            
        if not self._set_kernel:
            self.set_kernel(type_kernel, kernel_params)

        if not self._set_fit_method:
            self.set_fit_method(method, x0, maxiter, rtol)

        self._sol_visibility_weighted = self._solver.run()
        self.sol_visibility = self.build_full_visibility_model()

        self.sol_intensity = self._FT.fast_transform(self.sol_visibility, direction = "backward")

    def frank1d(self, u, v, Vis, Weights, alpha = 1.3, w_smooth = 1e-3, n_pts = 300):
        """
        Perform a 1D Frank fit on the visibility data.
        Parameters:
        -----------
        u : 1D array, unit: lambda
            u coordinates of the visibility data.
        v : 1D array, unit: lambda
            v coordinates of the visibility data.
        Vis : 1D array, unit: Jy
            Visibility data.
        Weights : 1D array, unit: 1/Jy^2
            Weights of the visibility data.
        alpha : float
            Regularization hyparameter for Frank.
        w_smooth : float
            Smoothing weight hyperparameter for Frank.
        n_pts : int
            Number of radial points for Frank.

        Returns:
        --------
        vis_fit_1d : 1D array, unit: Jy
            Fitted visibility data using 1D Frank.
        """
        print("Performing 1D Frank fit...")
        geom = self._Geometry
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        Rout = self._Rmax*rad_to_arcsec
        geom_f1d = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)
        FF = FrankFitter(Rout, n_pts, geom_f1d, alpha = alpha, weights_smooth = w_smooth)

        if not self._set_gridded_data:
            self.preprocess_vis(u, v, Vis, Weights, hermitian = True)
        u_gridded, v_gridded = self._gridded_data["u"], self._gridded_data["v"]
        vis_gridded = self._gridded_data["vis"]
        weights_gridded = self._gridded_data["weights"]

        sol = FF.fit(u_gridded, v_gridded, vis_gridded, weights_gridded)
        vis_fit_1d = sol.predict(u_gridded, v_gridded, sol.mean, geometry = geom_f1d)

        return vis_fit_1d
    
    @property
    def gridded_data(self):
        """ Gridded visibility data in a dict. """
        return self._gridded_data
    
    @property
    def visibility_model(self):
        """ 2D visibility model in Jy. """
        return self.sol_visibility.reshape(self._N, self._N)
    
    @property
    def intensity_model(self):
        """ 2D intensity model in Jy/sr """
        return self.sol_intensity.reshape(self._N, self._N).real
