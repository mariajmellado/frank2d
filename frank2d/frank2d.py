from .constants import rad_to_arcsec, deg_to_rad
from .fourier2d import FourierTransform2D
from .geometry import Geometry
from .process_vis import Gridding, PostProcess
from .fitting import IterativeSolverMethod
from .gaussian_process import SquaredExponential, Wendland
from .posterior_optimization import MAPEstimator

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
    def __init__(self, N, Rmax):
        """
        Initialize the Frank2D class.
        Parameters:
        -----------
        N : int
            Number of collocation points in each dimension (image will be NxN).
        Rmax : float
            Radius of the image in arcseconds.
        """
        self._N =  N
        self._Nx = N
        self._Ny = N
        self._N2 = self._N*self._N
        self._Rmax = Rmax/rad_to_arcsec
        self._FT = FourierTransform2D(self._Rmax, self._N)

        self._set_x0 = False
        self._set_kernel = False
        self._set_gridded_data = False
        self._set_fit_method = False
        self._set_MAP_estimator = False

        self._sol_visibility = None
        self._sol_intensity = None

    def set_kernel( self, kernel_type = 'wend', 
                    kernel_params = {'m': -2, 'c': 1e8, 'l': 5e4}
                    ):
        """
        Setter of the Gaussian Process kernel.
        Parameters:
        -----------
        kernel_type : str
            Type of kernel to use ('sqexp' or 'wend').
        kernel_params : list
            Parameters for the kernel, amplitude (given by m and c) and length scale (l).
        """
        print("===>  Setting GP Kernel " + kernel_type + "...")
        print("        + Kernel parameters: ", kernel_params)

        kernel_types_allowed = ['sqexp', 'wend']
        
        if kernel_type not in kernel_types_allowed:
            raise ValueError("Unknown kernel type. Use 'sqexp' or 'wend'.")
        else:
            self._kernel_info = {"type": kernel_type, "params": kernel_params}

        self._set_kernel = True

    def get_kernel(self, kernel_type):
        """
        Getter of the Gaussian Process kernel.
        Parameters:
        -----------
        kernel_info : dict
            Dictionary containing kernel type and parameters.
        Returns:
        --------
        Kernel : CovarianceMatrix object
            Correlation matrix operator.
        """
        types_allowed = ['sqexp', 'wend']

        if kernel_type == 'sqexp':
            return SquaredExponential
        elif kernel_type == 'wend':
            return Wendland

    def set_x0(self, guess):
        """
        Setter of the initial guess for the visibility fitting.
        Parameters:
        -----------
        guess : array
            Initial guess for the visibility model.
        """

        if guess is not None: # If user provides an initial guess.
            print("===>  Setting guess...")
            index = self.gridded_data_postprocess["index_weighted"]
            self._x0 = guess[index].flatten()
        else:
            self._x0 = guess

        self._set_x0 = True

    def set_fit_method(  self, method_name = 'bicgstab',
                         method_func = None,
                         maxiter = 50000, rtol = 1e-8):
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
        kernel_type = self._kernel_info["type"]
        kernel_params = self._kernel_info["params"]
        kernel_func = self.get_kernel(kernel_type)
        self._kernel = kernel_func(kernel_params, u, v)

        self._solver = IterativeSolverMethod(u, v, Vis, Weights,
                                             self._kernel,
                                             method_name = method_name,
                                             method_func = method_func, 
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
        print("===>  Setting gridded data...")

        self._gridded_data = {"u": u, "v": v, "vis": Vis, "weights": Weights}

        self._set_gridded_data = True

    def set_MAP_estimator(self, MAP_estimator):
        """
        Setter of the MAP estimator.
        Parameters:
        -----------
        MAP_estimator : MAPEstimator object
            MAPEstimator object containing the method to compute the MAP.
        """
        print("===>  Setting MAP estimator...")

        if isinstance(MAP_estimator, MAPEstimator) is False:
            raise ValueError("MAP_estimator must be an instance of MAPEstimator class.")
    
        self._MAP_estimator = MAP_estimator
        self._set_MAP_estimator = True

    def process_vis(self, data, hermitian = True):
        """
        Process the visibility data by gridding and dividing weighted data and non-weighted data.
        Parameters:
        -----------
        data : dict
            Dictionary containing 'u', 'v', 'vis' and 'weights' keys.
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
            grid = Gridding(self._Rmax, self._FT)
            try:
                u = data["u"]
                v = data["v"]
                Vis = data["vis"]
                Weights = data["weights"]
            except KeyError:
                raise ValueError("data dictionary must contain 'u', 'v', 'vis' and 'weights' keys.")
            u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights,
                                                                          hermitian = hermitian)
            self.set_gridded_data(u_gridded, v_gridded, vis_gridded, weights_gridded)
        
        self.process_gridded_vis()
        
    def process_gridded_vis(self):
        """
        Postprocess the gridded visibility data by separating weighted and non-weighted data.
        """
        self._PostProcess = PostProcess(self._gridded_data, self._Nx, self._Ny)
        self._gridded_data_postprocess = self._PostProcess.separate_data()
      
    def build_full_model(self):
        """
        Build the full visibility model from the weighted and non-weighted solution.
        Return
        -------
        V_full : 2D array, unit: Jy
            Full visibility model on a Nx x Ny grid.
        """
        print("Building full visibility model...")

        kernel = self.get_kernel(self._kernel_info["type"])
        kernel_params = self._kernel_info["params"]

        V_full = self._PostProcess.build_vis_model(kernel, kernel_params,
                                                   self._sol_visibility_weighted)

        return V_full

    def fit(self, 
            data = None,
            kernel_type = 'wend', kernel_params = {'m': -2, 'c': 1e8, 'l': 5e4},
            method_name = 'bicgstab', method_func = None,
            x0 = None, maxiter = 50000, rtol = 1e-8,
            hermitian = True, run_from_scratch = False):
        """
        Fit the visibility data using Gaussian Processes.
        Parameters:
        -----------
        data : dict
            Dictionary containing 'u', 'v', 'vis' and 'weights' keys.
            u : 1D array, unit: lambda
                u coordinates of the visibility data.
            v : 1D array, unit: lambda
                v coordinates of the visibility data.
            Vis : 1D array,  unit: Jy
                Visibility data.
            Weights : 1D array, unit: 1/Jy^2
                Weights of the visibility data.
        kernel_type : str
            Type of kernel to use ('sqexp' or 'wend').
        kernel_params : dict
            Parameters for the kernel, amplitude (given by m and c) and length scale (l).
        method_name : str
            Iterative solver method to use ('cg', 'bicgstab', etc.).
        method_func : function
            Custom iterative solver function. If None, default  is use the method name.
            This is the priority over method_name.
        x0 : array
            Initial guess for the iterative solver.
        maxiter : int
            Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved.
        rtol : float
            Relative tolerance for the iterative solver.
        hermitian : bool
            Whether to enforce Hermitian symmetry.
        run_from_scratch : bool
            Whether to run the fit from scratch (resetting all previous settings),
            i.e., running from after gridding.
        """
        if not self._set_gridded_data or data is not None:
            if not data:
                raise ValueError("If gridded data is not set, u, v, Vis and Weights must be provided.")
            try:
                u = data["u"]
                v = data["v"]
                Vis = data["vis"]
                Weights = data["weights"]
            except KeyError:
                raise ValueError("data dictionary must contain 'u', 'v', 'vis' and 'weights' keys.")
            self.process_vis(data, hermitian = hermitian)

        if run_from_scratch:
            self._set_x0 = False
            self._set_kernel = False
            self._set_fit_method = False

        if not self._set_x0:
            self.set_x0(x0)
            
        if not self._set_kernel:
            self.set_kernel(kernel_type = kernel_type,
                            kernel_params = kernel_params)

        if not self._set_fit_method:
            self.set_fit_method( method_name = method_name,
                                 method_func = method_func,
                                 maxiter = maxiter, rtol = rtol)
            
        self._solver.run()
        self._sol_visibility_weighted = self._solver.solution
        self._sol_visibility = self.build_full_model()
        self._sol_intensity = self.transform(self._sol_visibility)

    def search_MAP(self, data=None,
                   initial_guess={'m': -2, 'logl': 4},
                   N=50):
        if data is None:
            print("Using existing visibility data...")
            if not self._set_gridded_data:
                raise ValueError("Gridded data is not set, u, v, Vis and Weights must be provided.")
            data = self._gridded_data

        if not self._set_MAP_estimator:
            self._MAP_estimator = MAPEstimator(self._Rmax, N=N)
            self._set_MAP_estimator = True

        self._MAP_estimator.optimize(data, initial_guess)
        self._MAP = self._MAP_estimator.MAP

    def transform(self, vis, direction = "backward"):
        return self._FT.fast_transform(vis, direction = direction)

    def frank1d(self, geom,
                data = None, rout = None,
                alpha = 1.05, w_smooth = 1e-3, n_pts = 300 ):
        """
        Perform a 1D Frank fit on the visibility data.
        Parameters:
        -----------
        geom : Geometry object
            Geometry object containing source geometry parameters.
        data : dict
            Dictionary containing 'u', 'v', 'vis' and 'weights' keys.
            u : 1D array, unit: lambda
                u coordinates of the visibility data.
            v : 1D array, unit: lambda
                v coordinates of the visibility data.
            Vis : 1D array,  unit: Jy
                Visibility data.
            Weights : 1D array, unit: 1/Jy^2
                Weights of the visibility data.
        rout : float
            Outer radius for the 1D Frank fit in arcseconds.
        alpha : float
            Power-law index for the visibility fitting.
        w_smooth : float
            Smoothing weight for the visibility fitting.
        n_pts : int
            Number of points for the 1D Frank fit.

        Returns:
        --------
        vis_fit_1d : 1D array, unit: Jy
            Fitted visibility data using 1D Frank.
        """
        print('Performing 1D Frank fit...' + '\n')
        print(r'+ $\alpha$ = ', str(alpha), r' and $w_{smooth}$ = ', str(w_smooth) + '\n')
        print( '+ N = ', str(n_pts))
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        if rout is None:
            print("Using Rmax from Frank2D object...")
            rout = self._Rmax*rad_to_arcsec
        geom_f1d = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)
        FF = FrankFitter(rout, n_pts, geom_f1d, alpha = alpha, weights_smooth = w_smooth)

        if  data is None:
            if not self._set_gridded_data:
                raise ValueError("Gridded data is not set, u, v, Vis and Weights must be provided.")
            print("Using existing visibility data...")
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
    def Nx(self):
        """ Number of collocation points in x dimension."""
        return self._Nx
    
    @property
    def Ny(self):
        """ Number of collocation points in y dimension."""
        return self._Ny
    
    @property
    def dx(self):
        """ Cellsize in x dimension in arcseconds."""
        return (2*self.Rmax)/self.Nx
    
    @property
    def dy(self):
        """ Cellsize in y dimension in arcseconds."""
        return (2*self.Rmax)/self.Ny
    
    @property
    def gridded_data(self):
        """ Gridded visibility data in a dict. """
        return self._gridded_data

    @property
    def u(self):
        """ 
        u - 1d collocation points in lambda (0 centered).
        (N, 1) array.
        """
        return self._FT.u
    
    @property
    def u_grid(self):
        """ u collocation points in lambda (0 centered) on a grid."""
        return self._FT._Un.reshape((self._Nx, self._Ny), order = 'C')
    
    @property
    def v(self):
        """ 
        v 1d - collocation points in lambda (0 centered).
        (N, 1) array.
        """
        return self._FT.v
    
    @property
    def v_grid(self):
        """ v collocation points in lambda (0 centered) on a grid."""
        return self._FT._Vn.reshape((self._Nx, self._Ny), order = 'C')
    
    @property
    def x(self):
        """ x collocation points in rad."""
        return self._FT._x
    
    @property
    def x_grid(self):
        """ x collocation points in rad on a grid."""
        return self._FT._Xn.reshape((self._Nx, self._Ny), order = 'C')
    
    @property
    def y(self):
        """ y collocation points in rad."""
        return self._FT._y
    
    @property
    def y_grid(self):
        """ y collocation points in rad on a grid."""
        return self._FT._Yn.reshape((self._Nx, self._Ny), order = 'C')
    
    @property
    def visibility_model(self):
        """ 2D visibility model in Jy. """
        return self._sol_visibility.reshape((self._Nx, self._Ny), order='C')

    @property
    def Rmax(self):
        """ Maximum value of the x coordinate in arcseconds."""
        return self._Rmax*rad_to_arcsec

    @property
    def cellsize(self):
        """ Cellsize in arcseconds."""
        return (2*self._Rmax*rad_to_arcsec)/self._N
    
    @property
    def intensity_model(self):
        """ 2D intensity model in Jy/sr """
        return self._sol_intensity.reshape((self._Nx, self._Ny), order='C').real
    
    @property
    def FT(self):
        """ FourierTransform2D object."""
        return self._FT

    @property
    def MAPEstimator(self):
        """ MAPEstimator object."""
        if self._MAP_estimator is None:
            raise ValueError("No MAP estimator found. Run MAP_search first.")
        return self._MAP_estimator
    
    @property
    def MAP(self):
        """ Maximum A Posteriori params"""
        if self._MAP is None:
            raise ValueError("No MAP found. Run MAP_search first.")
        return self._MAP
    
    @property
    def sol_f1d(self):
        """ 1D Frank solution."""
        if self._sol_f1d is None:
            raise ValueError("No 1D Frank solution found. Run frank1d first.")
        return self._sol_f1d

    @property
    def solver(self):
        """ IterativeSolverMethod object."""
        if self._solver is None:
            raise ValueError("No solver found. Set it first.")
        return self._solver