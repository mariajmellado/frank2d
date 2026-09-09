from ..constants import rad_to_arcsec, deg_to_rad
from ..geometry import Geometry
from ..logger import Logger

from .fourier2d_gpu import FourierTransform2D
from .process_vis_gpu import Gridding, PostProcess
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
    def __init__(self, N, Rmax, verbose = False, dev_mode = False):
        """
        Initialize the Frank2D class.
        Parameters:
        -----------
        N : int
            Number of collocation points in each dimension (image will be NxN).
        Rmax : float
            Radius of the image in arcseconds.
        verbose : bool
            Whether to print verbose messages during the fitting process.
        dev_mode : bool
            Whether to run in development mode (for debugging purposes).
        """
        self._N =  N
        self._Nx = N
        self._Ny = N
        self._N2 = self._Nx * self._Ny
        self._Rmax = Rmax / rad_to_arcsec
        self._FT = FourierTransform2D(self._Rmax, self._N)

        self._set_x0 = False
        self._set_kernel = False
        self._set_gridded_data = False
        self._set_fit_method = False
        self._set_MAP = False
        self._set_MAPEstimator = False

        self._sol_visibility = None
        self._sol_intensity = None

        self._verbose = verbose
        self.show = Logger(verbose = self._verbose)
        self._dev_mode = dev_mode

        self.validate_grid_parameters(self._Rmax, self._N)
    
    def validate_grid_parameters(self, Rmax, N):
        """
        Validate the Rmax and N parameters for the gridding process based on the Nyquist theorem and safety margins.
        Warning messages mainly are printed if the parameters do not meet the requirements.
        Parameters:
        -----------
        Rmax : float
            Radius of the image in arcseconds.
        N : int
            Number of collocation points in each dimension (image will be NxN).
        """
        if Rmax <= 0:
            self.show.error("Rmax must be positive.")
        if N <= 0:
            self.show.error("N must be positive.")
    
        Qmax = self.FT.Qmax

        # Strict Nyquist limit (boundary factor of 4)
        N_nyquist_float = 4 * Qmax * (Rmax / rad_to_arcsec)
        N_nyquist_min = int(cp.ceil(N_nyquist_float))
        
        # Safe recommendation (boundary factor of 5 and strictly even)
        N_recommended_float = 5 * Qmax * (Rmax / rad_to_arcsec)
        N_recommended = int(cp.ceil(N_recommended_float))

        if N_recommended % 2 != 0:
            N_recommended += 1

        is_valid = True

        # Nyquist theorem validation.
        if N < N_nyquist_min:
            msg = (f"Grid size N={N} is strictly below the Nyquist limit (minimum N={N_nyquist_min}) "
                f"for R_max={R_max*rad_to_arcsec}\" and Q_max={Q_max:.2e} lambda. "
                f"High-frequency visibilities will be clipped and lost. "
                f"The recommended N is {N_recommended} or higher.")
            self.show.error(msg)
            
        # Safety margin validation.
        if N < N_recommended:
            msg = (f"Grid size N={N} satisfies the strict Nyquist limit but lacks a safety margin. "
                f"To avoid potential ringing or truncation artifacts at the grid boundaries, "
                f"a factor of 5 is recommended (N >= {N_recommended}).")
            self.show.warning(msg)

    def check_bounds(self, u, v):
            """
            Function to check if the frequencies are within the bounds of the
            grid (the uv domain is properly covered).
            Parameters
            ----------
            u : 1D array, unit = lambda
                u coordinates of the visibilities from the uvtable.
            v : 1D array, unit = lambda
                v coordinates of the visibilities from the uvtable.
            Returns
            -------
            bool
                True if all frequencies are within bounds, False otherwise.
            """
            Qmax_grid = self._FT.Qmax
            Qmin_grid = self._FT.Qmin2

            Qmax_data = cp.sqrt(u**2 + v**2).max()
            Qmin_data = cp.sqrt(u**2 + v**2).min()

            # Check whether the first (last) collocation point is smaller (larger)
            # than the shortest (longest) deprojected baseline in the dataset
            if Qmin_grid < Qmin_data:
                self.show.warning(r"First collocation point, q_0 = {:.3e} \lambda,"
                                " is at a baseline shorter than the"
                                " shortest baseline in the dataset,"
                                r" min(q_data) = {:.3e} \lambda. For q_0 << min(q_data),"
                                " the fit's total flux may be biased"
                                " low.".format(Qmin_grid, Qmin_data))

            if Qmax_grid < Qmax_data:
                # This validation is similar to the "Safety margin validation" in the validate_grid_parameters function,
                # but here we allow the user to proceed with a warning instead of an error, since in some cases
                # (e.g., testing different configurations) it may be desirable to fit to a shorter maximum baseline than
                # the longest one in the dataset.
                self.show.error(r" Last collocation point, q_n = {:.3e} \lambda, is at"
                                " a shorter baseline than the longest"
                                r" baseline in the dataset, max(q_data) = {:.3e} \lambda."
                                "  Please increase N."
                                " Or if you'd like to fit to shorter maximum baseline,"
                                " cut the (u, v) distribution before fitting"
                                " ".format(Qmax_grid, Qmax_data))

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
        self.show.info("===>  Setting GP Kernel " + kernel_type + "...")
        self.show.info("        + Kernel parameters: " + str(kernel_params))

        kernel_types_allowed = ['sqexp', 'wend']
        
        if kernel_type not in kernel_types_allowed:
            self.show.error("Unknown kernel type. Use 'sqexp' or 'wend'.")
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
            self.show.info("===>  Setting guess...")
            index = self._gridded_data_postprocess["index_weighted"]
            guess_cp = cp.asarray(guess)
            self._x0 = guess_cp[index].flatten()
        else:
            self._x0 = None

        self._set_x0 = True

    def set_fit_method(  self, method_name = 'bicgstab',
                         method_func = None,
                         maxiter = 50000, rtol = 1e-8,
                         precond_type = 'jacobi',
                         verbose = False
                        ):
        """
        Setter of the fitting method for the iterative solver.
        Parameters:
        -----------
        method : str
            Iterative solver method to use ('cg', 'bicgstab', etc.).
        method_func : function
            Custom iterative solver function. If None, default  is use the method name.
        maxiter : int
            Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved.
        rtol : float
            Relative tolerance for the iterative solver.
        precond_type : str
            Type of preconditioner to use. 
            Available 'jacobi' and 'ilu'.
            - jacobi: Jacobi preconditioner, which uses the inverse of the diagonal of the kernel matrix as a preconditioner.
            - ilu: Incomplete LU preconditioner, which approximates the LU decomposition of the kernel matrix and uses it as a preconditioner.
        """
        if not self._set_kernel:
            self.show.error("Set kernel before setting fit method.")

        data_weighted = self._gridded_data_postprocess["weighted"]
        u = data_weighted["u"]
        v = data_weighted["v"]
        Vis = data_weighted["vis"]
        Weights = data_weighted["weights"]

        # Create covariance matrix operator.
        kernel_type = self._kernel_info["type"]
        kernel_params = self._kernel_info["params"]
        kernel_func = self.get_kernel(kernel_type)
        kernel = kernel_func(kernel_params, u, v)

        self._solver = IterativeSolverMethod(u, v, Vis, Weights,
                                             kernel,
                                             method_name = method_name,
                                             method_func = method_func, 
                                             x0 = self._x0,
                                             maxiter = maxiter,
                                             rtol = rtol,
                                             precond_type = precond_type,
                                             verbose = verbose)
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
        self.show.info("===>  Setting gridded data...")

        self._gridded_data = {"u": u, "v": v, "vis": Vis, "weights": Weights}

        self._set_gridded_data = True

    def set_MAP(self, MAP):
        """
        Setter of the MAP parameters.
        Parameters:
        -----------
        MAP : dict
            Dictionary containing the MAP parameters.
        """
        self.show.info("===>  Setting MAP parameters...")

        self._MAP = MAP
        self._set_MAP = True
    
    def set_MAPEstimator(self, N_opt = 50, verbose = False):
        """
        Setter of the MAP estimator.
        Rmax should be the same for the whole algorithm.
        Parameters:
        -----------
        N_opt : int
            Number of collocation points for the optimization.
        """
        self.show.info("===>  Setting MAP estimator...")
        self._MAP_estimator = MAPEstimator(self._Rmax, N_opt = N_opt, verbose = verbose)
        self._set_MAPEstimator = True

    def process_vis(self, data, hermitian = True, verbose = False):
        """
        Process the visibility data by gridding and dividing weighted data and non-weighted data.
        Parameters:
        -----------
        data : dict
            Dictionary containing 'u', 'v', 'vis' and 'weights' keys.
            u : 1D np.array or cp.array, size relative to UV table, unit: lambda
                u coordinates of the visibility data.
            v : 1D np.array or cp.array, size relative to UV table, unit: lambda
                v coordinates of the visibility data.
            Vis : 1D np.array or cp.array, size relative to UV table, unit: Jy
                Visibility data.
            Weights : 1D np.array or cp.array, size relative to UV table, unit: 1/Jy^2
                Weights of the visibility data.
        hermitian : bool
            Whether to enforce Hermitian symmetry.
        Returns:
        None
        """
        if not self._set_gridded_data:
            if data is None:
                self.show.error("Gridded data is not set, u, v, Vis and Weights must be provided.")
            else:
                self.show.info("===>  Gridding visibility data...")
                try:
                    u = cp.asarray(data["u"])
                    v = cp.asarray(data["v"])
                    Vis = cp.asarray(data["vis"])
                    Weights = cp.asarray(data["weights"])
                except KeyError:
                    self.show.error("data dictionary must contain 'u', 'v', 'vis' and 'weights' keys.")
                # corroborate that they are cupy arrays.
    
                if not self._dev_mode:
                    self.check_bounds(u, v)

                grid = Gridding(self._Rmax, self._FT, verbose = verbose)
                u_gridded, v_gridded, vis_gridded, weights_gridded = grid.run(u, v, Vis, Weights,
                                                                            hermitian = hermitian)
                # grid.run may return numpy arrays; convert to cupy
                self.set_gridded_data(u_gridded, v_gridded, vis_gridded, weights_gridded)
        else:
            self.show.info("Using existing gridded data...")
        
        self.process_gridded_vis()

    def process_gridded_vis(self, verbose = False):
        """
        Postprocess the gridded visibility data by separating weighted and non-weighted data.
        """
        self._PostProcess = PostProcess(self._gridded_data, self._Nx, self._Ny, verbose = verbose)
        self._gridded_data_postprocess = self._PostProcess.separate_data()
      
    def build_full_model(self):
        """
        Build the full visibility model from the weighted and non-weighted solution.
        Return
        -------
        V_full : 2D array, unit: Jy
            Full visibility model on a Nx x Ny grid.
        """
        kernel = self.get_kernel(self._kernel_info["type"])
        kernel_params = self._kernel_info["params"]

        V_full = self._PostProcess.build_vis_model(kernel, kernel_params,
                                                   self._sol_visibility_weighted)

        return V_full

    def search_MAP(self, data = None,
                   initial_guess = {'m': -2, 'l':1e4, 'logq': 5, 'logp': -1},
                   N_opt = 50, verbose = False):
        """
        Search for the Maximum A Posteriori (MAP) parameters.
        Parameters:
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
        initial_guess : dict
            Initial guess for the MAP parameters.
            m : float
                Power-law index for the power spectrum of the visibilities.
            l : float
                Length scale for the kernel in lambda.
            logq : float, lambda
                Logarithm of certain spatial baseline in lambda.
            logp : float, lambda
                Logarithm of the power spectrum value associated to logq.
                From this values we obtain c solving logp = m*logq + c,
                where (m, c, l) are the parameters of the kernel for the GP.
        N_opt : int
            Number of collocation points for the optimization.
        verbose : bool
            Whether to print optimization progress.
        """
        self._N_opt = N_opt
            
        if not self._set_MAP:
            if data is None:
                self.show.info("Using existing visibility data...")
                if not self._set_gridded_data:
                    self.show.error("Gridded data is not set, u, v, Vis and Weights must be provided.")
                data = self._gridded_data

            if not self._set_MAPEstimator:
                self.set_MAPEstimator(N_opt = self._N_opt, verbose = verbose)

            self.show.info("===>  Searching for MAP...")
            self._MAP_estimator.optimize(data, initial_guess)
            self._MAP = self._MAP_estimator.MAP
            self.show.info("         + MAP found: " + str(self._MAP))


            self._set_MAP = True
        else:
            self.show.info("Using existing MAP parameters...")

    def fit(self, 
            data = None,
            find_MAP = False, initial_guess = {'m': -2, 'l': 1e4, 'logq': 5, 'logp': -2 },
            kernel_type = 'wend', kernel_params = {'m': -2, 'c': 1e8, 'l': 5e4},
            method_name = 'bicgstab', method_func = None,
            x0 = None, maxiter = 50000, rtol = 1e-8, precond_type = 'jacobi',
            hermitian = True, run_from_scratch = False
            ):
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
        find_MAP : bool
            Whether to search for the MAP parameters before fitting.
        initial_guess : dict
            Initial guess for the MAP parameters.
            m : float
                Power-law index for the power spectrum of the visibilities.
            l : float
                Length scale for the kernel in lambda.
            logq : float, lambda
                Logarithm of certain spatial baseline in lambda.
            logp : float, lambda
                Logarithm of the power spectrum value associated to logq.
                From this values we obtain c solving logp = m*logq + c,
                where (m, c, l) are the parameters of the kernel for the GP.
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
        
        self.process_vis(data, hermitian = hermitian, verbose = self._verbose)

        if run_from_scratch:
            self._set_x0 = False
            self._set_kernel = False
            self._set_fit_method = False

        if find_MAP:
            self.search_MAP(data = self._gridded_data,
                            initial_guess = initial_guess,
                            verbose = self._verbose)
            kernel_params = self._MAP

        if not self._set_x0:
            self.set_x0(x0)
            
        if not self._set_kernel:
            self.set_kernel(kernel_type = kernel_type,
                            kernel_params = kernel_params)

        if not self._set_fit_method:
            self.set_fit_method( method_name = method_name,
                                 method_func = method_func,
                                 maxiter = maxiter, rtol = rtol,
                                 precond_type = precond_type,
                                 verbose = self._verbose)
            
        self._solver.run()
        self._sol_visibility_weighted = self._solver.solution
        self._sol_visibility = self.build_full_model()
        self._sol_intensity = self.transform(self._sol_visibility)

    def transform(self, vis, direction = "backward"):
        return self._FT.fast_transform(vis, direction = direction)

    def frank1d(self, geom,
                data = None, rout = None,
                alpha = 1.05, w_smooth = 1e-3, n_pts = 300 ):
        """
        Perform a 1D Frank fit on the visibility data.
        This run in CPU by default.
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
        self.show.info('===> Performing 1D Frank fit...' + '\n')
        self.show.info(r'+ $\alpha$ = '+ str(alpha) + r' and $w_{smooth}$ = '+ str(w_smooth) + '\n')
        self.show.info( '+ N = '+ str(n_pts))
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        if rout is None:
            self.show.warning("Outer radius for Frank1D fit not provided. Using Rmax = " + str(self.Rmax) + " arcseconds.")
            rout = self._Rmax*rad_to_arcsec
        geom_f1d = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)
        FF = FrankFitter(rout, n_pts, geom_f1d, alpha = alpha, weights_smooth = w_smooth)

        if  data is None:
            if not self._set_gridded_data:
                self.show.error("Gridded data is not set, u, v, Vis and Weights must be provided.")
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
    def N(self):
        """ Number of collocation points."""
        return self._N

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
    def Qmax(self):
        """ Maximum value of the u coordinate in lambda."""
        return self.FT.Qmax

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
            self.show.error("No MAP estimator found. Run MAP_search first.")
        return self._MAP_estimator
    
    @property
    def MAP(self):
        """ Maximum A Posteriori params"""
        if self._MAP is None:
            self.show.error("No MAP found. Run MAP_search first.")
        return self._MAP
    
    @property
    def sol_f1d(self):
        """ 1D Frank solution."""
        if self._sol_f1d is None:
            self.show.error("No 1D Frank solution found. Run frank1d first.")
        return self._sol_f1d

    @property
    def solver(self):
        """ IterativeSolverMethod object."""
        if self._solver is None:
            self.show.error("No solver found. Set it first.")
        return self._solver

    @property
    def uv_points(self):
        """Collocation points in the frequency plane in lambda."""
        return self.FT.uv_points

    @property
    def xy_points(self):
        """Collocation points in the image plane in arcseconds."""
        return self.FT.xy_points * rad_to_arcsec
    