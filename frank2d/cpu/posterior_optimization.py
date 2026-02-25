import numpy as np
import time
import logging
import scipy
from scipy.special import gamma
from .constants import rad_to_arcsec, deg_to_rad
from .fourier2d import FourierTransform2D
from .minimizer import Powell
from .gaussian_process import Wendland

import abc
import matplotlib.pyplot as plt

"""
This module is based in classes and functions contained in Frankenstein-1D algorithm for fitting visibilities.
"""
class MAPEstimator(object):
    def __init__(self, Rmax, N = 50, minimizer = Powell):
        """
        Maximum A Posteriori Estimator for the parameters of the Gaussian Process
        Params
        ------
        Rmax : float
            Maximum radius to model, in arcseconds.
        N : int, optional
            Number of radial points to use in the model. Default is 50.
            If use N > 50, the optimization can be very slow.
        """
        self._Rmax = Rmax
        self._N = N
        self._FF = FourierBesselFitter(self._Rmax, N)

        self._set_minimizer = False
        self._minimizer = minimizer

        # Optimization results.
        self._times = []
        self._ms = []
        self._cs = []
        self._ls = []
        self._jDjs = []
        self._logdetDs = []
        self._logdetSs = []
        self._minus_log_posteriors = []

        self._best = {}
    
    def set_minimizer(self, minimizer, initial_guess = {"m": -2, "logl": 4, "p": -2}):
        """
        Set the minimizer to use for the posterior optimization.
        """
        initial_guess = self.process_initial_guess(initial_guess)

        self._minimizer = minimizer(self.eval_prob, initial_guess)
        self._set_minimizer = True

    def process_initial_guess(self, initial_guess):
        """
        Process the initial guess for the parameters.
        Is assumed that  logc = p - 5*m. p is always initialized to -2.
        Params
        ------
        initial_guess : dict, optional
            Initial guess for the parameters m and logl. If None, use m = -2 and logl = 4.
        """
        params = {}
        if 'p' not in initial_guess:
            raise ValueError("Initial guess must contain 'p' parameter.")
        if 'm' not in initial_guess:
            raise ValueError("Initial guess must contain 'm' parameter.")
        if 'logl' not in initial_guess:
            raise ValueError("Initial guess must contain 'logl' parameter.")

        params['p'] = float(initial_guess['p'])  # Fixed relation: logc = p - 5*m
        params['m'] = float(initial_guess['m'])
        params['l'] = float(10**initial_guess['logl'])

        # the order is important.
        self._params_order = ['p', 'm', 'l']
        list_params = [params[key] for key in self._params_order]
        return list_params
    
    def create_gaussian_model(self, data):
        """
        Create the GaussianModel object using the Frank2D fitting scheme.
        Params
        ------
        data : dict
            Dictionary containing the data to fit. Must contain the keys 'u', 'v', 'vis', and 'weights'.
        """
        print("Fitting the visibilities with Frank2D with 2DFT...")
        start_time = time.time()

        u, v, vis, weights = data['u'], data['v'], data['vis'], data['weights']
        self._FF.fit(u, v, vis, weights)

        end_time = time.time()
        print(f" + Time to preprocess the visibilities: {end_time - start_time:.2f} seconds")

        self._GM = self._FF.GaussianModel
        self._minus_log_posterior = self._GM.minus_log_posterior

    def process_x_minimizer(self, x):
        """
        Process the parameters from the minimizer to the parameters used in the GaussianModel.
        Params
        ------
        x : array-like
            Parameters to process. By default, x contains p, m, l.
        """
        # Follow the order defined in self._params_order and asign the values of x.
        params = { key: x[i] for i, key in enumerate(self._params_order) }
        m = params['m']
        l = params['l']
        p  = params['p']
        logc = p - 5*m
        c = 10**logc

        return {'m': m, 'c': c, 'l': l}

    def optimize(self, data, initial_guess = {"p": -2, "m": -2, "logl": 4}):
        """
        Optimize the posterior to find the MAP estimate of the parameters.
        Params
        ------
        data : dict
            Dictionary containing the data to fit. Must contain the keys 'u', 'v', 'vis', and 'weights'.
        initial_guess : dict, optional
            Initial guess for the parameters m and logl. If None, use m = -2 and logl = 4.
            The parameter c is always initialized to 10**(-2).

        """
        # Fit with Frankenstein1D scheme.
        self.create_gaussian_model(data)

        p0 = self._get_p0()
        self._p0 = p0

        print("Setting the initial guess...")
        initial_guess = self.process_initial_guess(initial_guess)

        if not self._set_minimizer:
            minimizer = Powell(self.eval_prob, initial_guess)
            self.set_minimizer(Powell)
        
        print("Optimizing the posterior...")
        start_time = time.time()

        self._minimizer.run()

        end_time = time.time()
        print(f" + Time to optimize the posterior: {end_time - start_time:.2f} seconds")

        x = self._minimizer.solution

        # Normalize the parameters.
        self._best = self.process_x_minimizer(x)
      
    def _get_p0(self):
        """
        Get the minus log posterior for the initial guess.
        """
        GM = self._GM

        params = {'m': 0, 'c': 10**(-2), 'l':10**4}
        p0 = GM.minus_log_posterior(params)
        jDj0, logdetS0, logdetD0 = GM.jDj, GM.logdetS, GM.logdetD

        return p0, jDj0, logdetS0, logdetD0
    
    def eval_prob(self, x):
        """
        Evaluate the negative log posterior for given parameters.
        Params
        ------
        x : array-like
            Parameters to evaluate. Follows the order defined in self._params_order.
        """
        GM = self._GM

        p0, jDj0, logdetS0, logdetD0 = self._p0

        params = self.process_x_minimizer(x)

        current = GM.minus_log_posterior(params) - p0
        jDj, S, D = GM.jDj - jDj0, GM.logdetS - logdetS0, GM.logdetD - logdetD0

        result = {
            'm': params['m'],
            'c': params['c'],
            'l': params['l'],
            'minus_log_posterior': current,
            'jDj': jDj,
            'logdetS': S,
            'logdetD': D,
        }
        self.save_results(result)

        return current

    def save_results(self, results):
        """
        Save the results of the posterior optimization.
        """
        self._ms.append(results['m'])
        self._cs.append(results['c'])
        self._ls.append(results['l'])
        self._jDjs.append(results['jDj'])
        self._logdetDs.append(results['logdetD'])
        self._logdetSs.append(results['logdetS'])
        self._minus_log_posteriors.append(results['minus_log_posterior'])

    @property
    def MAP(self):
        """Return the best parameters found in the posterior optimization."""
        return self._best

    @property
    def power_spectrum(self):
        """Return the power spectrum function."""
        return self._GM.power_spectrum
    
    @property
    def GaussianModel(self):
        """Return the GaussianModel object which refers to the Gaussian Process"""
        return self._GM

    @property
    def results_opt(self):
        """
        Save the results of the posterior optimization.
        """
        return {
            'm': self._ms,
            'c': self._cs,
            'l': self._ls,
            'jDj': self._jDjs,
            'logdetD': self._logdetDs,
            'logdetS': self._logdetSs,
            'minus_log_posterior': self._minus_log_posteriors,
        }

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._FF.Qmax


class FourierBesselFitter(object):
    """
    Fourier-Bessel series model for fitting visibilities
    """

    def __init__(self, Rmax, N, block_data=True,
                 block_size=10 ** 5, verbose=True):
        # Assuming optically thick disc by default.
        # Thus, scale height = 0.

        self._2DFT = FourierTransform2D(Rmax, N)
        self._Rmax = Rmax*rad_to_arcsec

        self._vis_map = VisibilityMapping(self._2DFT,
                                          block_data=block_data,
                                          block_size=block_size,
                                          check_qbounds=False,
                                          verbose=verbose)

    def preprocess_visibilities(self, u, v, V, weights):
        r"""Prepare the visibilities for fitting. 
        """
        return self._vis_map.map_visibilities(u, v, V, weights)

    def _build_matrices(self, mapping):
        r"""
        Compute the matrices M and j from the visibility data.
        """  
        self._M = mapping['M']
        self._j = mapping['j']
        self._V = mapping['V']

    def fit(self, u, v, V, weights):
        r"""
        Fit the visibilties
        """
        mapping = self.preprocess_visibilities(u, v, V, weights)
        self._build_matrices(mapping)

        return self._fit()

    def _fit(self):
        """Fit step. Computes the best fit given the pre-processed data"""
        self.GaussianModel = GaussianModel(self._2DFT, self._M, self._j, self._Rmax)
    
    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._2DFT.Qmax

class VisibilityMapping:
    r"""Builds the mapping between the visibility and image planes.

    VisibilityMapping generates the transform matrices :math:`H(q)` such that
    :math:`V_\nu(q) = H(q) I_\nu`. It also uses these to construct the design
    matrices :math:`M` and :math:`j` used in the fitting. 
    """
    def __init__(self, DFT, block_data=True,
                 block_size=10 ** 5, check_qbounds=True, verbose=True):
        # Store flags
        self.check_qbounds = check_qbounds
        self._chunking = block_data
        self._chunk_size = block_size
        self._2DFT = DFT

        self._verbose = verbose
   
    def map_visibilities(self, u, v, V, weights):
        r"""
        Compute the matrices :math:`M` and :math:`j` from the visibility data.
        Optimized for single channel/element usage.
        """

        if self._verbose:
            logging.info('    Building visibility matrices M and j')
        
        q = np.hypot(u, v)

        # Check consistency of the uv points with the model
        self._check_uv_range(q)

        # Ensure weights are real
        w = (np.ones_like(V) * weights).real

        # Initialization: Direct 2D/1D allocation (No 'channel' dimension)
        M = np.zeros((self.size, self.size), dtype='f8')
        j = np.zeros(self.size, dtype='f8')
        
        Ndata = len(V)

        # If chunking is used, we will build up M and j chunk-by-chunk
        if self._chunking:
            Nstep = int(self._chunk_size / self.size + 1)
        else:
            Nstep = Ndata

        start = 0
        end = min(Nstep, Ndata)

        while start < Ndata:
            # Slicing directly from main arrays (no need for intermediate filtered arrays)
            us_chunk = u[start:end]
            vs_chunk = v[start:end]
            ws_chunk = w[start:end]
            Vs_chunk = V[start:end]
            
            # Generate k only for the chunk (saves memory vs generating ones for full size)
            ks_chunk = np.ones(len(us_chunk))

            X = self._get_mapping_coefficients(ks_chunk, us_chunk, vs_chunk)

            wXT = np.transpose(np.conjugate(X)) * ws_chunk
            
            # Calculate and accumulate directly into M and j
            # No need for intermediate 'val' variable if memory is tight, 
            # but keeping it explicit for readability.
            val = np.matmul(wXT, X, dtype="complex128")

            M += val.real
            j += np.matmul(wXT, Vs_chunk, dtype="complex128").real

            start = end
            end = min(Ndata, end + Nstep)

        return {
            'M': M,
            'j': j,
            'V': V,
            'W': w,
        }

    def predict_visibilities(self, I, u, v, k=None):
        r"""Compute the predicted visibilities given the brightness profile, I"""
        if self._chunking:
            Ni = int(self._chunk_size / self.size + 1)
        else:
            Ni = len(u)

        end = 0
        start = 0
        V = []
        while end < len(u):
            start = end
            end = start + Ni
            ui = u[start:end]
            vi = v[start:end]
            
            ki = None
            if k is not None:
                ki = k[start:end]

            H = self._get_mapping_coefficients(ki, ui, vi)

            V.append(np.dot(H, I))
        return np.concatenate(V)

    def _get_mapping_coefficients(self, ks, u, v, inverse=False):
        """Get :math:`H(q)`, such that :math:`V(q) = H(q) I_\nu`"""
        scale = 1
        if inverse:
            scale = np.atleast_1d(1/scale).reshape(1,-1)
            direction='backward'
        else:
            direction='forward'

        H = self._2DFT.coefficients(u, v, direction=direction)*scale

        return H

    def _check_uv_range(self, uv):
        """Check that the uv domain is properly covered"""

        # Check whether the first (last) collocation point is smaller (larger)
        # than the shortest (longest) deprojected baseline in the dataset
        if self.check_qbounds:
            if self.q[0] < uv.min():
                logging.warning(r"WARNING: First collocation point, q[0] = {:.3e} \lambda,"
                                " is at a baseline shorter than the"
                                " shortest deprojected baseline in the dataset,"
                                r" min(uv) = {:.3e} \lambda. For q[0] << min(uv),"
                                " the fit's total flux may be biased"
                                " low.".format(self.q[0], uv.min()))

            if self.q[-1] < uv.max():
                raise ValueError(r"ERROR: Last collocation point, {:.3e} \lambda, is at"
                                 " a shorter baseline than the longest deprojected"
                                 r" baseline in the dataset, {:.3e} \lambda. Please"
                                 " increase N in FrankMultFrequencyFitter (this is"
                                 " `hyperparameters: n` if you're using a parameter"
                                 " file). Or if you'd like to fit to shorter maximum baseline,"
                                 " cut the (u, v) distribution before fitting"
                                 " (`modify_data: baseline_range` in the"
                                 " parameter file).".format(self.q[-1], uv.max()))

    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._2DFT.r * rad_to_arcsec

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._2DFT.Rmax * rad_to_arcsec

    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._2DFT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._2DFT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._2DFT.size

class GaussianModel:
    r"""
    Solves the linear regression problem to compute the posterior,

    .. math::
       P(I|q,V,p) \propto G(I-\mu, D),

    where :math:`I` is the intensity to be predicted, :math:`q` are the
    baselines and :math:`V` the visibility data. :math:`\mu` and :math:`D` are
    the mean and covariance of the posterior distribution.

    If :math:`p` is provided, the covariance matrix of the prior is included,
    with

    .. math::
        P(I|p) \propto G(I, S(p)),

    and the Bayesian Linear Regression problem is solved. :math:`S` is computed
    from the power spectrum, :math:`p`, if provided. Otherwise the traditional
    (frequentist) linear regression is used.

    The problem is framed in terms of the design matrix :math:`M` and
    information source :math:`j`.

    :math:`H(q)` is the matrix that projects the intensity :math:`I` to
    visibility space. :math:`M` is defined by

    .. math::
        M = H(q)^T w H(q),

    where :math:`w` is the weights matrix and

    .. math::
        j = H(q)^T w V.

    The mean and covariance of the posterior are then given by

    .. math::
        \mu = D j

    and

    .. math::
        D = [ M + S(p)^{-1}]^{-1},

    if the prior is provided, otherwise

    .. math::
        D = M^{-1}.
    """

    def __init__(self, DFT, M, j, Rmax):
        self._2DFT = DFT
        self._N = int(np.sqrt(M.shape[0]))
        self._Rmax = Rmax

        self._M = M
        self._j = j

        self.Ykm = self._2DFT.coefficients(direction="backward")
        self.Ykm_conj = self.Ykm.conj()

        # Spatial frequencies and related.
        self.u, self.v = self._2DFT.uv_points
        u1, u2 = np.meshgrid(self.u, self.u)
        v1, v2 = np.meshgrid(self.v, self.v)
        self.data = [u1, u2, v1, v2]
        self._q1 = np.hypot(u1, v1)
        self._q2 = np.hypot(u2, v2)
        self._qs = np.hypot(self.u, self.v)
        self._min_freq = np.sort(np.abs(np.unique(self.v)))[1]
        self._max_freq = np.max(self._qs)

        # Parameters to optimize.
        self.m = -2
        self.logc = 8
        self.logl = np.log10(7e4)

        # Wendland kernel parameters.
        self._r = np.sqrt((u1-u2)**2 + (v1-v2)**2)
        self._j_W, self._k_W = 4, 1

        params = {'m' : self.m, 'c': 10**self.logc, 'l': 10**self.logl}
        self._Wendland = Wendland(params, u1, v1, u2, v2)

        self._verbose = False

    def fit(self, params):
        """
        Fit the model with the given parameters m, logc, logl.
        Params
        ------
        params : dict
            Dictionary containing the parameters m, log_c, log_l.
        """
        if params is not None:
            self.m = params['m']
            self.c = 10**params['logc']
            self.l = 10**params['logl']
        
        S_real = self.calculate_S_real_space(self.m, self.c, self.l)
        self._Sinv = np.linalg.inv(S_real)
        self._Dinv = self._M + self._Sinv

        self._solve_mu()

    def _solve_mu(self, cg = False):
        """Compute the mean and variance"""
        self._mu = self.calculate_mu_cholesky(self._Dinv)

    def logdet(self, M):
        """
        Calculate the log determinant of matrix m using LDL decomposition.
        """
        l, d, p = scipy.linalg.ldl(M)
        d = np.diag(d)
        return np.sum(np.log(np.abs(d)))

    def preprocess_params(self, params):
        """
        Preprocess the parameters m, logc, logl.
        Params
        ------
        param : dict
            Dictionary containing the parameters m, log_c (or c), log_l (or l).
        """
        if params is None:
            raise ValueError("Params cannot be None.")
        m = params['m']
        if 'logc' in params:
            c = 10**params['logc']
        elif 'c' in params:
            c = params['c']
        if 'logl' in params:
            l = 10**params['logl']
        elif 'l' in params:
            l = params['l']

        params_normalized = {'m': m, 'c': c, 'l': l}
        
        return params_normalized

    def minus_log_posterior(self, params = None):
        """
        Calculate the negative log posterior for given parameters m, c, logl.
        The log posterior is given by
            log P(m, c, l | V) = logP(m,c,l) - 0.5*log|S| + 0.5*log|D| + 0.5*j^T D j

        Params
        ------
        param : dict, optional
            Dictionary containing the parameters m, log_c (or c), log_l (or l).
            If None, use the current values of the class.
        """

        params = self.preprocess_params(params)
        m, c, l = params['m'], params['c'], params['l']

        if self._verbose:
            print("--> Calculating m_log_posterior with m, logc, logl:", params['m'], np.log10(c), np.log10(l))

        # Calculate S in real space.
        S_real  = self.calculate_S_real_space(m, c, l)

        # Calculate the inverse of S with Cholesky decomposition.
        self._Sinv = self.calculate_S_inv_cholesky(S_real)

        # Calculate Dinv.
        self._Dinv = self._M + self._Sinv 

        # Calculate mu.
        self._solve_mu()

        # Calculate the log determinants.
        self._logdetS = self.logdet(S_real)
        self._logdetD = -self.logdet(self._Dinv)

        # Calculate j^T D j = j^T mu
        self._jDj = np.dot(np.transpose(self._j), self._mu)

        # We assume P(m) = P(c) = 1, then we have no prior on m and c.
        prior = 0

        log_posterior =  (prior - 0.5*self._logdetS + 0.5*self._logdetD + 0.5*self._jDj)
        minus_log_posterior = - log_posterior

        if self._verbose:
            print("     + m_log_post:", minus_log_posterior)
            print("     + logdetS:", self._logdetS)
            print("     + logdetD:", self._logdetD)
            print("     + jDj:", self._jDj)

        return minus_log_posterior

    def calculate_S_real_space(self, m, c, l):
        """
        Calculate the covariance matrix S in real space.
        """
        factor = self.factor_power_spectrum(m, c)
        S_fspace = self.Wendland_kernel(factor, l)
        S_real = np.matmul(self.Ykm, np.matmul(S_fspace, self.Ykm_conj), dtype = "complex128").real
        return S_real


    def factor_power_spectrum(self, m, c):
        """
        Calculate the geometric mean of the power spectrum at q1 and q2.
        """
        p1 = self._Wendland.power_spectrum(self._q1, m, c)
        p2 = self._Wendland.power_spectrum(self._q2, m, c)
        return np.sqrt(p1 * p2)

    def Wendland_kernel(self, amplitude, l):
        """
        Calculate the Wendland kernel using the Wendland class.
        """
        return self._Wendland.matrix(amplitude, l)

    def calculate_mu_cholesky(self, Dinv):
        """
        Calculate the mean of the posterior using Cholesky decomposition.
        """
        try: 
            Dchol = scipy.linalg.cho_factor(Dinv)
            mu =  scipy.linalg.cho_solve(Dchol, self._j)

        except np.linalg.LinAlgError:
            U, s, V = scipy.linalg.svd(Dinv, full_matrices=False)
            s1 = np.where(s > 0, 1. / s, 0)
            mu = np.dot(V.T, np.multiply(np.dot(U.T, self._j), s1))
        return mu

    def calculate_S_inv_cholesky(self, S):
        """
        Calculate the inverse of S using Cholesky decomposition.
        """
        try: 
            Schol = scipy.linalg.cho_factor(S)
            Sinv =  scipy.linalg.cho_solve(Schol, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            U, s, V = scipy.linalg.svd(S, full_matrices=False)
            s1 = np.where(s > 0, 1. / s, 0)
            Sinv = np.dot(V.T, np.multiply(np.dot(U.T, np.eye(S.shape[0])), s1))

        return Sinv

    @property
    def sol(self):
        """Return the mean of the posterior."""
        return self._mu
    
    @property
    def jDj(self):
        """Return the current j^T D j term."""
        return self._jDj
    
    @property
    def logdetD(self):
        """Return the current log determinant of D."""
        return self._logdetD
    
    @property
    def logdetS(self):
        """Return the current log determinant of S."""
        return self._logdetS

    @property
    def DFT2(self):
        """Return the FourierTransform2D object."""
        return self._2DFT