import numpy as np
import time
import logging
import scipy
from scipy.special import gamma

from .constants import rad_to_arcsec, deg_to_rad
from .fourier2d import FourierTransform2D
from .minimizer import Powell

import abc
import matplotlib.pyplot as plt

"""
This module is based in classes and functions contain in Frankenstein-1D algorithm for fitting visibilities.
"""
class MAPEstimator(object):
    def __init__(self, Rmax, Geometry, N = 50):
        """
        Maximum A Posteriori Estimator for the parameters of the Gaussian Process
        Params
        ------
        Rmax : float
            Maximum radius to model, in arcseconds.
        Geometry : Geometry
            Geometry object containing the disc geometry.
        N : int, optional
            Number of radial points to use in the model. Default is 50.
            If use N > 50, the optimization can be very slow.
        """
        self._Rmax = Rmax
        self._N = N
        self._Geometry = Geometry
        self._FF = FourierBesselFitter(self._Rmax*rad_to_arcsec, N, self._Geometry)

        self._set_minimizer = False
        self._minimizer = Powell

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
    
    def set_minimizer(self, minimizer, initial_guess = {"m": -2, "logl": 4}):
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
        p = -2.
        params = { "p" : p }

        if initial_guess is not None:
            if "m" in initial_guess : 
                params["m"] = float(initial_guess["m"])
            else:
                # dict must contain m.
                params["m"] = -2.
            if "logl" in initial_guess : 
                params["logl"] = float(initial_guess["logl"])
        else:
            # By default, m and logl participate in the optimization.
            params["m"] = -2.
            params["logl"] = 4.

        self._initial_guess = params

        return list(params.values())
    
    def process_x_minimizer(self, x, final_result = False):
        """
        This process depends on the initial guess.
        Minimizer always optimizes p and m, and logl if present in the initial guess.
        Params
        ------
        x : array-like
            Parameters to process. If len(x) == 2, x = [p, m] and logl = 4.0.
        final_result : bool, optional
            If True, return the parameters readable for the user, i.e., m, c, l.
        """
        guess = self._initial_guess
        result = {k: v for k, v in zip(guess.keys(), x)}

        p = result["p"] # p is always present.
        m = result["m"] # m is always present.
        logc = result["p"] - 5*m

        if "logl" in result:
            logl = result["logl"]
        else:
            logl = 4

        if final_result:
            return {'m': m, 'c': 10**logc, 'l': 10**logl}
        else:
            return {'m': m, 'logc': logc,  'logl': logl}

    def optimize(self, data, initial_guess = {"m": -2, "logl": 4}):
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
        print("Fitting the visibilities with Frank2D with 2DFT...")
        # Fit with Frankenstein1D scheme.
        start_time = time.time()
        u, v, vis, weights = data['u'], data['v'], data['vis'], data['weights']
        self._FF.fit(u, v, vis, weights)
        self._GM = self._FF.GaussianModel
        self._minus_log_posterior = self._GM.minus_log_posterior
        self._p0 = self._get_p0()

        end_time = time.time()
        print(f"  + Time to set up the matrices: {end_time - start_time:.2f} seconds")

        print("Setting the initial guess...")
        initial_guess = self.process_initial_guess(initial_guess)

        if not self._set_minimizer:
            minimizer = Powell(self.eval_prob, initial_guess)
            self.set_minimizer(Powell)
        
        print("Optimizing the posterior...")
        self._minimizer.run()

        x = self._minimizer.solution

        # Normalize the parameters.
        self._best = self.process_x_minimizer(x, final_result = True)
      
    def _get_p0(self):
        """
        Get the minus log posterior for the initial guess.
        """
        GM = self._GM

        params = {'m': 0, 'logc': -2, 'logl': 4}
        p0 = GM.minus_log_posterior(params)
        jDj0, logdetS0, logdetD0 = GM.jDj, GM.logdetS, GM.logdetD

        return p0, jDj0, logdetS0, logdetD0
    
    def eval_prob(self, x):
        """
        Evaluate the negative log posterior for given parameters.
        Params
        ------
        x : array-like
            Parameters to evaluate. If len(x) == 2, x = [p, m] and logl = 4.0.
        """

        GM = self._GM

        p0, jDj0, logdetS0, logdetD0 = self._get_p0()
        
        x_ = self.process_x_minimizer(x)
        params = {'m': x_['m'], 'logc': x_['logc'], 'logl': x_['logl']}

        current = GM.minus_log_posterior(params) - p0

        jDj, S, D = GM.jDj - jDj0, GM.logdetS - logdetS0, GM.logdetD - logdetD0
        self.save_results(params['m'], 10**params['logc'], 10**params['logl'], jDj, D, S, current)
            
        #print(f'------> p: {p:0.3f}, m: {m:0.3f}, logc: {logc:3e}, logl: {logl:3e}: {minus_log_likelihood:0.2f} ' +
        #    f'(jDj={jDj:.3e}, |S|={S:.3e}, |D|={D:.3e})')

        return current

    def save_results(self, m, c, l, jDj, logdetD, logdetS, minus_log_posterior):
        """
        Save the results of the posterior optimization.
        """
        self._ms.append(m)
        self._cs.append(c)
        self._ls.append(l)
        
        self._jDjs.append(jDj)
        self._logdetDs.append(logdetD)
        self._logdetSs.append(logdetS)
        self._minus_log_posteriors.append(minus_log_posterior)
    
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
        """Return the GaussianModel object."""
        return self._GM

class FourierBesselFitter(object):
    """
    Fourier-Bessel series model for fitting visibilities
    """

    def __init__(self, Rmax, N, geometry=None, nu=0, block_data=True,
                 assume_optically_thick=True, scale_height=None,
                 block_size=10 ** 5, verbose=True, geometry_on = True):
        
        
        Rmax /= rad_to_arcsec

        self._geometry = geometry

        self._2DFT = FourierTransform2D(Rmax, N, geometry)
        self._Rmax = Rmax*rad_to_arcsec

        if assume_optically_thick:
            if scale_height is not None:
                raise ValueError("Optically thick models must have zero "
                                 "scale-height")
            model = 'opt_thick'
        elif scale_height is not None:
            model = 'debris'
        else:
            model = 'opt_thin'

        self._vis_map = VisibilityMapping(self._2DFT, geometry, 
                                          model, geometry_on = geometry_on ,scale_height=scale_height,
                                          block_data=block_data, block_size=block_size,
                                          check_qbounds=False, verbose=verbose)

        self._verbose = verbose

        self._info  = {'Rmax' : self._2DFT.Rmax * rad_to_arcsec,
                       'N' : self._2DFT.size
                       }

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
        self._Wvalues = mapping['W']
        self._identities = mapping['identities']

        self._H0 = mapping['null_likelihood']

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

class VisibilityMapping:
    r"""Builds the mapping between the visibility and image planes.

    VisibilityMapping generates the transform matrices :math:`H(q)` such that
    :math:`V_\nu(q) = H(q) I_\nu`. It also uses these to construct the design
    matrices :math:`M` and :math:`j` used in the fitting. 
    
    VisibilityMapping supports following models:
      1. An optically thick and geometrically thin disc
      2. An optically thin and geometrically thin disc
      3. An opticall thin disc with a known Gaussian verictal structure.
    All models are axisymmetric.
    """
    def __init__(self, DFT, geometry,  
                 vis_model='opt_thick', geometry_on = True, scale_height=None, block_data=True,
                 block_size=10 ** 5, check_qbounds=True, verbose=True):
        
        _vis_models = ['opt_thick', 'opt_thin', 'debris']
        if vis_model not in _vis_models:
            raise ValueError(f"vis_model must be one of {_vis_models}")

        # Store flags
        self._vis_model = vis_model
        self.check_qbounds = check_qbounds
        self._verbose = verbose 
        self.geometry_on = geometry_on

        self._chunking = block_data
        self._chunk_size = block_size

        self._2DFT = DFT
        self._geometry = geometry

        # Check for consistency and report the model choice.
        self._scale_height = None
        if self._vis_model == 'opt_thick':
            if self._verbose:
                logging.info('  Assuming an optically thick model (the default): '
                             'Scaling the total flux to account for the source '
                             'inclination')
        elif self._vis_model == 'opt_thin':
            if self._verbose:
                logging.info('  Assuming an optically thin model: *Not* scaling the '
                             'total flux to account for the source inclination')
        elif self._vis_model == 'debris':
            if scale_height is None:
                raise ValueError('You requested a model with a non-zero scale height'
                                 ' but did not specify H(R) (scale_height=None)')
            self._scale_height = scale_height(self.r)
            self._H2 = 0.5*(2*np.pi*self._scale_height / rad_to_arcsec)**2
            
            if self._verbose:
                logging.info('  Assuming an optically thin model but geometrically '
                             'thick model: *Not* scaling the total flux to account for '
                             'the source inclination')
   
    def map_visibilities(self, u, v, V, weights, frequencies=None, geometry=None):                         #--------> 3
        r"""
        Compute the matrices :math:`M` abd :math:`j` from the visibility data.

        Also compute the null likelihood,
        .. math:
            `H0 = 0.5*\log[det(weights/(2*np.pi))]
             - 0.5*np.sum(V * weights * V):math:`
        
        """

        if geometry is None:
            geometry = self._geometry

        if self._verbose:
            logging.info('    Building visibility matrices M and j')
        
        q = np.hypot(u, v)

        # Check consistency of the uv points with the model
        self._check_uv_range(q)

        w = (np.ones_like(V) * weights).real

        multi_freq = True
        if frequencies is None:
            multi_freq = False
            frequencies = np.ones_like(V)
        channels = np.unique(frequencies)
        Ms = np.zeros([len(channels), self.size, self.size], dtype='f8')
        js = np.zeros([len(channels), self.size], dtype='f8')

        self._identities = []

        for i, f in enumerate(channels):
            idx = frequencies == f

            qi = q[idx]
            ki = np.ones(len(q[idx]))#k[idx]
            wi = w[idx]
            Vi = V[idx]
        
            # If chunking is used, we will build up M and j chunk-by-chunk
            if self._chunking:
                Nstep = int(self._chunk_size / self.size + 1)
            else:
                Nstep = len(Vi)

            start = 0
            end = Nstep
            Ndata = len(Vi)

            while start < Ndata:
                qs = qi[start:end]
                us = u[start:end]
                vs = v[start:end]
                ks = ki[start:end]
                ws = wi[start:end]
                Vs = Vi[start:end]

                X = self._get_mapping_coefficients(ks, us, vs)

                wXT = np.matmul(np.transpose(np.conjugate(X)), np.diag(ws), dtype = "complex128")
                val = np.matmul(wXT, X, dtype="complex128")

                Ms[i] += val.real
                js[i] += np.matmul(wXT, Vs, dtype="complex128").real

                start = end
                end = min(Ndata, end + Nstep)

        N = int(np.sqrt(self._2DFT.size))

        # Compute likelihood normalization H_0, i.e., the
        # log-likelihood of a source with I=0.
        H0 = 0.5 * np.sum(np.log(w / (2 * np.pi)) - V * w * V)

        if multi_freq:
            return {
                'mult_freq' : True,
                'channels' : channels,
                'M' : Ms,
                'j' : js,
                'null_likelihood' : H0,
                'hash' : [True, geometry, self._vis_model, self._scale_height],
            }
        else: 
            return {
                'mult_freq' : False,
                'channels' : None,
                'M' : Ms[0],
                'j' : js[0],
                'null_likelihood' : H0,
                'hash' : [False, geometry, self._vis_model, self._scale_height],
                'V' : Vi,
                'W' : wi,
                'identities': self._identities,
            }

    def predict_visibilities(self, I, u, v, k=None, geometry=None):
        r"""Compute the predicted visibilities given the brightness profile, I

        """
        # Chunk the visibility calulation for speed
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

            H = self._get_mapping_coefficients(ki, ui, vi, geometry)

            V.append(np.dot(H, I))
        return np.concatenate(V)

    def _get_mapping_coefficients(self, ks, u, v, geometry=None, inverse=False):
        """Get :math:`H(q)`, such that :math:`V(q) = H(q) I_\nu`"""
        scale = 1
        if self._vis_model == 'opt_thick':
            # Optically thick & geometrically thin
            if geometry is None:
                if not self.geometry_on:
                    scale = 1
                else:   
                    geometry = self._geometry
                    #scale = np.cos(geometry.inc * deg_to_rad)
                    scale = 1

        elif self._vis_model == 'opt_thin':
            # Optically thin & geometrically thin
            scale = 1

        elif self._vis_model == 'debris':
            # Optically thin & geometrically thick
            scale = np.exp(-np.outer(ks*ks, self._H2))
        else:
            raise ValueError("model not supported. Should never occur.")
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
        #return self._DHT.q
        return self._2DFT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._DHT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        #return self._DHT.size
        return self._2DFT.size

    @property
    def scale_height(self):
        "Vertial thickness of the disc, unit = arcsec"
        if self._scale_height is not None:
            return self._scale_height
        else:
            return None

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

        # Wendland kernel parameters.
        self._r = np.sqrt((u1-u2)**2 + (v1-v2)**2)
        self._j_W, self._k_W = 4, 1

        # Parameters to optimize.
        self.m = -2
        self.logc = 8
        self.logl = np.log10(7e4)

        self._optimizing = False

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

    def minus_log_posterior(self, param = None):
        """
        Calculate the negative log posterior for given parameters m, c, logl.
        The log posterior is given by
            log P(m, c, l | V) = logP(m,c,l) - 0.5*log|S| + 0.5*log|D| + 0.5*j^T D j

        Params
        ------
        param : dict, optional
            Dictionary containing the parameters m, log_c, log_l.
            If None, use the current values of the class.
        """

        if param is not None:
            m = param['m']
            c = 10**param['logc']
            l = 10**param['logl']
        else:
            m = self.m
            c = 10**self.log_c
            l = 10**self.log_l

        # Calculate S in real space.
        #start_time = time.time()
        S_real  = self.calculate_S_real_space(m, c, l)
        #print("+ %.2f seconds for S_real " % (time.time() - start_time))

        # Calculate the inverse of S with Cholesky decomposition.
        #start_time = time.time()
        self._Sinv = self.calculate_S_inv_cholesky(S_real)
        #print("+ %.2f seconds for S_real_inv " % (time.time() - start_time))

        # Calculate Dinv.
        self._Dinv = self._M + self._Sinv 

        # Calculate mu.
        #start_time = time.time()
        self._solve_mu()
        #print("+ %.2f seconds for mu " % (time.time() - start_time))

        # Calculate the log determinants.
        #start_time = time.time()
        self._logdetS = np.linalg.slogdet(S_real)[1]
        #print("+ %.2f seconds for log|S|  " % (time.time() - start_time))

        #start_time = time.time()
        self._logdetD = -np.linalg.slogdet(self._Dinv)[1]
        #print("+ %.2f seconds for log|D| " % (time.time() - start_time))

        # Calculate j^T D j = j^T mu
        self._jDj = np.dot(np.transpose(self._j), self._mu)

        # We assume P(m) = P(c) = 1, then we have no prior on m and c.
        prior = 0

        log_posterior =  (prior - 0.5*self._logdetS + 0.5*self._logdetD + 0.5*self._jDj)
        minus_log_posterior = - log_posterior

        return minus_log_posterior

    def calculate_S_real_space(self, m, c, l):
        factor = self.factor_power_spectrum(m, c)
        S_fspace = self.Wendland_kernel(factor, l)
        S_real = np.matmul(self.Ykm, np.matmul(S_fspace, self.Ykm_conj), dtype = "complex128").real

        return S_real

    def power_spectrum(self, q, m, c):
        if not np.isscalar(q):  
            q[q == 0] = self._min_freq
        elif q == 0:
            q = self._min_freq
        return c*(q**m)

    def factor_power_spectrum(self, m, c):
        p1 = self.power_spectrum(self._q1, m, c)
        p2 = self.power_spectrum(self._q2, m, c)
        return np.sqrt(p1 * p2)
    
    def P_k(self, r, k):    
        if k == 0:
            return np.ones_like(r)  # P_0(r) = 1
        elif k == 1:
            return 4*r +1  # P_1(r) = 4r + 1
        elif k == 2:
            return (35/3)*r**2 +6*r + 1  # P_2(r) = 35r^2 + 18r + 3
        else:
            raise ValueError("k must be 0, 1, or 2.")

    def Wendland_kernel(self, amplitude, l):
        #print("Wendland kernel")
        r = self._r
        H = 2*1.897367*l
        r_normalized = r/H
        factor = (1 - r_normalized)**self._j_W
        factor[r_normalized > 1] = 0

        return  amplitude * factor * self.P_k(r_normalized, self._k_W)

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
    