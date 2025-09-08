import numpy as np
import time
import logging
import scipy

from constants import rad_to_arcsec, deg_to_rad
from fourier2d import FourierTransform2D

import abc
import matplotlib.pyplot as plt

class FrankRadialFit(metaclass=abc.ABCMeta):
    """
    Base class for results of frank fits.

    Parameters
    ----------
    vis_map : VisibilityMapping object
        Mapping between image and visibility plane. 
    info: dict
        Dictionary containing useful quantities for reproducing a fit
        (such as the hyperparameters used)
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """
    def __init__(self, vis_map, info, geometry):
        self._vis_map = vis_map
        self._geometry = geometry
        self._info = info

    def predict(self, u, v, I=None, geometry=None):
        r"""
        Predict the visibilities in the sky-plane

        Parameters
        ----------
        u, v : array, unit = :math:`\lambda`
            uv-points to predict the visibilities at
        I : array, optional, unit = Jy
            Intensity points to predict the vibilities of. If not specified,
            the mean will be used. The intensity should be specified at the
            collocation points, I[k] = :math:`I(r_k)`
        geometry: SourceGeometry object, optional
            Geometry used to correct the visibilities for the source
            inclination. If not provided, the geometry determined during the
            fit will be used

        Returns
        -------
        V(u,v) : array, unit = Jy
            Predicted visibilties of a source with a radial flux distribution
            given by :math:`I` and the position angle, inclination and phase
            centre determined by the geometry object
        """
        if geometry is None:
            geometry = self._geometry

        if I is None:
            I = self.I

        #if geometry is not None:
        # u, v, wz = geometry.deproject(u, v, use3D=True)
        #else:
        wz = np.zeros_like(u)
            
        V = self._vis_map.predict_visibilities(I, u, v, wz, geometry=geometry)

        # Undo phase centering
        #if geometry is not None:
        #    _, _, V = geometry.undo_correction(u, v, V)

        return V

    def predict_deprojected(self, u, v, I=None, geometry=None,
                            block_size=10**5, assume_optically_thick=True):
        r"""
        Predict the visibilities in the deprojected-plane
        """
        if geometry is None:
            geometry = self._geometry
       
        if I is None:
            I = self.I

        V = self._vis_map.predict_visibilities(I, u, v, q*0, geometry=geometry)

        return V


    @abc.abstractproperty
    def MAP(self):
        pass

    @property
    def I(self):
        return self.MAP

    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._vis_map.r

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._vis_map.Rmax
    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._vis_map.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._vis_map.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._vis_map.size

    @property
    def geometry(self):
        """SourceGeometry object"""
        return self._geometry

    @property
    def info(self):
        """Fit quantities for reference"""
        return self._info



class FrankGaussianFit(FrankRadialFit):
    """
    Result of a frank fit with a Gaussian brightness model.
    """

    def __init__(self, DFT, fit, info={}, geometry=None):
        FrankRadialFit.__init__(self, DFT, info, geometry)
        self._fit = fit


    @property
    def mean(self):
        """Posterior mean, unit = Jy / sr"""
        return self._fit.mean

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return self.mean

    @property
    def covariance(self):
        """Posterior covariance, unit = (Jy / sr)**2"""
        return self._fit.covariance

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        return self._fit.power_spectrum


class FourierBesselFitter(object):
    """
    Fourier-Bessel series model for fitting visibilities
    """

    def __init__(self, Rmax, N, geometry=None, nu=0, block_data=True,
                 assume_optically_thick=True, scale_height=None,
                 block_size=10 ** 5, verbose=True, geometry_on = True):                 #--------> 1
        
        
        Rmax /= rad_to_arcsec

        self._geometry = geometry

        self._DFT = FourierTransform2D(Rmax, N, geometry)
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

        self._vis_map = VisibilityMapping(self._DFT, geometry, 
                                          model, geometry_on = geometry_on ,scale_height=scale_height,
                                          block_data=block_data, block_size=block_size,
                                          check_qbounds=False, verbose=verbose)


        self._verbose = verbose

        self._info  = {'Rmax' : self._DFT.Rmax * rad_to_arcsec,
                       'N' : self._DFT.size
                       }

    def preprocess_visibilities(self, u, v, V, weights=1):                                      #--------> 2
        r"""Prepare the visibilities for fitting. 
        """
        return self._vis_map.map_visibilities(u, v, V, weights)

    def _build_matrices(self, mapping):                                                          #--------> 4
        r"""
        Compute the matrices M and j from the visibility data.
        """  
        self._M = mapping['M']
        self._j = mapping['j']
        self._V = mapping['V']
        self._Wvalues = mapping['W']
        self._identities = mapping['identities']

        self._H0 = mapping['null_likelihood']

    def fit(self, u, v, V, weights=1):                                                           #--------> 0
        r"""
        Fit the visibilties
        """
        # self._geometry.fit(u, v, V, weights)                                                   #----> Should we?

        mapping = self.preprocess_visibilities(u, v, V, weights)
        self._build_matrices(mapping)

        return self._fit()

    def _fit(self):                                                                              #--------> 5
        """Fit step. Computes the best fit given the pre-processed data"""
        self._GaussianModel = GaussianModel(self._DFT, self._M, self._j, self._Rmax,
                            noise_likelihood=self._H0,
                            Wvalues= self._Wvalues, 
                            V = self._V,
                            identities = self._identities
                            )
        
        self._sol = FrankGaussianFit(self._vis_map, self._GaussianModel, self._info,
                                     geometry=None)
 
        return self._sol

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

        self._DFT = DFT
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

        # Use only the real part of V. UPDATED: We need imaginary part.
        V = V
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

        N = int(np.sqrt(self._DFT.size))

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

    def invert_visibilities(self, V, R, geometry=None):
        r"""Compute the brightness, I, from the visibilities. 
        
        Note this method does not work for an arbitrary distribution of 
        baselines and therefore cannot be used to determine the brightness
        given a generic set of data. Instead it needs the visibilities at 
        collocation points of the DiscrteHankelTransform, q.

        For geometrically thick models the visibilities used must be those for
        which kz = 0.

        Given the above constraints, this method computes the inverse of
        predict_visibilites.
        """
        # Chunk the visibility calulation for speed
        R = np.atleast_1d(R)
        if self._chunking:
            Ni = int(self._chunk_size / self.size + 1)
        else:
            Ni = len(R)

        end = 0
        start = 0
        I = []
        while end < len(R):
            start = end
            end = start + Ni
            Ri = R[start:end]

            H = self._get_mapping_coefficients(Ri, 0, geometry, inverse=True)

            I.append(np.dot(H, V))
        return np.concatenate(I)[R < self.Rmax]


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

        H = self._DFT.coefficients(u, v, direction=direction)*scale

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
        return self._DFT.r * rad_to_arcsec

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._DFT.Rmax * rad_to_arcsec

    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        #return self._DHT.q
        return self._DFT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._DHT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        #return self._DHT.size
        return self._DFT.size

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

    def __init__(self, DFT, M, j, Rmax, p=None, scale=None, guess=None,
                 Nfields=None, noise_likelihood=0,
                 Wvalues = None, V = None,
                 identities=None
                 ):

        self._DFT = DFT
        self._Wvalues = Wvalues
        self._V = V
        self._identities = identities
        self._N = int(np.sqrt(M.shape[0]))
        self._Rmax = Rmax

        # Correct shape of design matrix etc.        
        if len(M.shape) == 2:
            M = M.reshape(1, *M.shape)
        if len(j.shape) == 1:
            j = j.reshape(1, *j.shape)

        # Number of frequencies / radial points
        Nf, Nr = j.shape
        
        # Get the number of fields                                              #----> Should we?
        if Nfields is None:
            if guess is None:
                Nfields = 1
            else:
                guess = guess.reshape(-1, Nr)
                Nfields = guess.shape[0]
        elif guess is not None:
            guess = guess.reshape(Nfields, Nr)
        self._Nfields = Nfields

        if scale is None:
            self._scale = np.ones([Nf, Nfields], dtype='f8')
        else:
            self._scale = np.empty([Nf, Nfields], dtype='f8')
            self._scale[:] = scale.reshape(Nf, -1)    
        
        # Compute the design matrix
        self._M = np.zeros([Nr*Nfields, Nr*Nfields], dtype='f8')
        self._j = np.zeros(Nr*Nfields, dtype='f8')
        for si, Mi, ji in zip(self._scale, M, j):
            
            for n in range(0, Nfields):
                sn = n*Nr
                en = (n+1)*Nr 

                self._j[sn:en] += si[n] * ji
                for m in range(0, Nfields):
                    sm = m*Nr
                    em = (m+1)*Nr 

                    self._M[sn:en, sm:em] += si[n]*si[m] * Mi

        self._like_noise = noise_likelihood

        # Frankenstein2D 
        self.u, self.v = self._DFT.uv_points
        self.Ykm = self._DFT.coefficients(direction="backward")
        self.Ykm_f = self._DFT.coefficients(direction="forward")
        self.Ykm_f_conj = self.Ykm_f.conj()
        self.Ykm_conj = self.Ykm.conj()

        # Weights
        weights = self._Wvalues.copy()
        weights_regularized = weights + 1e-8  # Avoid division by zero.

        # Spatial frequencies and related.
        u1, u2 = np.meshgrid(self.u, self.u)
        v1, v2 = np.meshgrid(self.v, self.v)
        self.data = [u1, u2, v1, v2]
        self._q1 = np.hypot(u1, v1)
        self._q2 = np.hypot(u2, v2)
        self._qs = np.hypot(self.u, self.v)
        self._min_freq = np.sort(np.abs(np.unique(self.v)))[1]
        self._max_freq = np.max(self._qs)

        print("Minimum frequency in the data:", self._min_freq)
        print("Maximum frequency in the data:", self._max_freq)


        # Wendland kernel parameters.
        self.m , self.log_c = -2, 8
        self.log_l = 6

        self.c = 10**self.log_c  # Convert log_c to c.
        self.l = 10**self.log_l
        self._r = np.sqrt((u1-u2)**2 + (v1-v2)**2)
        self._j_W, self._k_W = 4, 1

        # Constants in the log probability of the posterior.
        #self._log_det_N = np.sum(np.log(1/weights_regularized))
        #self._VtNm1V = (np.conjugate(self._V).T @ (weights_regularized * self._V)).real
        #self._factor = np.log(2*np.pi)

        self._alpha = 1.5
        self._beta = 1.5*((self.l)**(-2))
        self._p0 = 1e-10

        self._optimizing = False

    def one_fit(self, param):
        """
        Fit the model with the given parameters m, c, log_l.
        """
        if param is not None:
            self.m, self.log_c, self.log_l = param
            self.c = 10**self.log_c  # Convert log_c to c.
            self.l = 10**self.log_l
        print("                 --------------> Fitting with m, c, log_l:", self.m, self.c, self.log_l)
        S_real = self.calculate_S_real_space(self.m, self.c, self.l)
        self._Sinv = np.linalg.inv(S_real)
        self._Dinv = self._M + self._Sinv

        self._fit()
    
    def optimize_posterior(self):
        self._logdets_D = []
        self._logdets_S = []
        self._jTDj = []
        self._logprior_2 = []
        self._logprior_1 = []
        self._log_posteriors = []
        self._likelihoods =[]
        self._times = []
        self._ms = []
        self._cs = []
        self._ls = []

        self._optimization_time = 0

        self._x0 = [self.m, self.log_c]  # Initial guess for m, c, log_l.
        self._m_bounds, self._log_c_bounds, self._log_l_bounds = (-6, -1.9), (7,30), (5, 6.5)
        self._bounds = [self._m_bounds, self._log_c_bounds]

        
        print("....................................OPTIMIZING.................................. ")
        print("                 --------------> Initial parameters: m, c, log_l:", self.m, self.log_c, self.log_l)
        print("                 --------------> Bounds for m, log_c:", self._m_bounds, self._log_c_bounds)
        self.optimize() # Finding best parameters to S matrix.
        self.one_fit(param = self._best)  # Fit with the best parameters.
        print("....................................OPTIMIZING.................................. ")

    def optimize(self, method = "emcee"):
        """
        from scipy.optimize import minimize, check_grad
        print("Checking the gradient first..")
        log_l = np.log(self.l)  # Convert l to log_l for optimization
        err = check_grad(self.likelihood, self.grad_likelihood, [self.m, self.c, log_l])
        print("                        -----------> Gradient error:", err)
        """

        start_time = time.time()
        self._optimizing = True
        self._optimization_res = None
        self._best =  None
        optimized_params = None

        if method == "differential_evolution":
            self._optimization_res = self.differential_evolution_optimizer()
            self._best = self._optimization_res.x

        if method == "genetic":
            self._optimization_res = self.genetic_optimizer()
            self._best = self._optimization_res.X
        
        if method == "emcee":
            self.emcee_optimizer()
        else:
            self._optimization_res = self.scipy_minimizer(method=method)
            self._best = self._optimization_res.x
        
        self.m, self.log_c, self.log_l = self._best
        self.c = 10**self.log_c  # Convert log_c to c.
        self.l = 10**self.log_l

        print("                 --------------> Best parameters founded for: m, log_c: ", self.m, self.log_c, self.log_l)

        self._optimization_time = time.time() - start_time

        print("--- %s minutes to minimizing ---" % (self._optimization_time/60))

    def _fit(self):
        """Compute the mean and variance"""
        self._mu = self.calculate_mu_cholesky(self._Dinv)
        #self._mu = self.calculate_mu_gc(Dinv)

    def m_log_posterior(self, param):
        from scipy.special import gamma
        m, log_c = param
        
        log_l = self.log_l

        c = 10**(log_c)  # Convert log_c to c.
        l = 10**(log_l)  # Convert log_l to l.
        print("......................................... Calculating likelihood with m, c, log(l): ", m, c, log_l)
        start_time = time.time()

        S_real  = self.calculate_S_real_space(m, c, l)

        #start_time = time.time()
        self._Sinv = np.linalg.inv(S_real)
        #print("       --- %s calculate S_real_inv ---" % (time.time()/60 - start_time/60))

        self._Dinv = self._M + self._Sinv 

        #start_time = time.time()
        self._fit()
        mu = self._mu
        #print("       --- %s mu ---" % (time.time()/60 - start_time/60))

        # Calculate the log determinants.
        #start_time = time.time()
        logdetS = np.linalg.slogdet(S_real)[1]
        #print("       --- %s determinant of S ---" % (time.time()/60 - start_time/60))

        #start_time = time.time()
        #print("       --- %s calculate D ---" % (time.time()/60 - start_time/60))

        #start_time = time.time()
        logdetD = -np.linalg.slogdet(self._Dinv)[1]
        #print("       --- %s determinant of D ---" % (time.time()/60 - start_time/60))

        #if we assume P(m) = P(c) = 1, then we have no prior on m and c. 
        # # np.log(np.abs((1/m)*(1/c))) 
        from scipy.special import gamma
        def log_inverse_gamma(q_i, p0):
            p_q_i = self._power_spectrum(q_i, m, c)
            #inverse_gamma = ((p0/p_q_i)**self._alpha)*np.exp(-p0/p_q_i)
            return self._alpha*np.log(p0/p_q_i) - (p0/p_q_i)

        #N = 300
        #array = np.unique(self._qs)
        #indexes = np.linspace(0, len(array)-1, N, dtype=int)[1:][:-1]  # Avoid the first and last points.
        #freqs_spaced = array[indexes]
        #print(" -------------------------------------------------> frequencies: ", freqs_spaced)
        log_prior_1 = 0 #(log_inverse_gamma(self._min_freq, 1e-15) + log_inverse_gamma(self._max_freq, 1e-15)) #+ np.sum(log_inverse_gamma(freqs_spaced, 1e-)))
        #print("                 --------------> sampling ", N, " points of ", len(array))
        #prior_1 = inverse_gamma(self._min_freq) * inverse_gamma(self._max_freq)
        
        # We want a gamma prior on l^2 (or an inverse gamma prior on l^(-2)).
        #constant1 = -np.log(self._beta*gamma(self._alpha)) + np.log(2) +(self._alpha + 1)*np.log(self._alpha) - np.log(gamma(self._alpha + 1))
        #k = l**(-2)
        #log_inverse_gamma_l = 2*(self._alpha + 1)*log_l - (self._beta/k) - 3*log_l
        #log_prior_2 = log_inverse_gamma_l
        log_prior_2 = 0  # No prior on l, so we set it to zero.

        jTDj = np.dot(np.transpose(self._j), mu)

        #constants_to_ignore = - constant1 + self._log_det_N + 0.5*self._VtNm1V + self._factor
        log_posterior =  (log_prior_1 + log_prior_2 - 0.5*logdetS + 0.5*logdetD + 0.5*jTDj) # + constants_to_ignore
        m_log_posterior = - log_posterior

        total_time = time.time() - start_time
        print("--- %s seconds to calculate log_likelihood ---" % (total_time))

        if self._optimizing:
            self._ms.append(m)
            self._cs.append(log_c)
            self._ls.append(log_l)
            
            self._logprior_1.append(log_prior_1)
            self._logprior_2.append(log_prior_2)
            self._jTDj.append(jTDj)
            self._logdets_D.append(logdetD)
            self._logdets_S.append(logdetS)
            self._log_posteriors.append(log_posterior)

            self._times.append(total_time)

        return m_log_posterior, 0.5*logdetS, -0.5*logdetD,  -0.5*jTDj

    def grad_likelihood(self, param):
        m, c = param

        print("Calculating gradient of likelihood with m, c, l: ", m, c)

        l = 10**(self.log_l)  # Convert log_l to l

        P_q1 = self._power_spectrum(self._q1, m, c)
        P_q2 = self._power_spectrum(self._q2, m, c)
        Ps = np.sqrt(P_q1 * P_q2)
        log_q1 = np.log(self._q1)
        log_q2 = np.log(self._q2)

        # Wendland
        c_W =2*1.897367
        H = c_W*l
        r_n = self._r/H
        factor_m1 = (1 - r_n)**(self._j_W-1)
        factor_m1[r_n > 1] = 0
        factor = (1 - r_n)**(self._j_W)
        factor[r_n > 1] = 0
        Pk = self.P_k(r_n,self._k_W)
        W = factor * Pk
        dPk = None

        if self._k_W == 1:
            dPk = -4*r_n/l

        # dlogP(m, c, l)/dlogm
        P_min_freq = self._power_spectrum(self._min_freq, m, c)
        P_max_freq = self._power_spectrum(self._max_freq, m, c)
        dlog_prior_dlogm = m*self._alpha* P_min_freq * P_max_freq * self._p0 * (np.log(self._min_freq)/ P_min_freq + np.log(self._max_freq)/P_max_freq)
        
        # dlogP(m, c, l)/dlogc
        dlog_prior_dlogc = c*self._alpha* P_min_freq * P_max_freq * self._p0 * (self._min_freq**(2*m)/ P_min_freq**3 + self._max_freq**(2*m) / P_max_freq**3)
        
        # dlogP(m, c, l)/dlogl
        dlog_prior_dlogl = 0  
        #dlog_prior_dlogl = 2*(self._alpha-1) - 2*(l**2)*self._beta

        # dlogS_dlogm
        dlogS_dlogm = (
            m/2
            * W
            * ((log_q1*P_q2 + log_q2*P_q1) / (P_q1 * P_q2)**(1/2))
        )

        # dlogS_dlogc
        dlogS_dlogc = (
            W/2
            * ((log_q1 + log_q2) / (P_q1 * P_q2)**(1/2))
        )

        # dlogS_dlogl
        dlogS_dlogl = (
            l
            * Ps
            * (-self._j_W*(r_n/H)*c_W*factor_m1*Pk + factor * dPk)
        )
        
        # dlogjTmu_dlogm, dlogjTmu_dlogc, dlogjTmu_dlogl
        S_real  = self.calculate_S_real_space(m, c, l)
        S_real_inv = np.linalg.inv(S_real)
        Dinv = self._M + S_real_inv
        D = np.linalg.inv(Dinv)
        mu = self.calculate_mu_gc(Dinv)
        mu_T = mu.T

        Sm1_Fm1 = np.matmul(S_real_inv, self.Ykm, dtype="complex128")
        Sm1_F_f_m1 = np.matmul(S_real_inv, self.Ykm_f_conj, dtype="complex128")
        F_Sm1 = np.matmul(self.Ykm_conj, S_real_inv, dtype="complex128")

        # dlogjTmu / dlogm
        dlogjTmu_dlogm = (
            mu_T @
            Sm1_Fm1 @
            dlogS_dlogm @
            F_Sm1 @
            mu
        )
        # djT\mu / dlogc   
        dlogjTmu_dlogc = (
            mu_T @
            Sm1_Fm1 @
            dlogS_dlogc @
            F_Sm1 @
            mu
        )

        # dlogjTmu / dlogl
        dlogjTmu_dlogl = (
            mu_T @
            Sm1_Fm1 @
            dlogS_dlogl @
            F_Sm1 @
            mu
        )

        # dlog_0p5_det_DSm1_dlogm
        F_Sm1_D = F_Sm1 @ D

        dlog_0p5_det_DSm1_dlogm = (
            1/2 *
            (
                np.trace(Sm1_Fm1 @ dlogS_dlogm @ F_Sm1_D) -
                np.trace(Sm1_F_f_m1 @ dlogS_dlogm @ self.Ykm_f)
            )
        )
        
        dlog_0p5_det_DSm1_dlogc = (
            1/2 *
            (
                np.trace(Sm1_Fm1 @ dlogS_dlogc @ F_Sm1_D) - 
                np.trace(Sm1_F_f_m1 @ dlogS_dlogc @ self.Ykm_f) 
            )
        )
        
        dlog_0p5_det_DSm1_dlogl = (
            1/2 *
            (
                np.trace(Sm1_Fm1 @ dlogS_dlogl @ F_Sm1_D) -
                np.trace(Sm1_F_f_m1 @ dlogS_dlogl @ self.Ykm_f) 
            )
        )

        dlog_dlogm = dlogjTmu_dlogm + dlog_0p5_det_DSm1_dlogm + dlog_prior_dlogm
        dlog_dlogc = dlogjTmu_dlogc + dlog_0p5_det_DSm1_dlogc + dlog_prior_dlogc
        #dlog_dlogl = dlog_prior_dlogl + dlogjTmu_dlogl + dlog_0p5_det_DSm1_dlogl

        grad_m_log_likelihood = - np.array([
                            dlog_dlogm,
                            dlog_dlogc
                            ]
                        ).real

        return grad_m_log_likelihood

    def genetic_optimizer(self):
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.operators.sampling.rnd import FloatRandomSampling

        class MinPYMOO(ElementwiseProblem):
            def __init__(self, model, bounds):
                self.model = model
                bounds_m, bounds_c = bounds
                xl = np.array([bounds_m[0], bounds_c[0]], dtype=float)
                xu = np.array([bounds_m[1], bounds_c[1]], dtype=float)
                super().__init__(n_var=2, n_obj=1, n_constr=0, xl=xl, xu=xu)

            def _evaluate(self, x, out, *args, **kwargs):
                m, c = float(x[0]), float(x[1])
                f = self.model([m, c])
                if not np.isfinite(f):
                    f = 1e50
                out["F"] = f

        problem = MinPYMOO(self.likelihood, self._bounds)

        """
        algorithm = GA(
            pop_size=80,
            sampling=LHS(),
            crossover=SBX(prob=1, eta=0.1),
            mutation=PM(prob=1, eta=0.1),
            eliminate_duplicates=True,
            verbose=True
        )
        """

        # --- CMA-ES setup ---
        # Start from the middle of the box and use a conservative step size.
        algorithm = CMAES(
            x0=np.array(self._x0),
            sigma=0.5,
            # pop_size=None -> uses CMA-ES default; you can set a value like 8 or 12 for 2D.
            # pop_size=12,
            # diagonal=True  # uncomment to start with diagonal covariance if you want more stability
        )

        termination = get_termination("n_eval", 200)

        res = minimize(problem, algorithm, termination, seed=42, verbose=True)

        return res

    def differential_evolution_optimizer(self):
        from scipy.optimize import differential_evolution

        print("Using differential evolution for global optimization")

        result = differential_evolution(    self.likelihood,
                                            bounds=self._bounds,
                                            maxiter=50,
                                            popsize=10,
                                            mutation=(0.5, 1),
                                            tol=1e-6,
                                            seed=42
                                        )

        if not result.success:
            print("Optimization failed:", result.message)

        return result

    def scipy_minimizer(self, method="trust-ncg"):
        from scipy.optimize import minimize

        result = minimize(  self.likelihood,
                            x0=np.array(self._x0),
                            method=method, 
                            #jac=self.grad_likelihood,
                            tol=1e-9,
                            options={
                                #'maxiter': 100,
                                'verbose': 3,}, # 'gtol': 1e-9}
                            bounds=self._bounds)
        if not result.success:
            print("Optimization failed:", result.message)

        return result

    def log_posterior(self, params):
        from numpy.linalg import slogdet

        m, log_c, log_l = params[0], params[1], self.log_l

        c = 10.0**log_c
        l = 10.0**self.log_l   # Keep l fixed during emcee optimization.

        try:
            S = self.calculate_S_real_space(m, c, l)
            signS, logdetS = slogdet(S)
            if (signS <= 0) or (not np.isfinite(logdetS)):
                return -np.inf

            Sinv = np.linalg.inv(S)
            Dinv = self._M + Sinv

            signDinv, logdetDinv = slogdet(Dinv)
            if (signDinv <= 0) or (not np.isfinite(logdetDinv)):
                return -np.inf

            logdetD = -logdetDinv

            self._Sinv = Sinv
            self._Dinv = Dinv

            self._fit()
            
            mu = self._mu
            j = self._j


            jTDj = np.dot(j.T, mu)

            log_post = ( 0.5*logdetS
                        + 0.5*logdetD
                        + 0.5*jTDj)

            if not np.isfinite(log_post):
                return -np.inf

            return log_post

        except Exception:
            # Catch any numerical errors and return -inf log-posterior.
            return -np.inf

    def emcee_optimizer(self):
        import emcee
        import numpy as np

        nwalkers = 25 # before 20, 30, 80
        ndim = 3
        burn = 50
        steps = 200

        m_lo, m_hi = self._m_bounds
        c_lo, c_hi = self._log_c_bounds
        l_lo, l_hi = self._log_l_bounds

        # ---  Fixed slope from previous analysis and REPARAMETRIZATION---
        a = -4.2
        b = 1.1
        
        sm_rep, slogc_rep = 1, 1

        def to_rep(m, logc):
            logc_rep = (logc - (a*m + b)) / slogc_rep
            m_rep = m / sm_rep
            return np.array([m_rep, logc_rep])

        def from_rep(m_rep, logc_rep):
            m = m_rep * sm_rep
            logc = a*m + b + logc_rep * slogc_rep
            return np.array([m, logc])

        # --- Log-probability in (u,v) space ---
        def log_prob_rep(params):
            m_rep, logc_rep, logl = params
            m, logc = from_rep(m_rep, logc_rep)
            if not (m_lo <= m <= m_hi and c_lo <= logc <= c_hi and l_lo <= logl <= l_hi):
                return -np.inf
            return self.log_posterior([m, logc])

        # --- Initialize walkers around MAP (transformed) ---
        theta_map = np.array([-2, 8, 5.5])
        center = to_rep(theta_map[0], theta_map[1]), theta_map[2]
        u, v = to_rep(theta_map[0], theta_map[1])
        center = np.array([u, v, theta_map[2]])
        scale = np.array([1.0, 1.0, 1.0]) # Small spread in u (perpendicular), larger in v (along the diagonal)
        p0_rep = center + np.random.randn(nwalkers, 3) * scale

        # --- Sampler moves ---
        moves = [
            (emcee.moves.StretchMove(a=2.0), 0.6),
            (emcee.moves.DEMove(sigma=0.6), 0.4),
        ]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_rep, moves=moves)

        try:
            # corre normalmente; puedes interrumpir con Ctrl+C en cualquier momento
            state = sampler.run_mcmc(p0_rep, steps, progress=True)

        except KeyboardInterrupt:
            print("Sampling interrupted by user (Ctrl+C). Saving partial results...")

        finally:
            # siempre se ejecuta: guarda lo que haya
            self.sampler = sampler
            steps_done = sampler.iteration
            if steps_done == 0:
                print("No samples to save.")
                
            burn_eff = min(burn, max(steps_done - 1, 0))  # por si cortas muy temprano

            # muestras en espacio reparametrizado (post-burn)
            flat_rep  = sampler.get_chain(discard=burn_eff, flat=True)   # shape (Nsamp, 2)
            flat_logp = sampler.get_log_prob(discard=burn_eff, flat=True)

            # transforma a (m, logc)
            flat_samples = np.array([from_rep(mr, cr) for mr, cr in flat_rep[:, :2]])
            flat_samples = np.hstack([flat_samples, flat_rep[:, 2:3]])  # añade logl sin cambiar

            # MAP y resumen
            imax = np.argmax(flat_logp)
            theta_map = flat_samples[imax]
            med = np.median(flat_samples, axis=0)
            q16, q84 = np.percentile(flat_samples, [16, 84], axis=0)

            # guarda en el objeto
            self.samples = flat_samples
            self.samples_logp = flat_logp
            self._best = theta_map

            print(f"acceptance ≈ {np.mean(sampler.acceptance_fraction):.3f} | steps={steps_done} | burn={burn_eff}")
            print("MAP:", theta_map, " median:", med, " [-, +]:", med-q16, q84-med)

    def calculate_S_real_space(self, m, c, l):
        #start_time = time.time()
        factor = self.factor_power_spectrum(m, c)
        S_fspace = self.Wendland_kernel(factor, l)
        
        #print("--- %s minutes to calculate S---" % (time.time()/60 - start_time/60))
        #start_time = time.time()
        S_real = np.matmul(self.Ykm, np.matmul(S_fspace, self.Ykm_conj), dtype = "complex128").real
        #print("--- %s minutes to calculate S_real---" % (time.time()/60 - start_time/60))

        return S_real

    def _power_spectrum(self, q, m, c):
        if not np.isscalar(q):  
            q[q == 0] = self._min_freq
        elif q == 0:
            q = self._min_freq
        return c*(q**m)

    def factor_power_spectrum(self, m, c):
        p1 = self._power_spectrum(self._q1, m, c)
        p2 = self._power_spectrum(self._q2, m, c)
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
        try: 
            Dchol = scipy.linalg.cho_factor(Dinv)
            mu =  scipy.linalg.cho_solve(Dchol, self._j)

        except np.linalg.LinAlgError:
            U, s, V = scipy.linalg.svd(Dinv, full_matrices=False)
            s1 = np.where(s > 0, 1. / s, 0)
            mu = np.dot(V.T, np.multiply(np.dot(U.T, self._j), s1))
        return mu

    def calculate_mu_gc(self, Dinv):
        from scipy.sparse import csr_matrix, issparse
        
        start_time = time.time()
        mu, exitCode = self.bicgstab(Dinv, self._j, rtol = 1e-6, maxiter = 5000)
        print("--- %s minutes calculate mu---" % (time.time()/60 - start_time/60))
        print("Ended?: ", exitCode == 0)
        result = np.allclose(np.dot(Dinv, mu), self._j)
        print("Is the result correct?: ", result)
        if result:
            self._mu =  mu


        import matplotlib.pyplot as plt
        N = self._N
        I_reshape = mu.reshape(N, N)
        fig, ax = plt.subplots(figsize=(5, 5))
        Rmax = self._Rmax
        plot = ax.imshow(I_reshape, cmap="magma", vmax = 2e10, origin='lower', extent=[-Rmax, Rmax, -Rmax, Rmax])
        plt.gca().invert_xaxis()
        cmap = plt.colorbar(plot)
        cmap.set_label(r'I [Jy $sr^{-1}$]', size = 15)
        ax.set_title(r'Fit, ' + str(N) + 'x' + str(N) + 'points')
        ax.set_xlabel("x ['']")
        ax.set_ylabel("y ['']")
        plt.show()

        return mu
    
    def make_system(self, A, M, x0, b):
        """Make a linear system Ax=b
        """
        from numpy import asanyarray, asarray, array, zeros
        from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
        IdentityOperator

        _coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}


        def coerce(x,y):
            if x not in 'fdFD':
                x = 'd'
            if y not in 'fdFD':
                y = 'd'
            return _coerce_rules[x,y]


        def id(x):
            return x

        A_ = A
        A = aslinearoperator(A)

        if A.shape[0] != A.shape[1]:
            raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')

        N = A.shape[0]

        b = asanyarray(b)

        if not (b.shape == (N,1) or b.shape == (N,)):
            raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                            'incompatible')

        if b.dtype.char not in 'fdFD':
            b = b.astype('d')  # upcast non-FP types to double

        if hasattr(A,'dtype'):
            xtype = A.dtype.char
        else:
            xtype = A.matvec(b).dtype.char
        xtype = coerce(xtype, b.dtype.char)

        b = asarray(b,dtype=xtype)  # make b the same type as x
        b = b.ravel()

        # process preconditioner
        if M is None:
            if hasattr(A_,'psolve'):
                psolve = A_.psolve
            else:
                psolve = id
            if hasattr(A_,'rpsolve'):
                rpsolve = A_.rpsolve
            else:
                rpsolve = id
            if psolve is id and rpsolve is id:
                M = IdentityOperator(shape=A.shape, dtype=A.dtype)
            else:
                M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                                dtype=A.dtype)
        else:
            M = aslinearoperator(M)
            if A.shape != M.shape:
                raise ValueError('matrix and preconditioner have different shapes')

        # set initial guess
        if x0 is None:
            x = zeros(N, dtype=xtype)
        elif isinstance(x0, str):
            if x0 == 'Mb':  # use nonzero initial guess ``M @ b``
                bCopy = b.copy()
                x = M.matvec(bCopy)
        else:
            x = array(x0, dtype=xtype)
            if not (x.shape == (N, 1) or x.shape == (N,)):
                raise ValueError(f'shapes of A {A.shape} and '
                                f'x0 {x.shape} are incompatible')
            x = x.ravel()

        return A, M, x, b

    def _get_atol_rtol(self, name, b_norm, atol=0., rtol=1e-5):
        """
        A helper function to handle tolerance normalization
        """
        if atol == 'legacy' or atol is None or atol < 0:
            msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
                "if set, `atol` must be a real, non-negative number.")
            raise ValueError(msg)

        atol = max(float(atol), float(rtol) * float(b_norm))

        return atol, rtol

    def bicgstab(self, A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
             callback=None):
        """
        Solve a linear system using the BiConjugate Gradient Stabilized method.
        """

        A, M, x, b = self.make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('bicgstab', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return b, 0

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec = A.matvec
        psolve = M.matvec

        # These values make no sense but coming from original Fortran code
        # sqrt might have been meant instead.
        rhotol = np.finfo(x.dtype.char).eps**2
        omegatol = rhotol

        # Dummy values to initialize vars, silence linter warnings
        rho_prev, omega, alpha, p, v = None, None, None, None, None

        r = b - matvec(x) if x.any() else b.copy()
        rtilde = r.copy()

        for iteration in range(maxiter):
            actual_tol = np.linalg.norm(r)
            print("-----> iteration: ", iteration, ", with tol : ", actual_tol,  " vs ", atol)

            if actual_tol < atol:  # Are we done?
                return x, 0

            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                return x, -10

            if iteration > 0:
                if np.abs(omega) < omegatol:  # omega breakdown
                    return x, -11

                beta = (rho / rho_prev) * (alpha / omega)
                p -= omega*v
                p *= beta
                p += r
            else:  # First spin
                s = np.empty_like(r)
                p = r.copy()

            phat = psolve(p)
            v = matvec(phat)
            rv = dotprod(rtilde, v)
            if rv == 0:
                return x, -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]

            if np.linalg.norm(s) < atol:
                x += alpha*phat
                return x, 0

            shat = psolve(s)
            t = matvec(shat)
            omega = dotprod(t, s) / dotprod(t, t)
            x += alpha*phat
            x += omega*shat
            r -= omega*t
            rho_prev = rho

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return x, maxiter


    def plot_stats_optimization(self):
        iterations = np.arange(1, len(self._log_posteriors) + 1)

        fig, axs = plt.subplots(3, 3, figsize=(10, 8))
        axs = axs.ravel()

        axs[0].plot(iterations, self._log_posteriors)
        axs[0].set_title("Log likelihood variation")
        axs[0].set_xlabel("Iterations")

        axs[1].plot(iterations, self._jTDj)
        axs[1].set_title("jTDj term variation")
        axs[1].set_xlabel("Iterations")

        axs[2].plot(iterations, self._logdets_D)
        axs[2].set_title("Log|D| term variation")
        axs[2].set_xlabel("Iterations")

        axs[3].plot(iterations, self._logdets_S)
        axs[3].set_title("Log|S| variation")
        axs[3].set_xlabel("Iterations")

        axs[4].plot(iterations, self._logprior_1)
        axs[4].set_title("Log prior1 variation")
        axs[4].set_xlabel("Iterations")

        axs[5].plot(iterations, self._logprior_2)
        axs[5].set_title("Log prior2 variation")
        axs[5].set_xlabel("Iterations")

        axs[6].plot(iterations, self._ms, label="m")
        axs[6].set_title(f"m parameter ({self._m_bounds})")
        axs[6].set_xlabel("Iterations")

        axs[7].plot(iterations, self._cs, label="c")
        axs[7].set_title(f"log(c) parameter ({self._log_c_bounds})")
        axs[7].set_xlabel("Iterations")

        axs[8].plot(iterations, self._ls, label="l")
        axs[8].set_title(f"log(l) length scale ({self._log_l_bounds})")
        axs[8].set_xlabel("Iterations")

        fig.suptitle(f'Total time {self._optimization_time/60:2f} min, with best values (m, c) =  ({self._best[0]:2f}, {self._best[1]:2f})', fontsize=10)

        fig.tight_layout()
        plt.show()

    @property
    def mean(self):
        """Return the mean of the posterior."""
        return self._mu
