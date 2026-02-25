

class Params(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_param(self, key):
        return getattr(self, key)

class PowerSpectrum(object):
    """Class to compute and store power spectrum information."""

    def __init__(self, func = None, Param = None):
        """
        Initialize the PowerSpectrum with frequencies, power values, and parameters.

        Parameters
        ----------
        func : array-like
            Corresponding power values at the given frequencies.
        Param : Params
            An instance of Params class containing parameters.
        """
        if func is None:
            self._func = self.slope_func
        else:
            self._func = func

        if Param is None:
            self._params = Params(m=-3, c=1)
        else:
            if self.validate_Params(Param):
                self._params = Param
            else:
                raise TypeError("Param must be an instance of the Params class.")
    
    def value(self, freq, min_freq):
        """
        Compute the power spectrum function value at given frequencies.
        Parameters:
        freq : array or scalar
            Spatial frequency.
        min_freq : float
            Minimum frequency to avoid singularities.
        Returns:
        array or scalar
            Power spectrum evaluated at freq.
        """
        return self._func(freq, min_freq, self._params)
    
    def validate_Params(self, params):
        """
        Validate that the provided params is an instance of Params class.
        Parameters:
        params : object
            The parameters to validate.
        Returns:
        bool
            True if params is an instance of Params, False otherwise.
        """
        if isinstance(params, Params):
            return True
        return False
    
    def slope_func(self, q, min_q, params):
        """
        Power spectrum slope function.
        Parameters:
        ---------
        q: array or scalar, unit = lambda
            Spatial frequency.
        min_q: float
            Minimum frequency to avoid singularities.
        m: float
            Power-law index.
        c: float
            Amplitude of the power spectrum.

        Returns:
        ---------
        P(q): array or scalar
            Power spectrum evaluated at q.
        
        """
        m = Params.get_param('m')
        c = Params.get_param('c')

        if not np.isscalar(q):  
            q[q == 0] = min_freq
        elif q == 0:
            q = self._min_freq
        return c*(q**m)
    