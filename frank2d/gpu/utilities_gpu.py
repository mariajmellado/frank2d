from cupyx.scipy.sparse.linalg import LinearOperator
from ..constants import rad_to_arcsec
import cupy as cp

"""
This module provides utility functions for Frank2D package (CuPy backend).
"""

def get_optimal_N(R_max_arcsec, Q_max_lambda, eta=5):
    """
    Calculate the minimum number of collocation points (N) for the Frank2D
    algorithm, based on the maximum radius (R_max) and the longest observed
    baseline (Q_max).

    The sampling factor eta sets how far the Fourier grid extends past Q_max;
    the default of 5 places its outer edge roughly 25% beyond the longest
    observed baseline.

    Parameters
    ----------
    R_max_arcsec : float
        Maximum radius of the image grid, in arcseconds.
    Q_max_lambda : float
        Longest observed baseline, in wavelengths.
    eta : float, optional
        Sampling factor (default is 5).
    """
    N_float = eta * Q_max_lambda * (R_max_arcsec / rad_to_arcsec)
    return int(cp.floor(N_float))


class DotLinearOperator(LinearOperator):
    def __init__(self, matrix, shape):
        """
        Initializes the operator.
        
        Parameters
        ----------
        matrix : cp.ndarray
            The dense 2D array to be used for matrix-vector products.
        """
        self.matrix = matrix
        
        super().__init__(shape=shape, dtype=matrix.dtype)

    def _matvec(self, x):
        """Defines the matrix-vector product (A * x)"""
        return self.matrix.dot(x)