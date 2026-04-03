from scipy.sparse.linalg import LinearOperator
from ..constants import rad_to_arcsec

import numpy as np

"""
This module provides utility functions for Frank2D package.
"""

def get_optimal_N(R_max_arcsec, Q_max_lambda, padding=5):
    """
    Calculate the optimal number of pixels (N) for the Frank2D algorithm
    based on the maximum radius (R_max) and maximum spatial frequency (Q_max).

    N must be large enough to capture the spatial frequencies up to Q_max, 
    and even to ensure symmetry in the Fourier transform (including the zero frequency).
    Parameters
    ----------
    R_max_arcsec : float
        Maximum radius in arcseconds.
    Q_max_lambda : float
        Maximum spatial frequency in units of lambda/D.
    padding : int, optional
        Additional padding factor to ensure sufficient sampling (default is 5).
    """

    N_float = padding * Q_max_lambda * (R_max_arcsec / rad_to_arcsec)
    
    N_int = int(np.floor(N_float))
    
    # Ensure N is even for symmetry.
    if N_int % 2 != 0:
        N_int += 1
        
    return N_int

def linear_operator(matrix, size):
    """
    Function to create a linear operator from a given matrix.
    Parameters
    ----------
    matrix : 2D array
        The matrix to be converted into a linear operator.
    size : tuple[int, int]
        The shape of the linear operator (rows, columns).
    Returns
    -------
    LinearOperator
        A linear operator that performs matrix-vector multiplication.
    """
    def matvec(x):
        return matrix.dot(x)

    return LinearOperator(size, matvec=matvec)

class DotLinearOperator(LinearOperator):
    def __init__(self, matrix, shape):
        """
        Initializes the operator.
        
        Parameters
        ----------
        matrix : np.ndarray
            The dense 2D array to be used for matrix-vector products.
        """
        self.matrix = matrix
        
        super().__init__(shape=shape, dtype=matrix.dtype)

    def _matvec(self, x):
        """Defines the matrix-vector product (A * x)"""
        return self.matrix.dot(x)