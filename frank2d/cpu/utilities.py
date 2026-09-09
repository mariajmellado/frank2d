from scipy.sparse.linalg import LinearOperator
from ..constants import rad_to_arcsec

import numpy as np

"""
This module provides utility functions for Frank2D package.
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
    return int(np.floor(N_float))

def next_fft_friendly(n, factors=(2, 3, 5)):
    """
    Smallest integer >= n whose prime factorisation contains only the given
    factors (5-smooth by default). These are the sizes for which the FFT can
    use its specialised radix routines all the way down.

    Parameters
    ----------
    n : int
        Lower bound on the grid size.
    factors : tuple of int, optional
        Allowed prime factors (default is (2, 3, 5)).
    """
    if n <= 1:
        return 1
    limit = n * max(factors)
    best = None
    a = 1
    while a < limit:
        b = a
        while b < limit:
            c = b
            while c < limit:
                if c >= n and (best is None or c < best):
                    best = c
                c *= factors[2]
            b *= factors[1]
        a *= factors[0]
    return best


def get_optimal_N_fft(R_max_arcsec, Q_max_lambda, eta=5, factors=(2, 3, 5)):
    """
    Same lower bound as get_optimal_N, then rounded up to the nearest
    FFT-friendly size. The cost of the FFT is governed by the prime
    factorisation of N rather than by N alone, so a slightly larger grid built
    from small factors is usually cheaper than the exact minimum.

    Parameters
    ----------
    R_max_arcsec : float
        Maximum radius of the image grid, in arcseconds.
    Q_max_lambda : float
        Longest observed baseline, in wavelengths.
    eta : float, optional
        Sampling factor (default is 5).
    factors : tuple of int, optional
        Allowed prime factors (default is (2, 3, 5)).
    """
    N_min = get_optimal_N(R_max_arcsec, Q_max_lambda, eta)
    return next_fft_friendly(N_min, factors)

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