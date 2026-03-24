"""
This module contains helper functions for the Frank2D algorithm.
"""
from .constants import rad_to_arcsec
import numpy as np

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