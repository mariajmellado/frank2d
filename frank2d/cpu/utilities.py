from scipy.sparse.linalg import LinearOperator
import logging

"""
This module provides utility functions for Frank2D package.
"""

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