from scipy.sparse.linalg import LinearOperator

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