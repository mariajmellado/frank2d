from cupyx.scipy.sparse.linalg import LinearOperator

"""
This module provides utility functions for Frank2D package (CuPy backend).
"""

def linear_operator(matrix, size):
    """
    Wrapper function for creating a CuPy LinearOperator.

    Parameters
    ----------
    matrix : cupyx.scipy.sparse.spmatrix
        Sparse matrix (e.g., CSR) living on GPU.
    size : tuple[int, int]
        Shape (m, n) of the operator.

    Returns
    -------
    LinearOperator
        A GPU linear operator with matvec (and rmatvec) defined.
    """
    def matvec(x):
        return matrix.dot(x)

    return LinearOperator(size, matvec=matvec)
