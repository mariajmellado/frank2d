from scipy.sparse.linalg import LinearOperator

"""
This module provides utility functions for Frank2D package.
"""

def linear_operator(matrix, size):
    """
    Wrapper function for creating linear operator.
    """
    def dot_product(x):
        return matrix.dot(x)

    return LinearOperator(size, matvec=dot_product)