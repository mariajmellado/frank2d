from cupyx.scipy.sparse.linalg import LinearOperator

"""
This module provides utility functions for Frank2D package (CuPy backend).
"""

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