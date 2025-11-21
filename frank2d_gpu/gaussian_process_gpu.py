import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from .utilities_gpu import DotLinearOperator
from scipy.spatial import KDTree

import numpy as np
import time

"""
This module contains classes for constructing Kernel for the Gaussian Process
in Frank's 2D algorithm.
"""

class CorrelationMatrix():
    def __init__(self, params, u, v, u2 = None, v2 = None):
        """
        Covariance matrix class for the Gaussian Process.
        Parameters:
        params: dictionary
            Parameters of the covariance function.
            Must contain:
                m: float
                    Power-law index.
                c: float
                    Amplitude of the power spectrum.
                l: float, unit = lambda
                    Correlation length scale.
        u: array, unit = lambda
            u coordinates of the spatial frequencies.
        v: array, unit = lambda
            v coordinates of the spatial frequencies.
        u2: array, unit = lambda, optional
            u coordinates of the second set of spatial frequencies to correlate.
            If None, u2 = u.
        v2: array, unit = lambda, optional
            v coordinates of the second set of spatial frequencies to correlate.
            If None, v2 = v.
        """
        self._params = params
        self._u = self._u2 = u
        self._v = self._v2 = v
        self._q = self._q2 = cp.hypot(self._u, self._v)
        self._min_freq = cp.sort(cp.abs(cp.unique(self._v)))[1]
        self._size = self._size2 = len(u)

        if isinstance(u2, cp.ndarray): # u2 is not None.
            self._u2 = u2
            self._v2 = v2
            self._q2 = cp.hypot(self._u2, self._v2)
            self._size = len(u2)
            self._min_freq = cp.minimum(self._min_freq, cp.sort(cp.abs(cp.unique(self._v2)))[1])

    def power_spectrum(self, q, m, c):
        """
        Power spectrum function.
        Parameters:
        ---------
        q: array or scalar, unit = lambda
            Spatial frequency.
        m: float
            Power-law index.
        c: float
            Amplitude of the power spectrum.

        Returns:
        ---------
        P(q): array or scalar
            Power spectrum evaluated at q.
        
        """
        min_freq = cp.asarray(self._min_freq)
        
        q_copy = q.copy() # Evitar modificar el input q
        q_copy = cp.where(q_copy == 0, min_freq, q_copy)
        
        return c * (q_copy**m)
    
    def sparse(self):
        """
        Returns the Wendland covariance matrix as a sparse linear operator.
        """
        size = (self._size, self._size2)
        return  DotLinearOperator(self.sparse_matrix(), size)

class Wendland(CorrelationMatrix):
    def __init__(self, params, u, v, u2 = None, v2 = None):
        """
        Wendland covariance matrix.
        Wendland functions are compactly supported radial basis functions that are positive definite
        in certain dimensions. They are useful for constructing covariance functions that have a finite range,
        making them suitable for modeling spatial data with a limited correlation length.
        The Wendland covariance function used here is defined as:
        C(r) = (1 - r)^j * P_k(r) for r < 1
            0               for r >= 1
        where:
        - r is the normalized distance (r = d/l, where d is the Euclidean distance and l is the correlation length scale).
        - j is a smoothness parameter that determines the differentiability of the function.
        - P_k(r) is a polynomial of degree k that ensures positive definiteness.
        The choice of j and k depends on the desired smoothness and the dimensionality of the space.
        """
        super().__init__(params, u, v, u2, v2)
              
        self._m = params["m"]
        self._c = params["c"]
        self._l = params["l"]

        self._j, self._k = 4, 1
        self._H = 2*1.897367*self._l

        self._uh = self._u/self._H
        self._vh = self._v/self._H

        self._power_spectrum_q1 = self.power_spectrum(self._q, self._m, self._c)

        self._uh2 = self._u2/self._H
        self._vh2 = self._v2/self._H


    def P_k(self, r, k):
        """
        Polynomial P_k(r) for Wendland covariance function.

        Parameters:
        -----------
        r: array or scalar
            Normalized distance.
        k: int
            Degree of the polynomial. Supported values are 0, 1, and 2.
        """
        if k == 0:
            return cp.ones_like(r)  # P_0(r) = 1
        elif k == 1:
            return 4*r +1  # P_1(r) = 4r + 1
        elif k == 2:
            return (35/3)*r**2 +6*r + 1  # P_2(r) = (35/3)r^2 + 6r + 1
        else:
            raise ValueError("k must be 0, 1, or 2.")

    def row_vectorized(self, u1, v1, q1, u2_val, v2_val, ps_val):
        amp = cp.sqrt(q1 * ps_val)
        r_normalized = cp.sqrt((u1 - u2_val)**2 + (v1 - v2_val)**2)
        factor = (1 - r_normalized)**self._j
        factor = cp.where(r_normalized > 1, 0, factor) # Reemplazo de factor[r_normalized > 1] = 0
        
        return amp * factor * self.P_k(r_normalized, self._k)

    def sparse_matrix(self):
        """
        Constructs the sparse covariance matrix using the Wendland covariance function.
        Returns:
        csr_matrix: scipy.sparse.csr_matrix
            Sparse covariance matrix in Compressed Sparse Row format.
            
        Note:
        This method constructs the sparse covariance matrix by iterating over each row,
        calculating the non-zero entries based on the Wendland covariance function,
        and storing them in CSR format. The sparsity is determined by the compact support
        of the Wendland function, which is zero beyond a certain distance.
        KDTree is used to efficiently find neighboring points within the support radius.
        """
        uh_cpu = self._uh.get()
        vh_cpu = self._vh.get()
        uh2_cpu = self._uh2.get()
        vh2_cpu = self._vh2.get()
        
        tree = KDTree(np.array([uh_cpu, vh_cpu]).T)
        tree2 = KDTree(np.array([uh2_cpu, vh2_cpu]).T)
        ngb = tree2.query_ball_tree(tree, 1.0)

        cpu_indices = []
        cpu_indptr = [0]
        cpu_row_indices = []
        
        for i, ngb_i in enumerate(ngb):
            n_entries = len(ngb_i)
            cpu_indices.extend(ngb_i)
            cpu_indptr.append(cpu_indptr[-1] + n_entries)
            cpu_row_indices.extend([i] * n_entries)

        if not cpu_indices:
             return csr_matrix((self._size, self._size2))

        col_indices_gpu = cp.asarray(cpu_indices)
        row_indices_gpu = cp.asarray(cpu_row_indices) 
        
        u1_gpu = self._uh[col_indices_gpu]
        v1_gpu = self._vh[col_indices_gpu]
        q1_gpu = self._power_spectrum_q1[col_indices_gpu]

        u2_val_gpu = self._uh2[row_indices_gpu]
        v2_val_gpu = self._vh2[row_indices_gpu]
        
        q2_gpu = cp.asarray(self._q2)[row_indices_gpu]
        ps_val_gpu = self.power_spectrum(q2_gpu, self._m, self._c)

        data_gpu = self.row_vectorized( u1_gpu, v1_gpu, q1_gpu, 
                                        u2_val_gpu, v2_val_gpu, 
                                        ps_val_gpu
                                    )
        
        indptr_gpu = cp.asarray(cpu_indptr)

        size = (self._size, self._size2)
        kernel_csr = cp.sparse.csr_matrix((data_gpu, col_indices_gpu, indptr_gpu), shape=size)

        return kernel_csr
