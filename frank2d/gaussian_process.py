import numpy as np
from .utilities import linear_operator
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree


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
        self._q = self._q2 = np.hypot(self._u, self._v)
        self._min_freq = np.sort(np.abs(np.unique(self._v)))[1]
        self._size = self._size2 = len(u)

        if isinstance(u2, np.ndarray): # u2 is not None.
            self._u2 = u2
            self._v2 = v2
            self._q2 = np.hypot(self._u2, self._v2)
            self._size = len(u2)
            self._min_freq = min(self._min_freq, np.sort(np.abs(np.unique(self._v2)))[1])

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
        if not np.isscalar(q):  
            q[q == 0] = self._min_freq
        elif q == 0:
            q = self._min_freq
        return c*(q**m)
    
    def sparse(self):
        """
        Returns the Wendland covariance matrix as a sparse linear operator.
        """
        size = (self._size, self._size2)
        return  linear_operator(self.sparse_matrix(), size)

class SquaredExponential(CorrelationMatrix):
    def __init__(self, params, u, v, u2 = None, v2 = None):
        r"""
        Squared Exponential covariance matrix.
        The Squared Exponential covariance function, also known as the Gaussian covariance function,
        is defined as:
        C(r) = \sigma^2 * exp(-r^2 / (2*l^2))
        where:
        - r is the Euclidean distance between two points in space.
        - \sigma^2  is the variance (amplitude) of the process.
        - l is the correlation length scale, which determines how quickly the correlation decays with distance.
        This covariance function is infinitely differentiable, leading to very smooth realizations of the Gaussian Process.
        """

        super().__init__(params, u, v, u2, v2)
        self._m = params["m"]
        self._c = params["c"]
        self._l = params["l"]

        self._sparse_tol = 1e-30 # higher value means more sparse.
        self._r_cutoff = np.sqrt(-2.0 * np.log(self._sparse_tol))

        self._ul = self._u/self._l
        self._vl = self._v/self._l

        self._ul2 = self._u2/self._l
        self._vl2 = self._v2/self._l

        self._power_spectrum_q1 = self.power_spectrum(self._q, self._m, self._c)
                    

    def row(self, i):
        """
        Returns the i-th row of the covariance matrix.
        """
        amp = np.sqrt(self._power_spectrum_q1 * self.power_spectrum(self._q2[i], self._m, self._c))

        return amp * np.exp(-0.5 * ((self._ul - self._ul2[i]) ** 2 + (self._vl - self._vl2[i]) ** 2))
    
    def _row_sparse(self, i, u1_norm, v1_norm, q1):
        """
        Internal helper for sparse_matrix.
        Calculates a subset of row i based on provided normalized coords.
        """
        ps = self.power_spectrum(self._q2[i], self._m, self._c)
        amp = np.sqrt(q1 * ps)
        
        r_sq = (u1_norm - self._ul2[i])**2 + (v1_norm - self._vl2[i])**2
        
        return amp * np.exp(-0.5 * r_sq)

    
    def sparse_matrix(self):
        """
        Constructs the sparse covariance matrix using the Squared Exponential covariance function.
        
        Returns:
        csr_matrix: scipy.sparse.csr_matrix
            Sparse covariance matrix in Compressed Sparse Row format.
            
        Note:
        This method constructs an *approximate* sparse covariance matrix.
        It uses a KDTree to find neighbors within a cutoff radius, r_cutoff,
        and sets all kernel elements beyond this radius to zero.
        The cutoff radius is calculated from 'sparse_tol' in the params:
        r_cutoff = sqrt(-2 * log(sparse_tol))
        This radius is applied to the coordinates *normalized* by the length scale 'l'.
        """
        data = []
        indices = []
        indptr = [0]

        coords1 = np.array([self._ul, self._vl]).T
        coords2 = np.array([self._ul2, self._vl2]).T
        
        tree = KDTree(coords1)
        tree2 = KDTree(coords2)
        
        ngb = tree2.query_ball_tree(tree, self._r_cutoff)
        
        for i, ngb_i in enumerate(ngb):
            ngb_i = list(ngb_i)
            if not ngb_i:
                indptr.append(len(data))
                continue
            
            row_values = self._row_sparse(i, 
                                          u1_norm=self._ul[ngb_i], 
                                          v1_norm=self._vl[ngb_i], 
                                          q1=self._power_spectrum_q1[ngb_i])
            
            data.extend(row_values)
            indices.extend(ngb_i)
            indptr.append(len(data))
        
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        size = (self._size, self._size2)
        kernel_csr = csr_matrix((data, indices, indptr), shape=size)

        return kernel_csr


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
            return np.ones_like(r)  # P_0(r) = 1
        elif k == 1:
            return 4*r +1  # P_1(r) = 4r + 1
        elif k == 2:
            return (35/3)*r**2 +6*r + 1  # P_2(r) = (35/3)r^2 + 6r + 1
        else:
            raise ValueError("k must be 0, 1, or 2.")


    def row(self, i, u1=None, v1=None, q1=None):
        """
        Returns the i-th row of the covariance matrix.
        """
        if u1 is None:
            u1 = self._uh
            v1 = self._vh
            q1 = self._power_spectrum_q1

        ps = self.power_spectrum(self._q2[i], self._m, self._c)
        amp = np.sqrt(q1 * ps)

        r_normalized = np.sqrt((u1-self._uh2[i])**2 + (v1-self._vh2[i])**2)
        factor = (1 - r_normalized)**self._j
        factor[r_normalized > 1] = 0

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
        data = []
        indices = []
        indptr = [0]

        tree = KDTree(np.array([self._uh, self._vh]).T)
        tree2 = KDTree(np.array([self._uh2, self._vh2]).T)
        ngb = tree2.query_ball_tree(tree, 1.0)
        
        for i, ngb_i in enumerate(ngb):
            row = self.row(i, self._uh[ngb_i], self._vh[ngb_i], self._power_spectrum_q1[ngb_i])
            data.extend(row)
            indices.extend(ngb_i)
            indptr.append(len(data))
        
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        size = (self._size, self._size2)
        kernel_csr = csr_matrix((data, indices, indptr), shape=size)

        return kernel_csr




