import cupy as cp
import cupyx.scipy.sparse as cxs
import time

from .utilities_gpu import linear_operator

import numpy as np


class IterativeSolverMethod():
    def __init__(self, u, v, vis, weights, kernel, method = 'bicgstab', rtol = 1e-7,  x0 = None, maxiter = None):
        """
         Class to handle the iterative solver methods.

        Parameters
        ----------
        u : 1D array, unit = lambda
            The u coordinates of the visibilities.
        v : 1D array, unit = lambda
            The v coordinates of the visibilities.
        vis : 1D array, unit = Jy
            The visibilities.
        weights : 1D array, unit = 1/Jy^2
            The weights of the visibilities.
        kernel_row : function
            Function that returns the kernel row for a given index.
        method : str, optional
            The iterative solver method to use. Options are 'cg', 'bicg', 'bicgstab', 'cgs'. Default is 'bicgstab'.
        rtol : float, optional
            The relative tolerance for the solver. Default is 1e-7.
        x0 : 1D array, optional, unit = Jy
            Initial guess for the visibility solution. Default is None.
        """
        self._u = u
        self._v = v
        self._vis = vis
        self._weights = weights
        self._kernel = kernel
        
        self._x0 = x0
        self._rtol = rtol
        self._maxiter = maxiter
        self._method = method

        self._A = None
        self._b = None
        self._A_precond = None

        self._sparse_system = None
        self._solution = None
    
    
    def set_A(self, A):
        """
        Sets the matrix A for the linear system.
        """
        self._A = A
    
    def set_b(self, b):
        """
        Sets the right-hand side vector b for the linear system.
        """
        self._b = b

    def set_A_precond(self, A_precond):
        """
        Sets the preconditioner matrix for the linear system.
        """
        self._A_precond = A_precond

    def get_method(self):
        """
        Returns the iterative solver method based on the specified method name.
        """
        method = self._method
        if method == 'bicgstab':
            return self.bicgstab

    def bicgstab(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, psolve = None, callback=None):
        """
        BiConjugate Gradient Stabilized Method (BiCGSTAB) solver (GPU, CuPy).

        Parameters
        ----------
        A : LinearOperator
            The real or complex N-by-N linear operator with .matvec.
        b : array-like
            Right-hand side of the linear system. Shape (N,) or (N,1).
        x0 : array-like, optional
            Starting guess for the solution. If None, zeros are used.
        rtol : float, optional
            Relative tolerance for convergence. Default is 1e-7.
        atol : float, optional
            Absolute tolerance for convergence. Default is 0.
        maxiter : int, optional
            Maximum number of iterations. Default is N*10.
        M : LinearOperator, optional
            Preconditioner for A, applied as v -> M.matvec(v). If None, identity.
        callback : function, optional
            Called as callback(xk) after each iteration.

        Returns
        -------
        x : cp.ndarray
            The converged solution on GPU.
        info : int
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : breakdown (-10 rho, -11 omega/tt)
        """
        n = b.size
        x = cp.zeros_like(b) if x0 is None else cp.asarray(x0, dtype=b.dtype).ravel()

        bnrm2 = cp.linalg.norm(b)
        tol = max(float(atol), float(rtol) * float(bnrm2))
        
        print("Final tolerance : ", tol)
        if bnrm2 == 0.0:
            return b, 0

        if maxiter is None:
            maxiter = n * 10

        matvec = A.matvec
        psolve = psolve

        dotprod = cp.vdot if cp.iscomplexobj(x) else cp.dot
        eps = float(cp.finfo(x.dtype).eps)
        rhotol = eps ** 2
        omegatol = rhotol

        r = b - matvec(x) if bool(cp.any(x)) else b.copy()
        rtilde = r.copy()

        p = None
        v = None
        rho_prev = None
        alpha = None
        omega = None

        s = cp.empty_like(r)

        for iteration in range(maxiter):
            act_tol = float(cp.linalg.norm(r))
            print(".... iteration: ", iteration)
            print("                        -> actual tol: ", str(act_tol), "vs ", str(tol))
            if act_tol <= tol:
                return x, 0

            rho = dotprod(rtilde, r)
            if float(cp.abs(rho)) < rhotol:  # rho breakdown
                return x, -10

            if iteration > 0:
                if float(cp.abs(omega)) < omegatol:  # omega breakdown
                    return x, -11
                beta = (rho / rho_prev) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p = r.copy()

            phat = psolve(p)
            v = matvec(phat)

            rv = dotprod(rtilde, v)
            if rv == 0:
                return x, -11
            alpha = rho / rv
            r = r - alpha * v
            s[:] = r

            if float(cp.linalg.norm(s)) <= tol:
                x = x + alpha * phat
                return x, 0

            shat = psolve(s)
            t = matvec(shat)

            tt = dotprod(t, t)
            if tt == 0:
                return x, -11
            omega = dotprod(t, s) / tt

            x = x + alpha * phat + omega * shat
            r = s - omega * t
            rho_prev = rho

            if callback is not None:
                callback(x)

        return x, maxiter

    def build_sparse_linear_system(self):
        """
        Create linear system in sparse approach, using sparse matrix storage and linear operators.
        The system to build is:
                Ax = b
        Where:
        A is I + N^{-1} S_{data}.
        b is (N^{-1} V_{data}).
        """

        weights = self._weights
        vis = self._vis
        
        start_time = time.time()
        kernel_csr = self._kernel.sparse_matrix()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'--> time kernel = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        N = kernel_csr.shape[0]

        # Scaling with Hadamard product.
        A = kernel_csr.multiply(weights[:, None])
        A = A + cxs.eye(N, dtype=A.dtype, format="csr")

        A.sum_duplicates()
        A.sort_indices()

        # Preconditioner matrix M = diag(A)^{-1}.
        diagA = A.diagonal()
        M = 1.0/diagA
        #M = cxs.csr_matrix((1.0/diagA, cp.arange(N), cp.arange(N+1)), shape=(N, N))

        b = cp.asarray(weights * vis)
        
        self.set_A(linear_operator(A, A.shape))
        self.set_A_precond(M)
        self.set_b(b)
    
    def solve_linear_system(self, solver):
        """
        Solve the linear system using the specified iterative solver method.
        Parameters
        ----------
        solver : function
            The iterative solver method to use.
        """
        psolve=lambda v: v * self._A_precond

        if self._x0 is None:
            x, info = solver( self._A, self._b, M = self._A_precond,
                              rtol = self._rtol, maxiter = self._maxiter,
                              psolve = psolve
                            )
        else:
            x, info = solver( self._A, self._b, M = self._A_precond,
                              x0 = self._x0,
                              rtol = self._rtol, maxiter = self._maxiter,
                              psolve = psolve
                            )

        self._res_linear_system = x, info

    def run(self):
        """
        Solves the linear system to find V* using the specified iterative solver method.
        The system to solve:

            Ax = b <=> (I + N^{-1} S_{data})  V* = N^{-1} V_{data}

        Where:
        A : LinearOperator
            Right-hand side matrix (I + N^{-1} S_{data}).
            Where:
                I : Identity matrix.
                N : Noise covariance matrix (diagonal with weights). Unit = 1/Jy^2.
                S_{data} : Correlation matrix of the data (from the kernel).
        b : 1D array
            Right-hand side vector (N^{-1} V_{data}). 
            Where:
                V_{data} : Measured visibilities. Unit = Jy.
                N : Noise covariance matrix (diagonal with weights). Unit = 1/Jy^2.
        x : 1D array
            The solution vector (V*). Unit = Jy.
        """
        # Create the linear system.
        print("Creating sparse linear system...")
        start_time = time.time()

        self.build_sparse_linear_system()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'--> times building system = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Solve the linear system.
        print("Solving linear system...")
        start_time = time.time()

        solver = self.get_method()
        self.solve_linear_system(solver = solver)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'--> times solving system = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        x, info = self._res_linear_system
        # Report on the success of the fitting.
        fit_correctly = cp.allclose(self._A.matvec(x), self._b)
        print("               ---> CGM converged?  ", info == 0)
        print("                    ---> Fit correctly?  ", bool(fit_correctly))
        if fit_correctly:
            print("                     !!!!  Sucess..  !!!!")

        self._solution = x

        return x

    @property
    def sol(self):
        """
        Returns the solution of the linear system.
        """
        return self._solution