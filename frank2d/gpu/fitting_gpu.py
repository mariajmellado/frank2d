import cupy as cp
import cupyx.scipy.sparse as cxs
import cupyx.scipy.linalg as spla
from cupyx.scipy.sparse.linalg import LinearOperator


import time
from tqdm import tqdm
from .utilities_gpu import DotLinearOperator
import pprint

import numpy as np


class IterativeSolverMethod():
    def __init__(self, u, v, vis, weights, kernel,
                 method_name = 'bicgstab', method_func = None,
                 rtol = 1e-8,  x0 = None, maxiter = None,
                 precond_type = 'jacobi',
                 verbose = True):
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
        self._preconditioner = precond_type

        self._solver_name = method_name
        self._solver_func = method_func
        self._x0 = x0
        self._rtol = rtol
        self._maxiter = maxiter

        self._A = None
        self._b = None
        self._A_precond = None

        self._sparse_system = None
        self._solution = None

        self._tol_fit_correctly = 1e-5

        self._fit_data = {}
        self._fit_info = {}

        self._verbose = verbose

    def get_solver(self):
        """
        Returns the iterative solver method based on the specified method name.
        """
        solver = self._solver_name
        if solver == 'bicgstab':
            return self.bicgstab
        else:
            raise ValueError(f"Solver method '{solver}' not recognized.")
    
    def set_solver(self, solver_func):
        """
        Sets the iterative solver method.
        """
        self._solver = solver_func
    
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

    def bicgstab(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, psolve = None, callback=None):
        """
        BiConjugate Gradient Stabilized Method (BiCGSTAB) solver (GPU, CuPy).
        Based on SciPy implementation but adapted for CuPy and with preconditioning via psolve function.
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
        psolve : function
            Preconditioner solve function, called as psolve(v).
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
        self._fit_info['maxiter'] = maxiter
        self._fit_info['rtol'] = rtol
        self._fit_info['final_tol'] = atol

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

        self._tols = []

        for iteration in tqdm(range(maxiter), desc="         + Fitting with BiCGStab...", unit=" iterations"):
            act_tol = float(cp.linalg.norm(r))
            self._tols.append(act_tol)
            if act_tol <= tol:
                self._fit_info['convergence_by'] = "norm of r"
                self._fit_info['iterations'] = iteration
                return x, 0

            rho = dotprod(rtilde, r)
            if float(cp.abs(rho)) < rhotol:  # rho breakdown
                self._fit_info['convergence_by'] = "norm of rho"
                self._fit_info['iterations'] = iteration
                return x, -10

            if iteration > 0:
                if float(cp.abs(omega)) < omegatol:  # omega breakdown
                    self._fit_info['convergence_by'] = "norm of omega"
                    self._fit_info['iterations'] = iteration
                    return x, -11
                beta = (rho / rho_prev) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p = r.copy()

            phat = psolve(p)
            v = matvec(phat)

            rv = dotprod(rtilde, v)
            if rv == 0:
                self._fit_info['convergence_by'] = "norm of rv"
                self._fit_info['iterations'] = iteration
                return x, -11
            alpha = rho / rv
            r = r - alpha * v
            s[:] = r

            if float(cp.linalg.norm(s)) <= tol:
                self._fit_info['convergence_by'] = "norm of s"
                self._fit_info['iterations'] = iteration
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

        self._fit_info['convergence_by'] = "maxiter"
        self._fit_info['iterations'] = iteration
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
        print(f'        +  Kernel = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        N = kernel_csr.shape[0]

        # Scaling with Hadamard product.
        A = kernel_csr.multiply(weights[:, None])
        A = A + cxs.eye(N, dtype=A.dtype, format="csr")

        A.sum_duplicates()
        A.sort_indices()

        preconditioner = self._preconditioner
        self._fit_info['preconditioner_method'] = preconditioner
        if preconditioner == "jacobi":
            # Preconditioner matrix M = diag(A)^{-1}.
            diagA = A.diagonal()
            M = cxs.csr_matrix((1.0/diagA, cp.arange(N), cp.arange(N+1)), shape=(N, N))
            M = DotLinearOperator(M, M.shape)

        elif preconditioner == "ilu":
            # Hibrid adaptation GPU -> CPU.
            import scipy.sparse.linalg as spla
            import scipy.sparse as sp

            print('        !  Transferring matrix to CPU for ILU factorization...')
            start_time = time.time()
    
            A_cpu = A.get()
            if not sp.isspmatrix_csc(A_cpu):
                A_cpu = A_cpu.tocsc()
            A_cpu = A_cpu.astype(np.complex128)
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'        +   GPU->CPU & CSC = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

            start_time = time.time()
            try:
                # 4. Calcular ILU en CPU usando SciPy
                ilu = spla.spilu(A_cpu, drop_tol=1e-4, fill_factor=10)
            except RuntimeError as e:
                print(f"ILU failed: {e}")
                ilu = spla.spilu(A_cpu + 1e-8 * sp.eye(A_cpu.shape[0]))
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'        +  ILU (on CPU) = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

            def ilu_solve_wrapper(x):
                x_cpu = cp.asnumpy(x)       # GPU -> CPU
                y_cpu = ilu.solve(x_cpu)    # Solve en CPU
                return cp.asarray(y_cpu)    # CPU -> GPU

            M = LinearOperator(A.shape, matvec=ilu_solve_wrapper)

        else:
            raise ValueError(f"Preconditioner method '{preconditioner}' not recognized.")
            
        b = cp.asarray(weights * vis)

        self.set_A(DotLinearOperator(A, A.shape))
        self.set_A_precond(M)
        self.set_b(b)
    
    def solve_linear_system(self):
        """
        Solve the linear system using the specified iterative solver method.
        """
        solver = self._solver
        psolve = lambda v: self._A_precond.matvec(v)

        if self._x0 is None:
            x, info = solver( self._A, self._b, M = self._A_precond,
                              rtol = self._rtol, maxiter = self._maxiter,
                              psolve = psolve
                            )
        else:
            self._fit_info['using_x0'] = True
            x, info = solver( self._A, self._b, M = self._A_precond,
                              x0 = self._x0,
                              rtol = self._rtol, maxiter = self._maxiter,
                              psolve = psolve
                            )

        self._solution_linear_system = x, info

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
        print("===>  Creating sparse linear system...")
        start_time = time.time()

        self.build_sparse_linear_system()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'        +  Building system = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Solve the linear system.
        print("===>  Solving linear system...")
        start_time = time.time()
        if self._solver_func is not None:
            self.set_solver(self._solver_func)
        else: 
            method = self.get_solver()
            self.set_solver(method)

        self.solve_linear_system()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'            +  BiCGStab = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        x, info = self._solution_linear_system

        # Report on the success of the fitting.
        fit_correctly = np.allclose( self._A.matvec(x),
                                     self._b,
                                     rtol=self._tol_fit_correctly
                                    )

        print("          + CGM converged?  ", info == 0)
        print("          + Fit correctly?  ", self.fit_correctly(fit_correctly))

        # save results
        self._fit_data['tols'] = self._tols
        self._fit_info['CGM_converged'] = (info == 0)
        self._fit_info['Fit_correctly'] = fit_correctly

        if self._verbose:
            pprint.pprint(self._fit_info)
        self._solution = x

        return x

    def fit_correctly(self, val):
        """
        Returns a string indicating whether the fitting was successful.
        Parameters
        ----------
        val : bool
            Indicates if the fitting was successful.
        Returns
        -------
        str
            A string indicating success or failure.
        """
        if str(val) == 'True':
            return " ✔✔✔✔✔✔✔ Sucess.."
        else:
            return " ✘✘✘✘✘✘ Failed.."

    @property
    def solution(self):
        """
        Returns the solution of the linear system.
        """
        return self._solution

    @property
    def solver(self):
        """
        Returns the iterative solver method.
        """
        return self.get_solver()
    
    @property
    def A(self):
        """
        Returns the matrix A of the linear system.
        """
        return self._A
    
    @property
    def b(self):
        """
        Returns the right-hand side vector b of the linear system.
        """
        return self._b
    
    @property
    def A_precond(self):
        """
        Returns the preconditioner matrix of the linear system.
        """
        return self._A_precond
    @property
    def fit_data(self):
        """
        Returns the data optimization dictionary.
        """
        return self._fit_data
  
    @property
    def info(self):
        """
        Returns the fitting information dictionary.
        """
        return self._fit_info
