import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as cxs
from scipy.sparse.linalg._isolve.utils import make_system
import matplotlib.pyplot as plt
import time
import pprint
from tqdm import tqdm

from .utilities import DotLinearOperator

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
        if solver == 'cg':
            return self.cg
        elif solver == 'bicg':
            return self.bicg
        elif solver == 'bicgstab':
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

    def cg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol = max(float(atol), float(rtol) * float(bnrm2))

        if bnrm2 == 0:
            return b, 0

        n = len(b)

        if maxiter is None:
            maxiter = n*10

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        matvec = A.matvec
        psolve = M.matvec
        r = b - matvec(x) if x.any() else b.copy()

        # Dummy value to initialize var, silences warnings
        rho_prev, p = None, None

        for iteration in range(maxiter):
            if np.linalg.norm(r) < atol:  # Are we done?
                return x, 0

            z = psolve(r)
            rho_cur = dotprod(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
            else:  # First spin
                p = np.empty_like(r)
                p[:] = z[:]

            q = matvec(p)
            alpha = rho_cur / dotprod(p, q)
            x += alpha*p
            r -= alpha*q
            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return x, maxiter

    def bicg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol = max(float(atol), float(rtol) * float(bnrm2))

        if bnrm2 == 0:
            return b, 0

        n = len(b)
        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec, rmatvec = A.matvec, A.rmatvec
        psolve, rpsolve = M.matvec, M.rmatvec

        rhotol = np.finfo(x.dtype.char).eps**2

        # Dummy values to initialize vars, silence linter warnings
        rho_prev, p, ptilde = None, None, None

        r = b - matvec(x) if x.any() else b.copy()
        rtilde = r.copy()

        for iteration in range(maxiter):
            if np.linalg.norm(r) < atol:  # Are we done?
                return x, 0

            z = psolve(r)
            ztilde = rpsolve(rtilde)
            # order matters in this dot product
            rho_cur = dotprod(rtilde, z)

            if np.abs(rho_cur) < rhotol:  # Breakdown case
                return x, -10

            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z
                ptilde *= beta.conj()
                ptilde += ztilde
            else:  # First spin
                p = z.copy()
                ptilde = ztilde.copy()

            q = matvec(p)
            qtilde = rmatvec(ptilde)
            rv = dotprod(ptilde, q)

            if rv == 0:
                return x, -11

            alpha = rho_cur / rv
            x += alpha*p
            r -= alpha*q
            rtilde -= alpha.conj()*qtilde
            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return x, maxiter

    def bicgstab(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        """
        BiConjugate Gradient Stabilized Method (BiCGSTAB) solver from SciPy.

        Parameters
        ----------
        A : {sparse matrix, dense matrix, LinearOperator}
            The real or complex N-by-N matrix of the linear system.
        b : {array, matrix}
            Right-hand side of the linear system. Has shape (N,) or (N,1).
        x0 : {array, matrix}, optional
            Starting guess for the solution. If None, zeros are used.
        rtol : float, optional
            Relative tolerance for convergence. Default is 1e-7.
        atol : float, optional
            Absolute tolerance for convergence. Default is 0.
        maxiter : int, optional
            Maximum number of iterations. Default is N*10.
        M : {sparse matrix, dense matrix, LinearOperator}, optional
            Preconditioner for A. The preconditioner should approximate the inverse of A.
        callback : function, optional
            User-supplied function to call after each iteration. It is called as callback(xk),
            where xk is the current solution vector.
        Returns
        -------
        x : array
            The converged solution.
        info : int
            Provides convergence information:
                0  : successful exit
                >0 : convergence to tolerance not achieved, number of iterations
                <0 : illegal input or breakdown
                    -10 : rho breakdown
                    -11 : omega breakdown
        Notes
        -----
        The BiCGSTAB method is an iterative method for solving large, sparse, non-symmetric linear systems.
        It is based on the BiConjugate Gradient method but includes a stabilization step to improve convergence.
        The method is particularly effective for large, sparse systems where direct methods are impractical.
        References
        ----------
        [1] H. A. van der Vorst, "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems," SIAM J. Sci. Stat. Comput., vol. 13, no. 2, pp. 631-644, Mar. 1992. doi: 10.1137/0913035.
        [2] Y. Saad, "Iterative Methods for Sparse Linear Systems," 2nd ed., SIAM, 2003.

        See the SciPy documentation for more details:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html
        """
        
        A, M, x, b = make_system(A, M, x0, b)

        bnrm2 = np.linalg.norm(b)
        atol = max(float(atol), float(rtol) * float(bnrm2))
        self._fit_info['maxiter'] = maxiter
        self._fit_info['rtol'] = rtol
        self._fit_info['final_tol'] = atol
    
        if bnrm2 == 0:
            return b, 0
    
        n = len(b)
    
        dotprod = np.vdot if np.iscomplexobj(x) else np.dot
    
        matvec = A.matvec
        psolve = M.matvec
    
        # These values make no sense but coming from original Fortran code
        # sqrt might have been meant instead.
        rhotol = np.finfo(x.dtype.char).eps**2
        omegatol = rhotol
    
        # Dummy values to initialize vars, silence linter warnings
        rho_prev, omega, alpha, p, v = None, None, None, None, None
    
        r = b - matvec(x) if x.any() else b.copy()
        rtilde = r.copy()

        self._tols = []
    
        for iteration in tqdm(range(maxiter), desc="         + Fitting with BiCGStab...", unit=" iterations"):
            time.sleep(0.05)
            act_tol = np.linalg.norm(r)
            self._tols.append(act_tol)
            #if iteration % 100 == 0:
            #    print("                 ",
            #        " actual tol ", f'{act_tol:.2e}', " versus ", f'{atol:.2e}')
            if act_tol < atol:
                self._fit_info['convergence_by'] = "norm of r"
                self._fit_info['iterations'] = iteration
                return x, 0
    
            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                self._fit_info['convergence_by'] = "norm of rho"
                self._fit_info['iterations'] = iteration
                return x, -10

            beta = 0
            if iteration > 0:
                if np.abs(omega) < omegatol:  # omega breakdown
                    self._fit_info['convergence_by'] = "norm of omega"
                    self._fit_info['iterations'] = iteration
                    return x, -11
    
                beta = (rho / rho_prev) * (alpha / omega)
                p -= omega*v
                p *= beta
                p += r
            else:  # First spin
                s = np.empty_like(r)
                p = r.copy()
    
            phat = psolve(p)
            v = matvec(phat)
            rv = dotprod(rtilde, v)
            if rv == 0:
                self._fit_info['convergence_by'] = "norm of rv"
                self._fit_info['iterations'] = iteration
                return x, -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]
    
            if np.linalg.norm(s) < atol:
                self._fit_info['convergence_by'] = "norm of s"
                self._fit_info['iterations'] = iteration
                x += alpha*phat
                return x, 0
    
            shat = psolve(s)
            t = matvec(shat)
            omega = dotprod(t, s) / dotprod(t, t)
            x += alpha*phat
            x += omega*shat
            r -= omega*t
            rho_prev = rho
    
            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
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
            M = cxs.csr_matrix((1.0/diagA, np.arange(N), np.arange(N+1)), shape=(N, N))
            M = DotLinearOperator(M, M.shape)

        elif preconditioner == "ilu":
            import scipy.sparse.linalg as spla
            import scipy.sparse as sp

            start_time = time.time()
            if not sp.isspmatrix_csc(A):
                A = A.tocsc()
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'        +   CSC = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

             # ILU requires complex128 dtype.
            A = A.astype(np.complex128)

            start_time = time.time()
            try:
                ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=10)
            except RuntimeError as e:
                print(f"ILU failed: {e}")
                ilu = spla.spilu(A + 1e-8 * sp.eye(A.shape[0]))
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'        +  ILU = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

             # Preconditioner as a linear operator.
            M = LinearOperator(A.shape, matvec=ilu.solve)
        else:
            raise ValueError(f"Preconditioner '{preconditioner}' not recognized.")

        b = weights * vis

        self.set_A(DotLinearOperator(A, A.shape))
        self.set_A_precond(M)
        self.set_b(b)
    
    def solve_linear_system(self):
        """
        Solve the linear system using the specified iterative solver method.
        """
        solver = self._solver
        if self._x0 is None:
            x, info = solver( self._A, self._b, M = self._A_precond,
                              rtol = self._rtol, maxiter = self._maxiter
                            )
        else:
            self._fit_info['using_x0'] = True
            x, info = solver( self._A, self._b, M = self._A_precond,
                              x0 = self._x0,
                              rtol = self._rtol, maxiter = self._maxiter
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
        fit_correctly_bool = self.fit_correctly(fit_correctly)

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
