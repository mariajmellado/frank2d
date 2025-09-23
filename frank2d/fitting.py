import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg._isolve.utils import make_system
import matplotlib.pyplot as plt
import time

from .utilities import linear_operator

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
        
        self._method = method
        self._x0 = x0
        self._rtol = rtol
        self._maxiter = maxiter

        self._A = None
        self._b = None
        self._A_precond = None

        self._sparse_system = None
        self._solution = None
    
    def get_method(self):
        """
        Returns the iterative solver method based on the specified method name.
        """
        method = self._method
        if method == 'cg':
            return self.cg
        elif method == 'bicg':
            return self.bicg
        elif method == 'bicgstab':
            return self.bicgstab
        elif method == 'cgs':
            return self.cgs
    
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

    def _get_atol_rtol(self, name, b_norm, atol=0., rtol=1e-5):
        """
        Scipy's helper function to handle tolerance normalization.
        See in https://github.com/scipy/scipy/blob/v1.16.2/scipy/sparse/linalg/_isolve/iterative.py#L158-L304.
        
        Parameters
        ----------
        name : str
            Name of the solver method.
        b_norm : float
            Norm of the right-hand side vector b.
        atol : float, optional
            Absolute tolerance. Default is 0.
        rtol : float, optional
            Relative tolerance. Default is 1e-5.
        """
        if atol == 'legacy' or atol is None or atol < 0:
            msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
                "if set, `atol` must be a real, non-negative number.")
            raise ValueError(msg)

        print(f'         * rtol: {float(rtol)}')

        atol = max(float(atol), float(rtol) * float(b_norm))
        print(f'         * final tolerance: {atol}')

        return atol, rtol

    def cg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)

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

        atol, _ = _get_atol_rtol('bicg', bnrm2, atol, rtol)

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
    
        atol, _ = self._get_atol_rtol('bicgstab', bnrm2, atol, rtol)
    
        print("Final tolerance : ", atol)
    
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

        tols = []
    
        for iteration in range(maxiter):
            print(".... iteration: ", iteration)
            act_tol = np.linalg.norm(r)
            tols.append(act_tol)
            print("                        -> actual tol: ", str(act_tol), "vs ", str(atol))
            if act_tol < atol:  # Are we done?
                print(" --------------------------------------> CGM converged in ", iteration, " iterations with tol ", act_tol)

                self.plot_tolerance(iteration, tols)
                
                return x, 0
    
            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                print("converged by norm of rho")
                self.plot_tolerance(iteration, tols)
                return x, -10
    
            if iteration > 0:
                if np.abs(omega) < omegatol:  # omega breakdown
                    print("converged by norm of omega")
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
                print("converged by rv = 0")
                self.plot_tolerance(iteration, tols)
                return x, -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]
    
            if np.linalg.norm(s) < atol:
                print("converged by norm of s")
                x += alpha*phat
                self.plot_tolerance(iteration, tols)
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

            self.plot_tolerance(maxiter-1, tols)
            
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
        N = self._vis.shape[0]

        data_A = []
        indices_A = []
        indptr_A = [0]

        data_A_precond = []
        indices_A_precond = []
        indptr_A_precond = [0]

        data_b = []
        indices_b = []
        indptr_b = [0]

        weights = self._weights
        vis = self._vis

        for i in range(N):  
            row = self._kernel.row(i)

            non_zero_indices = np.nonzero(row)[0]
            non_zero_values = row[non_zero_indices]
            A_values = non_zero_values * weights[i]

            # A
            diagonal_pos = np.where((non_zero_indices == i))[0][0]
            A_values[diagonal_pos] += 1
            diag_value = A_values[diagonal_pos]
            
            data_A.extend(A_values)
            indices_A.extend(non_zero_indices)
            indptr_A.append(len(data_A))

            # Preconditioner of A
            data_A_precond.extend([diag_value**(-1)])
            indices_A_precond.extend([i])
            indptr_A_precond.append(len(data_A_precond))
            
        data_A = np.array(data_A)
        indices_A = np.array(indices_A)
        indptr_A = np.array(indptr_A)

        data_A_precond = np.array(data_A_precond)
        indices_A_precond = np.array(indices_A_precond)
        indptr_A_precond = np.array(indptr_A_precond)

        A_csr = csr_matrix((data_A, indices_A, indptr_A), shape=(N, N))
        A_precond_csr = csr_matrix((data_A_precond, indices_A_precond, indptr_A_precond), shape=(N, N))

        self.set_A(linear_operator(A_csr, (N, N)))
        self.set_A_precond(linear_operator(A_precond_csr, (N, N)))
        self.set_b(weights * vis)

    def plot_tolerance(self, iteration, tols):
        iterations = np.arange(1, iteration + 2)
        tols = np.array(tols)

        plt.figure(figsize = (5,3))
        plt.plot(iterations, tols)
        plt.xlabel('iterations')
        plt.ylabel('tolerance')
        plt.show()

        plt.figure(figsize = (5,3))
        plt.plot(iterations, np.log(tols))
        plt.xlabel('iterations')
        plt.ylabel('log tolerance')
        plt.show()

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
        start_time = time.time()
        print("Creating sparse linear system...")
        self.build_sparse_linear_system()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'--> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        solver = self.get_method()

        # Solve the linear system.
        start_time = time.time()
        print("Solving linear system...")
        if self._x0 is None:
            x, info = solver(self._A, self._b, M = self._A_precond, rtol = self._rtol, maxiter = self._maxiter)
        else:
            x, info = solver(self._A, self._b, M = self._A_precond, x0 = self._x0, rtol = self._rtol, maxiter = self._maxiter)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'--> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Report on the success of the fitting.
        fit_correctly = np.allclose(self._A.matvec(x), self._b)
        print("-> CGM converged?  ", info == 0)
        print("-> Fit correctly?  ", fit_correctly)
        if fit_correctly:
            print("!!!!  Sucess  !!!!")

        self._solution = x
        return x

    @property
    def sol(self):
        """
        Returns the solution of the linear system.
        """
        return self._sol