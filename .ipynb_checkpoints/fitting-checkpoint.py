import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg._isolve.utils import make_system
import time

class IterativeSolverMethod():
    def __init__(self, Vis, Weights, kernel_row, FT, method = 'cg', rtol = 1e-7,  x0=None):
        self._FT = FT
        self._N2 = self._FT._Ny*self._FT._Nx

        self._Vis = Vis
        self._Weights = Weights
        self._kernel_row = kernel_row
        
        self._method = method
        self._x0 = x0
        self._rtol = rtol

        self._A = None
        self._b = None
        self._A_precond = None

        self._sparse_system = None

        self._set_A = False
        self._set_b = False
        self._set_A_precond = False
    
    def get_method(self):
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
        print("Setting A...")
        self._A = A
        self._set_A = True
    
    def set_b(self, b):
        print("Setting b...")
        self._b = b
        self._set_b = True

    def set_A_precond(self, A_precond):
        print("Setting A_precond...")
        self._A_precond = A_precond
        self._set_A_precond = True

    def _get_atol_rtol(self, name, b_norm, atol=0., rtol=1e-5):
        """
        A helper function to handle tolerance normalization
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
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('cg', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        if maxiter is None:
            maxiter = int(n*1e4)
        print("maxiter: ", maxiter)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        matvec = A.matvec
        psolve = M.matvec
        r = b - matvec(x) if x.any() else b.copy()

        # Dummy value to initialize var, silences warnings
        rho_prev, p = None, None

        p_prev = None
        r_prev = None

        for iteration in range(maxiter):
            print("iteration: ", iteration)
            print("          actual tolerance: ", np.linalg.norm(r))
            if np.linalg.norm(r) < atol:  # Are we done?
                print("  --> CGM converged in ", iteration, " iterations")
                return postprocess(x), 0
            
            z = psolve(r)
            rho_cur = dotprod(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev
                p *= beta
                p += z


                # Verificación de la conjugación con la dirección anterior
                conjugacy = dotprod(p_prev, matvec(p))
                print(f"Conjugacy check (should be close to 0): {conjugacy}")
                if np.abs(conjugacy) > 1e-10:  # Umbral de tolerancia
                    print(f"!!!!!!!! Warning: Loss of conjugacy at iteration {iteration} (value: {conjugacy})")

                # Verificación de la ortogonalidad de los residuos
                orthogonality = dotprod(r_prev, r)
                print(f"Orthogonality check (should be close to 0): {orthogonality}")
                if np.abs(orthogonality) > 1e-10:
                    print(f"!!!!!!!! Warning: Loss of orthogonality at iteration {iteration} (value: {orthogonality})")
 
            else:  # First spin
                p = np.empty_like(r)
                p[:] = z[:]

            q = matvec(p)
            alpha = rho_cur / dotprod(p, q)
            x += alpha*p
            r -= alpha*q
            rho_prev = rho_cur

            p_prev = p.copy()
            r_prev = r.copy()

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return postprocess(x), maxiter

    def cgs(self, A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('cgs', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec = A.matvec
        psolve = M.matvec

        rhotol = np.finfo(x.dtype.char).eps**2

        r = b - matvec(x) if x.any() else b.copy()

        rtilde = r.copy()
        bnorm = np.linalg.norm(b)
        if bnorm == 0:
            bnorm = 1

        # Dummy values to initialize vars, silence linter warnings
        rho_prev, p, u, q = None, None, None, None

        for iteration in range(maxiter):
            rnorm = np.linalg.norm(r)
            if rnorm < atol:  # Are we done?
                return postprocess(x), 0

            rho_cur = dotprod(rtilde, r)
            if np.abs(rho_cur) < rhotol:  # Breakdown case
                return postprocess, -10

            if iteration > 0:
                beta = rho_cur / rho_prev

                # u = r + beta * q
                # p = u + beta * (q + beta * p);
                u[:] = r[:]
                u += beta*q

                p *= beta
                p += q
                p *= beta
                p += u

            else:  # First spin
                p = r.copy()
                u = r.copy()
                q = np.empty_like(r)

            phat = psolve(p)
            vhat = matvec(phat)
            rv = dotprod(rtilde, vhat)

            if rv == 0:  # Dot product breakdown
                return postprocess(x), -11

            alpha = rho_cur / rv
            q[:] = u[:]
            q -= alpha*vhat
            uhat = psolve(u + q)
            x += alpha*uhat

            # Due to numerical error build-up the actual residual is computed
            # instead of the following two lines that were in the original
            # FORTRAN templates, still using a single matvec.

            # qhat = matvec(uhat)
            # r -= alpha*qhat
            r = b - matvec(x)

            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return postprocess(x), maxiter

    def bicg(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('bicg', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

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
                return postprocess(x), 0

            z = psolve(r)
            ztilde = rpsolve(rtilde)
            # order matters in this dot product
            rho_cur = dotprod(rtilde, z)

            if np.abs(rho_cur) < rhotol:  # Breakdown case
                return postprocess, -10

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
                return postprocess(x), -11

            alpha = rho_cur / rv
            x += alpha*p
            r -= alpha*q
            rtilde -= alpha.conj()*qtilde
            rho_prev = rho_cur

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            return postprocess(x), maxiter

    def bicgstab(self, A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=None, M=None, callback=None):
        print("     * BICGSTAB")
        A, M, x, b, postprocess = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)

        atol, _ = self._get_atol_rtol('bicgstab', bnrm2, atol, rtol)

        if bnrm2 == 0:
            return postprocess(b), 0

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10
        print("         * maxiter: ", maxiter)

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

        for iteration in range(maxiter):
            print("             .. iteration: ", iteration)
            if np.linalg.norm(r) < atol:  # Are we done?
                print("        * CGM converged in ", iteration, " iterations")
                return postprocess(x), 0

            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                return postprocess(x), -10

            if iteration > 0:
                if np.abs(omega) < omegatol:  # omega breakdown
                    return postprocess(x), -11

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
                return postprocess(x), -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]

            if np.linalg.norm(s) < atol:
                x += alpha*phat
                return postprocess(x), 0

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
            return postprocess(x), maxiter

    def linear_op_A(self):
        def SWI(vis_model):
            return np.array([ 
                    np.dot(np.add(
                        self._kernel_row(i)*self._Weights, np.eye(1, self._N2, i).flatten()), vis_model)
                    for i in range(self._N2)
                    ])
        def WSI(vis_model):
            return np.array([ 
                    np.dot(np.add(
                        self._kernel_row(i)*self._Weights[i], np.eye(1, self._N2, i).flatten()), vis_model)
                    for i in range(self._N2)
                    ])
        
        return LinearOperator((self._N2, self._N2 ), matvec = SWI, rmatvec = WSI)
    
    def linear_op_b(self):
        def SW(vis_gridded):
            return np.array([
                    np.dot(
                        self._kernel_row(i)*self._Weights, vis_gridded)
                    for i in range(self._N2)
                    ])
        def WS(vis_gridded):
            return np.array([
                    np.dot(
                        self._kernel_row(i)*self._Weights[i], vis_gridded)
                    for i in range(self._N2)
                    ])

        bop = LinearOperator((self._N2, self._N2), matvec= SW, rmatvec= WS)
        return bop.matvec(self._Vis)

    def linear_op_A_precond(self):
        def diag_SWI(vis_model):
            return np.array([ 
                    ((self._kernel_row(i)[i] * self._Weights[i] + 1)**(-1) * vis_model[i])
                    for i in range(self._N2)
                    ])

        return LinearOperator((self._N2, self._N2), matvec=diag_SWI, rmatvec=diag_SWI)

    def create_sparse_system(self):
        data_A = []
        indices_A = []
        indptr_A = [0]

        data_Aprecond = []
        indices_Aprecond = []
        indptr_Aprecond = [0]

        data_b = []
        indices_b = []
        indptr_b = [0]

        for i in range(self._N2):  
            row = self._kernel_row(i)
            weights = self._Weights

            non_zero_indices = np.nonzero(row)[0]

            weighted_non_zero_values = row[non_zero_indices] * weights[non_zero_indices]

            # b
            data_b.extend(weighted_non_zero_values)
            indices_b.extend(non_zero_indices)
            indptr_b.append(len(data_b))
            
            diagonal_pos = np.searchsorted(non_zero_indices, i)
            weighted_non_zero_values[diagonal_pos] += 1
            diag_value = weighted_non_zero_values[diagonal_pos]

            # A
            data_A.extend(weighted_non_zero_values)
            indices_A.extend(non_zero_indices)
            indptr_A.append(len(data_A))

            # Preconditioner of A
            data_Aprecond.extend([diag_value**(-1)])
            indices_Aprecond.extend([i])
            indptr_Aprecond.append(len(data_Aprecond))
            
        data_A = np.array(data_A)
        indices_A = np.array(indices_A)
        indptr_A = np.array(indptr_A)

        data_Aprecond = np.array(data_Aprecond)
        indices_Aprecond = np.array(indices_Aprecond)
        indptr_Aprecond = np.array(indptr_Aprecond)

        data_b = np.array(data_b)
        indices_b = np.array(indices_b)
        indptr_b = np.array(indptr_b)

        A_csr = csr_matrix((data_A, indices_A, indptr_A), shape=(self._N2, self._N2))
        A_precond_csr = csr_matrix((data_Aprecond, indices_Aprecond, indptr_Aprecond), shape=(self._N2, self._N2))
        b_csr = csr_matrix((data_b, indices_b, indptr_b), shape=(self._N2, self._N2))

        self.set_A(self.linear_operator(A_csr))
        self.set_A_precond(self.linear_operator(A_precond_csr))
        self.set_b(self.linear_operator(b_csr).matvec(self._Vis))

    def linear_operator(self, A):
        def dot_product(x):
            return A.dot(x)

        return LinearOperator((self._N2, self._N2), matvec=dot_product)    


    def solve(self):
        print("  *  Constructing linear operators...")
        start_time = time.time()

        self.create_sparse_system()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Solve the linear system
        solver = self.get_method()
        print("  *  Solving linear system...")
        start_time = time.time()

        if self._x0 is None:
            x, info = solver(self._A, self._b, M = self._A_precond, rtol = self._rtol)
        else:
            x, info = solver(self._A, self._b, M = self._A_precond, x0 = self._x0, rtol = self._rtol)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'     --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')

        # Report on the success of the fitting
        fit_correctly = np.allclose(self._A.matvec(x), self._b)
        print("  --> CGM converged?  ", info == 0)
        print("  --> Fit correctly?  ", fit_correctly)
        if fit_correctly:
            print("                                    !!!!!!!!!!!!!!!!!!!")

        return x