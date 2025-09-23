#!/usr/bin/env python
# coding: utf-8

# ## Frank2D CPU 
# ### Description
# This algorithm uses gridded data from gridding the UVTABLE data, to optimize the calculation of the visibilities solution through Frankenstein scheme. 

# 
# #### Imports 

# In[1]:


import os
import sys
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
import time


from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.sparse.linalg import LinearOperator, splu
from scipy.sparse.linalg._isolve.utils import make_system
from scipy.sparse import csr_matrix

current_dir =  os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

#frank2d
from frank2d import Frank2D
from constants import rad_to_arcsec, deg_to_rad
from plot import Plot
from fitting import IterativeSolverMethod
from preprocess_vis import Gridding
from geometry import Geometry


# #### Functions

# #### BICGM
# 

# In[2]:


def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
            "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)
    
    print(f'         * rtol: {float(rtol)}')
    
    atol = max(float(atol), float(rtol) * float(b_norm))
    print(f'         * final tolerance : {atol}')
    
    return atol, rtol


# In[3]:

def plot_tolerance(iteration, tols):
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
    

def bicgstab(A, b, x0=None, *, rtol=1e-7, atol=0., maxiter=20000, M=None, callback=None):
        print("     * BICGSTAB")
        A, M, x, b = make_system(A, M, x0, b)
        bnrm2 = np.linalg.norm(b)
    
        atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol)
    
        print("....................... TOLERANCE: ", atol)
    
        if bnrm2 == 0:
            return b, 0
    
        n = len(b)
    
        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

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

        tols = []
    
        for iteration in range(maxiter):
            print(".... iteration: ", iteration)
            act_tol = np.linalg.norm(r)
            tols.append(act_tol)
            print("                        -> actual tol: ", str(act_tol), "vs ", str(atol))
            if act_tol < atol:  # Are we done?
                print(" --------------------------------------> CGM converged in ", iteration, " iterations with tol ", act_tol)

                plot_tolerance(iteration, tols)
                
                return x, 0
    
            rho = dotprod(rtilde, r)
            if np.abs(rho) < rhotol:  # rho breakdown
                print("converged by norm of rho")
                plot_tolerance(iteration, tols)
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
                plot_tolerance(iteration, tols)
                return x, -11
            alpha = rho / rv
            r -= alpha*v
            s[:] = r[:]
    
            if np.linalg.norm(s) < atol:
                print("converged by norm of s")
                x += alpha*phat
                plot_tolerance(iteration, tols)
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

            plot_tolerance(maxiter-1, tols)
            
            return x, maxiter


# ####  Functions for optimization

# In[4]:


def linear_operator(A, size):
    def dot_product(x):
        return A.dot(x)

    return LinearOperator(size, matvec=dot_product)


# In[5]:


def P_k(r, k):
    if k == 0:
        return np.ones_like(r)  # P_0(r) = 1
    elif k == 1:
        return 4*r +1  # P_1(r) = 4r + 1
    elif k == 2:
        return (35/3)*r**2 +6*r + 1  # P_2(r) = 35r^2 + 18r + 3
    else:
        raise ValueError("k must be 0, 1, or 2.")


# In[6]:


def kernel_row(u, v, i, min_freq, kernel_params, u2 = None, v2 = None):
    m = kernel_params['m']
    c = kernel_params['c']
    l = kernel_params['l']
    
    def power_spectrum(q, m, c):
        if not np.isscalar(q):  
            q[q == 0] = min_freq
        elif q == 0:
            q = min_freq
        return c*(q**m)
    
    j, k = 4, 1
    H = 2*1.897367*l

    q = np.hypot(u, v)
    power_spectrum_q1 = power_spectrum(q, m, c)
    uh = u/H
    vh = v/H

    q2 = uh2 = vh2 =  None

    if isinstance(u2, np.ndarray):
        q2 = np.hypot(u2[i], v2[i])
        uh2 = u2[i]/H
        vh2 = v2[i]/H
    else:
        q2 = q[i]
        uh2 = uh[i]
        vh2 = vh[i]

    amp = np.sqrt(power_spectrum_q1 * power_spectrum(q2, m, c))
    
    r_normalized = np.sqrt((uh-uh2)**2 + (vh-vh2)**2)
    factor = (1 - r_normalized)**j
    factor[r_normalized > 1] = 0

    return amp * factor * P_k(r_normalized, k)


# In[7]:


def create_sparse_kernel(kernel_function, u, v, min_freq, kernel_params, u2 = None, v2 = None ):
    data = []
    indices = []
    indptr = [0]
    size1 = size2 = None

    if isinstance(u2, np.ndarray):
        size1 = len(u2)
        size2 = len(u)
    else:
        size1 = size2 = len(u)

    for i in range(size1):
        row = kernel_function(u, v, i, min_freq, kernel_params, u2 = u2, v2 = v2) if isinstance(u2, np.ndarray) else kernel_function(u, v, i, min_freq, kernel_params)
            
        non_zero_indices = np.nonzero(row)[0]
        kernel_values = row[non_zero_indices]
        
        data.extend(kernel_values)
        indices.extend(non_zero_indices)
        indptr.append(len(data))

    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)

    size = (size1, size2)
    kernel_csr = csr_matrix((data, indices, indptr), shape=size)
    kernel = linear_operator(kernel_csr, size)

    return kernel


# In[8]:


def create_sparse_system_data(u_gridded_data, v_gridded_data, vis_gridded_data, weights_gridded_data, min_freq, kernel_params):
    N_data = vis_gridded_data.shape[0]
    
    data_A = []
    indices_A = []
    indptr_A = [0]

    data_Aprecond = []
    indices_Aprecond = []
    indptr_Aprecond = [0]
    
    for i in range(N_data):  
        row = kernel_row(u_gridded_data, v_gridded_data, i, min_freq, kernel_params)
        non_zero_indices = np.nonzero(row)[0]
        non_zero_values = row[non_zero_indices]
        Nm1_S_values = non_zero_values * weights_gridded_data[i]

        # A
        diagonal_pos = np.where((non_zero_indices == i))[0][0]
        Nm1_S_values[diagonal_pos] += 1
        diag_value = Nm1_S_values[diagonal_pos]
        
        data_A.extend(Nm1_S_values)
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

    A_csr = csr_matrix((data_A, indices_A, indptr_A), shape=(N_data, N_data))
    A_precond_csr = csr_matrix((data_Aprecond, indices_Aprecond, indptr_Aprecond), shape=(N_data, N_data))

    A = linear_operator(A_csr, (N_data, N_data))
    A_precond = linear_operator(A_precond_csr, (N_data, N_data))
    b = weights_gridded_data * vis_gridded_data

    return A, A_precond, b


# ### Frank2D CPU

# **Params for algorithm: N, u_gridded, v_gridded, Visibilities_gridded, Weights_gridded**

# In[9]:


def Frank2D_optimized(N, u_gridded, v_gridded, vis_gridded, weights_gridded,
                      maxiter = 50000, 
                      kernel_params = {'m': -2, 'c': 1e8, 'l': 1e5},
                      rtol = 1e-9, 
                      x0 = None):
    print("RUNNING WITH ", kernel_params )
    
    input = weights_gridded.reshape(N, N)
    data_index = np.argwhere(input.flatten() != 0)
    no_data_index = np.argwhere(input.flatten() == 0) 

    # data
    u_gridded_data = u_gridded[data_index].flatten()
    v_gridded_data = v_gridded[data_index].flatten()
    vis_gridded_data  = vis_gridded[data_index].flatten()
    weights_gridded_data = weights_gridded[data_index].flatten()
    
    if x0 != None:
        x0_data = x0[data_index].flatten()
    else: 
        x0_data = None

    # no data or data with weights == 0.
    u_gridded_no_data = u_gridded[no_data_index].flatten()
    v_gridded_no_data = v_gridded[no_data_index].flatten()
    vis_gridded_no_data  = vis_gridded[no_data_index].flatten()
    weights_gridded_no_data = weights_gridded[no_data_index].flatten()

    #  We intend to solve now:  (I + N^{-1} S_{11})  V* = N^{-1} V_{data} with CGM
    min_freq = np.sort(np.abs(np.unique(v_gridded_data)))[1]

    print("------> Creating linear operators")
    start_time = time.time()
    S11_linearOp = create_sparse_kernel(kernel_row, u_gridded_data, v_gridded_data, min_freq, kernel_params)
    S12_T_linearOp = create_sparse_kernel(kernel_row, u_gridded_data, v_gridded_data, min_freq, kernel_params,
                                          u2 = u_gridded_no_data, v2 = v_gridded_no_data)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
    print("---------------------------------")

    print("------> Creating sparse system")
    start_time = time.time()
    params = [u_gridded_data, v_gridded_data, vis_gridded_data, weights_gridded_data, min_freq, kernel_params]
    A_opt, A_precond_opt, b_opt = create_sparse_system_data(*params)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
    print("---------------------------------")

    print("------> Running CGM")
    start_time = time.time()
    x_data, info = bicgstab(A_opt, b_opt, x0 = x0_data, M = A_precond_opt, rtol = rtol, maxiter=maxiter)
    print("CGM with info: ", info)
    C_ = x_data
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'  --> time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
    print("---------------------------------")
    
    print("------> CGM converged?  ", info == 0)
    fit_correctly = np.allclose(A_opt.matvec(C_), b_opt)
    print("              --> Fit correctly?  ", fit_correctly)
    if fit_correctly:
        print("                                                     !!!!!!!!!!!!!!!!!!! CORRECT !!!!!!!!!!!!!!!!!!!")
    print("---------------------------------")
    

    def recovering_sol(C):
        V_1_m = S11_linearOp.matvec(C)
        V_2_m = S12_T_linearOp.matvec(C)
        
        V_m = np.zeros((N, N), dtype = 'c16')
        
        data_coords = np.unravel_index(data_index.flatten(), (N, N))
        no_data_coords = np.unravel_index(no_data_index.flatten(), (N, N))
        
        V_m[data_coords] = V_1_m
        V_m[no_data_coords] = V_2_m
    
        u_m = np.zeros((N, N))
        v_m = np.zeros((N, N))
        
        u_m[data_coords] = u_gridded_data
        u_m[no_data_coords] = u_gridded_no_data
        
        v_m[data_coords] = v_gridded_data
        v_m[no_data_coords] = v_gridded_no_data
        
        return V_m

    V_m_2 = recovering_sol(C_)
    return V_m_2

