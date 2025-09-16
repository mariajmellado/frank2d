import numpy as np
from scipy.stats import binned_statistic_2d
import constants as const
import cupy as cp

class Gridding_GPU(object):
    def __init__(self, N, Rmax, FT, Geometry):
        self._N = N
        self._Rmax = Rmax
        self._FT =  FT
        self._Geometry = Geometry

    def run(self, u, v, Vis, Weights, type = 'weighted', shift = False, hermitian = True):  
        
        u_, v_, Vis_ = u, v, Vis

        if self._Geometry._deproject:
            print("Deprojecting...")
            u_, v_, Vis_ = self._Geometry.apply_correction(u, v, Vis)
        
        # Calculating bin edges.
        bin_centers = self.edges_centers(self._FT._u_shifted)[0]
        bin_edges_u = self.edges_centers(self._FT._u_shifted)[1]
        bin_edges_v = self.edges_centers(self._FT._v_shifted)[1]


        if type == 'weighted':
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.weighted_gridding(u_, v_, Vis_, Weights,
                                                                                        bin_centers, bin_edges_u, bin_edges_v,
                                                                                        shift = shift, hermitian = hermitian)

            return u_gridded, v_gridded, vis_gridded, weights_gridded

    def edges_centers(self, freq):
        correction = (freq[1] - freq[0])/2
        
        # Creating the grid with shifted scheme.
        bin_centers = freq
        bin_edges_=  bin_centers - correction
        bin_edges = cp.concatenate((bin_edges_, cp.array([bin_edges_[-1] + 2 * correction])))
        return bin_centers, bin_edges
    
    def weighted_gridding(self, u, v, Vis, Weights, centers, edges_u, edges_v, shift = False, hermitian = True):
        # Calculating values in grid
        vis_weights_sum_bin, _, _ = binned_statistic_2d_cupy(u, v, Vis*Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        weights_sum_bin, _, _ = binned_statistic_2d_cupy(u, v, Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        vis_gridded_matrix =  vis_weights_sum_bin/weights_sum_bin
        weights_gridded_matrix, _, _ = binned_statistic_2d_cupy(u, v, Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)

        # Change Nans by 0 in vis.
        vis_gridded = cp.nan_to_num(vis_gridded_matrix, nan=0)
        weights_gridded = cp.nan_to_num(weights_gridded_matrix, nan=0)

        # Imposing hermitian conjugate property.
        if hermitian:
            vis_gridded, weights_gridded = self.enforce_hermitian_symmetry(vis_gridded, weights_gridded)

        if shift:
            # Shifting the grid, i.e. spatial frequencies centered in 0.
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.shiftting(centers, vis_gridded, weights_gridded)
        else:
            # Unshifted grid.
            u_gridded, v_gridded = self._FT._Un, self._FT._Vn # unshifted by default.
            vis_gridded = cp.fft.fftshift(vis_gridded).flatten()
            weights_gridded = cp.fft.fftshift(weights_gridded).flatten()

        # Change Nans by 0 in vis again.
        vis_gridded = cp.nan_to_num(vis_gridded, nan=0)
        weights_gridded = cp.nan_to_num(weights_gridded, nan=0)
            
        return u_gridded, v_gridded, vis_gridded, weights_gridded

    def enforce_hermitian_symmetry(self, vis, wts):
        vis = cp.fft.fftshift(vis)
        wts = cp.fft.fftshift(wts)
        nx, ny = vis.shape
        cx, cy = (nx // 2), (ny // 2)

        print("Enforcing Hermitian symmetry...")
        for x in range(nx):  
            for y in range(ny):
                x_sym = (-x) % nx
                y_sym = (-y) % ny

                v_xy = vis[y, x]
                v_neg_xy = vis[y_sym, x_sym]

                w_xy = wts[y, x]
                w_neg_xy = wts[y_sym, x_sym]

                # Aplicamos hermiticidad si hay al menos un peso válido
                if w_xy > 0 or w_neg_xy > 0:
                    w_tot = w_xy + w_neg_xy
                    if w_tot > 0:
                        val = (cp.conj(v_neg_xy) * w_neg_xy + v_xy * w_xy) / w_tot
                        vis[y, x] = val
                        vis[y_sym, x_sym] = cp.conj(val)

                        wts[y, x] = w_tot
                        wts[y_sym, x_sym] = w_tot

        vis = cp.fft.ifftshift(vis)
        wts = cp.fft.ifftshift(wts)
        return vis, wts

    def shiftting(self, freqs, vis_matrix, weights_matrix):
        vis_gridded = vis_matrix.flatten()
        weights_gridded = weights_matrix.flatten()
        u_, v_ = cp.meshgrid(freqs, freqs, indexing='ij') 
        u_gridded, v_gridded = u_.reshape(-1), v_.reshape(-1)
        return u_gridded, v_gridded, vis_gridded, weights_gridded
        
def binned_statistic_2d_cupy(x, y, values, statistic='mean', bins=10, range=None, expand_binnumbers=False):
    if isinstance(bins, (list, tuple)) and len(bins) == 2:
        x_edges, y_edges = cp.asarray(bins[0]), cp.asarray(bins[1])
        bins_x, bins_y = len(x_edges) - 1, len(y_edges) - 1
    else:
        bins_x = bins_y = bins
        hist, x_edges, y_edges = cp.histogram2d(x, y, bins=bins, range=range)

    x_bin = cp.digitize(x, x_edges) - 1
    y_bin = cp.digitize(y, y_edges) - 1
    valid = (x_bin >= 0) & (x_bin < bins_x) & (y_bin >= 0) & (y_bin < bins_y)
    x_bin, y_bin, values = x_bin[valid], y_bin[valid], values[valid]
    bin_idx = (x_bin * bins_y + y_bin).astype(cp.int32)

    if statistic == 'count':
        counts = cp.zeros(bins_x * bins_y, dtype=cp.float32)
        cp.ElementwiseKernel(
            'int32 idx',
            'raw float32 out',
            'atomicAdd(&out[idx], 1.0f)',
            'bincount_count_kernel'
        )(bin_idx, counts)
        bin_stat = counts.reshape(bins_x, bins_y)

    else:
        # Prealocation of output arrays.
        real_out = cp.zeros(bins_x * bins_y, dtype=cp.float32)
        cp.ElementwiseKernel(
            'int32 idx, float32 val',
            'raw float32 out',
            'atomicAdd(&out[idx], val)',
            'bincount_real_kernel'
        )(bin_idx, cp.real(values).astype(cp.float32), real_out)
        real_vals = real_out.reshape(bins_x, bins_y)

        if cp.iscomplexobj(values):
            imag_out = cp.zeros(bins_x * bins_y, dtype=cp.float32)
            cp.ElementwiseKernel(
                'int32 idx, float32 val',
                'raw float32 out',
                'atomicAdd(&out[idx], val)',
                'bincount_imag_kernel'
            )(bin_idx, cp.imag(values).astype(cp.float32), imag_out)
            imag_vals = imag_out.reshape(bins_x, bins_y)
            sum_vals = real_vals + 1j * imag_vals
        else:
            sum_vals = real_vals

        if statistic == 'sum':
            bin_stat = sum_vals
        elif statistic == 'mean':
            count_out = cp.zeros(bins_x * bins_y, dtype=cp.float32)
            cp.ElementwiseKernel(
                'int32 idx',
                'raw float32 out',
                'atomicAdd(&out[idx], 1.0f)',
                'bincount_count_kernel'
            )(bin_idx, count_out)
            count_vals = count_out.reshape(bins_x, bins_y)
            bin_stat = cp.divide(sum_vals, count_vals, where=(count_vals > 0))
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")

    if expand_binnumbers:
        return bin_stat, x_edges, y_edges, (x_bin, y_bin)

    return bin_stat, x_edges, y_edges