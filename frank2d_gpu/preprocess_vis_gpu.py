import cupy as cp
import numpy as np


class Gridding(object):
    def __init__(self, Rmax, FT, Geometry):
        """
        Class to grid visibilities in a regular grid (CuPy backend).
        """
        self._Rmax = Rmax
        self._FT =  FT
        self._Geometry = Geometry

        self._set_grid = False
    
    def set_bins(self, bin_centers_u, bin_centers_v):
        """
        Set the bin centers for gridding.
        Parameters
        ----------
        bin_centers_u : 1D array, unit = lambda
            Frequencies where the bins are centered in u direction.
        bin_centers_v : 1D array, unit = lambda
            Frequencies where the bins are centered in v direction.
        """
        self._bin_centers_u = bin_centers_u
        self._bin_centers_v = bin_centers_v

        self._set_grid = True

    def run(self, u, v, Vis, Weights, type='weighted',
            unshift=False, hermitian=True):
        """
        Function to grid visibilities in a regular grid.
        """
        u_, v_, Vis_ = u, v, Vis
        
        if not self._set_grid:
            # Calculating bin edges.
            self._bin_centers_u = self._FT._u_shifted
            self._bin_centers_v = self._FT._v_shifted

        bin_edges_u = self.edges_centers(self._bin_centers_u)
        bin_edges_v = self.edges_centers(self._bin_centers_v)

        if type == 'weighted':
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.weighted_gridding(
                u_, v_, Vis_, Weights,
                bin_edges_u, bin_edges_v,
                unshift=unshift, hermitian=hermitian
            )
            return u_gridded, v_gridded, vis_gridded, weights_gridded

    def edges_centers(self, bin_centers):
        """
        Compute bin edges from shifted bin centers.
        """
        correction = cp.abs(bin_centers[1] - bin_centers[0]) / 2
        bin_edges_ = bin_centers - correction
        bin_edges = cp.concatenate((bin_edges_, bin_edges_[-1:]+2*correction))
        return bin_edges
    
    def weighted_gridding(self, u, v, Vis, Weights, edges_u, edges_v,
                          unshift=False, hermitian=True):
        """
        Weighted gridding on a regular (u, v) grid.
        """
        # Weighted sums per bin using histogram2d (separating real/imag)
        vw = Vis * Weights
        H_w, _, _ = cp.histogram2d(u, v, bins=[edges_u, edges_v], weights=Weights)
        H_vw_r, _, _ = cp.histogram2d(u, v, bins=[edges_u, edges_v], weights=cp.real(vw))
        H_vw_i, _, _ = cp.histogram2d(u, v, bins=[edges_u, edges_v], weights=cp.imag(vw))

        vis_gridded_matrix = (H_vw_r + 1j * H_vw_i) / H_w

        # Replace NaNs with 0 and transpose to match original orientation
        vis_gridded = cp.nan_to_num(vis_gridded_matrix, nan=0).T
        weights_gridded = cp.nan_to_num(H_w, nan=0).T

        # Enforce Hermitian symmetry if requested
        if hermitian:
            vis_gridded, weights_gridded = self.enforce_hermitian_symmetry(vis_gridded, weights_gridded)

        if unshift == True:
            # Unshifted grid.
            print("Unshiftting grid..")
            vis_gridded = cp.fft.fftshift(vis_gridded).ravel(order="C") 
            weights_gridded = cp.fft.fftshift(weights_gridded).ravel(order="C") 
            if self._set_grid == False:
                u_gridded, v_gridded = self._FT.uv_points_unshifted
            else:
                u_, v_ = cp.fft.fftshift(self._bin_centers_u), cp.fft.fftshift(self._bin_centers_v)
                u_gridded, v_gridded = cp.meshgrid(u_, v_)
                u_gridded, v_gridded = u_gridded.ravel(order="C"), v_gridded.ravel(order="C")
        else:
            # Default grid shifted i.e. spatial frequencies centered in 0.
            vis_gridded = vis_gridded.ravel(order="C")  
            weights_gridded = weights_gridded.ravel(order="C")
            if self._set_grid == False:
                print("Warning: You are using the default grid from the Fourier Transform object.")
                u_gridded, v_gridded = self._FT._Un, self._FT._Vn
            else:
                u_gridded, v_gridded = cp.meshgrid(self._bin_centers_u, self._bin_centers_v)
            u_gridded, v_gridded = u_gridded.ravel(order="C"), v_gridded.ravel(order="C")

        # Final NaN cleanup (shouldn't be needed but safe)
        vis_gridded = cp.nan_to_num(vis_gridded, nan=0)
        weights_gridded = cp.nan_to_num(weights_gridded, nan=0)
            
        return u_gridded, v_gridded, vis_gridded, weights_gridded

    def enforce_hermitian_symmetry(self, vis, wts):
        """
        Function to enforce Hermitian symmetry on the gridded visibilities.
        Parameters
        ----------
        vis : 2D array, unit = Jy
            Gridded visibilities.
        wts : 2D array, unit = 1/Jy^2
            Gridded weights.
        Returns
        -------
        vis : 2D array, unit = Jy
            Gridded visibilities with Hermitian symmetry enforced.
        wts : 2D array, unit = 1/Jy^2
            Gridded weights with Hermitian symmetry enforced.
        """
        vis = vis.get()
        wts = wts.get()

        nx, ny = vis.shape  
        cx, cy = (nx // 2), (ny // 2)

        for x in range(nx):  
            for y in range(ny):
                x_sym = (-x) % nx
                y_sym = (-y) % ny

                v_xy = vis[y, x]
                v_neg_xy = vis[y_sym, x_sym]

                w_xy = wts[y, x]
                w_neg_xy = wts[y_sym, x_sym]

                if w_xy > 0 or w_neg_xy > 0:    
                    w_tot = w_xy + w_neg_xy
                    if w_tot > 0:
                        val = (np.conj(v_neg_xy) * w_neg_xy + v_xy * w_xy) / w_tot
                        vis[y, x] = val
                        vis[y_sym, x_sym] = np.conj(val)

                        wts[y, x] = w_tot
                        wts[y_sym, x_sym] = w_tot
        
        vis = cp.array(vis)
        wts = cp.array(wts)

        return vis, wts

    def shiftting(self, freqs, vis_matrix, weights_matrix):
        """
        Shift gridded visibilities to have zero frequency at the center.
        """
        vis_gridded = vis_matrix.reshape(-1)
        weights_gridded = weights_matrix.reshape(-1)
        u_, v_ = cp.meshgrid(freqs, freqs, indexing='ij')
        u_gridded, v_gridded = u_.reshape(-1), v_.reshape(-1)
        return u_gridded, v_gridded, vis_gridded, weights_gridded