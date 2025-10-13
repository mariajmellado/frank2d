import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt


class Gridding(object):
    def __init__(self, Rmax, FT, Geometry):
        """
        Class to grid visibilities in a regular grid.
        Parameters
        ----------
        N : int
            Number of pixels in one side of the image.
        Rmax : float
            Maximum radius of the image in radians.
        FT : FourierTransform object
            Object that contains the Fourier Transform parameters.
        Geometry : Geometry object
            Object that contains the geometry parameters.
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

    def run(self, u, v, Vis, Weights, type = 'weighted',
            unshift = False, hermitian = True):  
        """
        Function to grid visibilities in a regular grid.
        Parameters
        ----------
        u : 1D array, unit = lambda
            u coordinates of the visibilities from the uvtable.
        v : 1D array, unit = lambda
            v coordinates of the visibilities from the uvtable.
        Vis : 1D array, unit = Jy
            Visibilities from the uvtable.
        Weights : 1D array, unit = 1/Jy^2
            Weights from the uvtable.
        type : str, optional
            Type of gridding to perform. Currently only 'weighted' is implemented. Default is 'weighted'.
        unshift : bool, optional
            If True, the output grid will be unshifted to have the zero frequency at the corner. Default is False.
        hermitian : bool, optional
            If True, the output grid will enforce the Hermitian symmetry. Default is True.
        Returns
        -------
        u_gridded : 1D array, unit = lambda
            u coordinates of the gridded visibilities.
        v_gridded : 1D array, unit = lambda
            v coordinates of the gridded visibilities.
        vis_gridded : 1D array, unit = Jy
            Gridded visibilities.
        weights_gridded : 1D array, unit = 1/Jy^2
            Gridded weights.
        """
        u_, v_, Vis_ = u, v, Vis
        
        if not self._set_grid:
            # Calculating bin edges.
            self._bin_centers_u = self._FT._u_shifted
            self._bin_centers_v = self._FT._v_shifted

        bin_edges_u = self.edges_centers(self._bin_centers_u)
        bin_edges_v = self.edges_centers(self._bin_centers_v)

        if type == 'weighted':
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.weighted_gridding(u_, v_, Vis_, Weights,
                                                                                        bin_edges_u, bin_edges_v,
                                                                                        unshift = unshift,
                                                                                        hermitian = hermitian)

            return u_gridded, v_gridded, vis_gridded, weights_gridded

    def edges_centers(self, bin_centers):
        """
        Function to calculate the edges of the bins for gridding
        with the uv-plane shifted.
        Parameters
        ----------
        bin_centers : 1D array, unit = lambda
            Frequencies where the bins are centered.
        Returns
        -------
        bin_edges : 1D array, unit = lambda
            Edges of the bins.
        """
        correction = np.abs(bin_centers[1] - bin_centers[0])/2 # e.g. 0.5

        # Creating the grid with shifted scheme.
        # bin centers: e.g. [-1, 0, 1]
        bin_edges_=  bin_centers - correction # e.g. [-1.5, -0.5, 0.5]
        bin_edges = np.concatenate((bin_edges_, [bin_edges_[-1] + 2*correction])) # e.g. [-1.5, -0.5, 0.5, 1.5]
        return bin_edges
    
    def weighted_gridding(self, u, v, Vis, Weights, edges_u, edges_v, unshift = False, hermitian = True):
        """
        Function to grid visibilities in a regular grid using weighted gridding.
        Parameters
        ----------
        u : 1D array, unit = lambda
            u coordinates of the visibilities from the uvtable.
        v : 1D array, unit = lambda
            v coordinates of the visibilities from the uvtable.
        Vis : 1D array, unit = Jy
            Visibilities from the uvtable.
        Weights : 1D array, unit = 1/Jy^2
            Weights from the uvtable.
        edges_u : 1D array, unit = lambda
            Edges of the bins in u direction.
        edges_v : 1D array, unit = lambda
            Edges of the bins in v direction.
        unshift : bool, optional
            If True, the output grid will be unshifted to have the zero frequency at the corner. Default is False.
        hermitian : bool, optional
            If True, the output grid will enforce the Hermitian symmetry. Default is True.

        Returns
        -------
        u_gridded : 1D array, unit = lambda
            u coordinates of the gridded visibilities.
        v_gridded : 1D array, unit = lambda
            v coordinates of the gridded visibilities.
        vis_gridded : 1D array, unit = Jy
            Gridded visibilities.
        weights_gridded : 1D array, unit = 1/Jy^2
            Gridded weights.
        """
        # Calculating values in grid
        vis_weights_sum_bin, x_, y_, _ = binned_statistic_2d(u, v, Vis*Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        weights_gridded_matrix, _, _, _ = binned_statistic_2d(u, v, Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        vis_gridded_matrix =  vis_weights_sum_bin/weights_gridded_matrix

        # Change Nans by 0 in vis.
        # The transpose is because binned_statistic_2d returns (nx, ny) array.
        vis_gridded = np.nan_to_num(vis_gridded_matrix, nan=0).T
        weights_gridded = np.nan_to_num(weights_gridded_matrix, nan=0).T

        # Imposing hermitian conjugate property.
        if hermitian:
            vis_gridded, weights_gridded = self.enforce_hermitian_symmetry(vis_gridded, weights_gridded)
        
        if unshift == True:
            # Unshifted grid.
            print("Unshiftting grid..")
            vis_gridded = np.fft.fftshift(vis_gridded).ravel(order="C") 
            weights_gridded = np.fft.fftshift(weights_gridded).ravel(order="C") 
            if self._set_grid == False:
                u_gridded, v_gridded = self._FT.uv_points_unshifted
            else:
                u_, v_ = np.fft.fftshift(self._bin_centers_u), np.fft.fftshift(self._bin_centers_v)
                u_gridded, v_gridded = np.meshgrid(u_, v_)
                u_gridded, v_gridded = u_gridded.ravel(order="C"), v_gridded.ravel(order="C")
        else:
            # Default grid shifted i.e. spatial frequencies centered in 0.
            vis_gridded = vis_gridded.ravel(order="C")  
            weights_gridded = weights_gridded.ravel(order="C")
            if self._set_grid == False:
                print("Warning: You are using the default grid from the Fourier Transform object.")
                u_gridded, v_gridded = self._FT._Un, self._FT._Vn
            else:
                u_gridded, v_gridded = np.meshgrid(self._bin_centers_u, self._bin_centers_v)
            u_gridded, v_gridded = u_gridded.ravel(order="C"), v_gridded.ravel(order="C")

        # Change Nans by 0 in vis again.
        vis_gridded = np.nan_to_num(vis_gridded, nan=0)
        weights_gridded = np.nan_to_num(weights_gridded, nan=0)
            
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
        # Shift zero frequency to the center for easier indexing.
        vis = np.fft.fftshift(vis)
        wts = np.fft.fftshift(wts)
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

        vis = np.fft.ifftshift(vis)
        wts = np.fft.ifftshift(wts)
        return vis, wts

    def shiftting(self, freqs, vis_matrix, weights_matrix):
        """
        Function to shift the gridded visibilities to have the zero frequency at the center.
        Parameters
        ----------
        freqs : 1D array, unit = lambda
            Frequencies where the bins are centered.
        vis_matrix : 2D array, unit = Jy
            Gridded visibilities.
        weights_matrix : 2D array, unit = 1/Jy^2
            Gridded weights.
        Returns
        -------
        u_gridded : 1D array, unit = lambda
            u shifted coordinates of the gridded visibilities.
        v_gridded : 1D array, unit = lambda
            v shifted coordinates of the gridded visibilities.
        vis_gridded : 1D array, unit = Jy
            Shifted gridded visibilities.
        weights_gridded : 1D array, unit = 1/Jy^2
            Shifted gridded weights.
        """
        vis_gridded = vis_matrix.reshape(-1)
        weights_gridded = weights_matrix.reshape(-1)
        u_, v_ = np.meshgrid(freqs, freqs, indexing='ij') 
        u_gridded, v_gridded = u_.reshape(-1), v_.reshape(-1)
        return u_gridded, v_gridded, vis_gridded, weights_gridded