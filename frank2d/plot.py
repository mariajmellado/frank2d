import numpy as np
import matplotlib.pyplot as plt
from .constants import rad_to_arcsec, deg_to_rad
from scipy.stats import binned_statistic
import matplotlib.colors as colors

# frank1d utilities
from frank.geometry import SourceGeometry
from frank.radial_fitters import FrankFitter
from frank.utilities import UVDataBinner

"""
This module contains classes for plotting the results of Frank's 2D algorithm.
"""

class Plot(object):
    def __init__(self, Frank2D, Geometry):
        """
        Plotting class for Frank's 2D algorithm.
        Parameters
        ---------
        Frank2D: Frank2D object
            The Frank2D object containing the results to be plotted.
        Geometry: Geometry object
            The Geometry object containing the source geometry information.
        """
        self._frank2d = Frank2D
        self._Geometry = Geometry
        
        gridded_input = self._frank2d._gridded_data
        self._u_gridded, self._v_gridded = gridded_input['u'], gridded_input['v']
        self._vis_gridded, self._weights_gridded = gridded_input['vis'], gridded_input['weights']
        self._Nx = self._Ny = Frank2D._N

        
    def intensity_model(self, title="Frank2D intensity model", fig_size = 6, vmin = 0, vmax = 1e11, lim = 2):
        """
        Plot the intensity model.
        Parameters
        ---------
        title: str, optional
            Title of the plot.
        fig_size: float, optional
            Size of the figure. Default is 6.
        vmin: float, optional
            Minimum value for the color scale. Default is 0.
        vmax: float, optional
            Maximum value for the color scale. Default is 1e11.
        lim: float, optional
            Limit for the x and y axes in arcseconds. Default is 2.
        """
        frank2d = self._frank2d

        I = frank2d.sol_intensity

        x_ = frank2d._FT._Xn * rad_to_arcsec
        y_ = frank2d._FT._Yn * rad_to_arcsec
        Nx, Ny = self._Nx, self._Ny
        x, y = x_, y_

        plt.figure(figsize=(fig_size, fig_size))

        X = x.reshape(Ny, Nx)
        Y = y.reshape(Ny, Nx)
        I = I.reshape(Ny, Nx)

        I_flip = np.fliplr(I)
        X_flip = -np.fliplr(X)

        norm = colors.PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
        plot = plt.pcolormesh(X_flip, Y, I_flip,
                              cmap='magma',norm=norm)

        cmap = plt.colorbar(plot, shrink=0.8)
        cmap.set_label(r' $I_{\nu}$ [Jy $sr^{-1}$]', size=10)

        plt.title(title)
        plt.xlabel(r'$\Delta\alpha$ ["]')
        plt.ylabel(r'$\Delta\delta$ ["]')

        plt.gca().set_aspect(1)  


        plt.xlim(lim, -lim)
        plt.ylim(-lim, lim)

        xticks = np.linspace(lim, -lim, 5)
        yticks = np.linspace(-lim, lim, 5)
        plt.xticks(xticks)
        plt.yticks(yticks)

        plt.show()


    def visibility_model(self, title=r'$log |Vis_{Model}|$', fig_size = 6, deproject = False):
        """
        Plot the visibility model.
        Parameters
        ---------
        title: str, optional
            Title of the plot.
        fig_size: float, optional
            Size of the figure. Default is 6.
        deproject: bool, optional
            Whether to deproject the u and v coordinates. Default is False.
        """
        
        frank2d = self._frank2d
        Nx, Ny = self._Nx, self._Ny

        vis_model = frank2d.sol_visibility.reshape(Nx, Ny)
        
        u, v = frank2d._FT._Un, frank2d._FT._Vn

        u_shifted, v_shifted = np.fft.fftshift(u.reshape(Nx, Ny)), np.fft.fftshift(v.reshape(Nx, Ny))
        vis_shifted = np.fft.fftshift(vis_model)

        if deproject: 
            u_shifted, v_shifted, _ = self._Geometry.deproject(u_shifted.flatten(), v_shifted.flatten())

        plt.figure(figsize = (fig_size, fig_size))
        plt.pcolormesh(v_shifted,
                       u_shifted,
                       np.log(np.abs(vis_shifted)),
                       cmap="viridis", vmin=-12, vmax=-2)
        plt.xlabel(r'u [ $\lambda$]')
        plt.ylabel(r'v [ $\lambda$]')
        plt.gca().set_aspect('equal') 
        cmap = plt.colorbar(shrink=0.8)
        cmap.set_label(r'V [Jy]', size=12)
        plt.title(r'log|$Vis_{model}$|')
        plt.show()


    def visibility_gridded_input(self, title=r'$log |Vis_{gridded input}|$', fig_size = 6, deproject = False):
        """
        Plot the gridded input visibilities.
        Parameters
        ---------
        title: str, optional
            Title of the plot.
        fig_size: float, optional
            Size of the figure. Default is 6.
        deproject: bool, optional
            Whether to deproject the u and v coordinates. Default is False.
        """
        frank2d = self._frank2d
        Nx, Ny = self._Nx, self._Ny
        
        u_gridded, v_gridded = self._u_gridded, self._v_gridded
        vis_gridded, weights_gridded = self._vis_gridded, self._weights_gridded
        
        u_shifted, v_shifted = np.fft.fftshift(u_gridded.reshape(Nx, Ny)), np.fft.fftshift(v_gridded.reshape(Nx, Ny))
        vis_shifted = np.fft.fftshift(vis_gridded.reshape(Nx, Ny))

        if deproject: 
            u_shifted, v_shifted, _ = self._Geometry.deproject(u_shifted.flatten(), v_shifted.flatten())

        plt.figure(figsize = (fig_size, fig_size))
        plt.pcolormesh(v_shifted,
                       u_shifted,
                       np.log(np.abs(vis_shifted)),
                       cmap="viridis", vmin=-12, vmax=-2)
        plt.xlabel(r'u [1e6 $\lambda$]')
        plt.ylabel(r'v [1e6 $\lambda$]')
        plt.title(r'log |$Vis_{Input}$|')
        cmap = plt.colorbar()
        cmap.set_label(r'V [Jy]', size=12)