import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import matplotlib.colors as colors

from ..constants import rad_to_arcsec, deg_to_rad
from ..logger import Logger

# frank1d utilities
from frank.utilities import UVDataBinner
from frank.utilities import convolve_profile

from astropy.io import fits
from gofish import imagecube
from .posterior_optimization import MAPEstimator

"""
This module contains classes for plotting the results of Frank's 2D algorithm.
"""

# Matplotlib settings for LaTeX rendering.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{bm}",
})

class Plot(object):
    def __init__(self, Frank2D, Geometry, verbose = False):
        """
        Class for plotting the results of Frank2D.
        Parameters
        ----------
        Frank2D : Frank2D object
            The Frank2D object containing the results of the model.
        Geometry : Geometry object
            The Geometry object for the source.
            This is necessary to apply phas shifts and deprojections.
        """
        self._Frank2D = Frank2D
        self._Geometry = Geometry
        self._FT = Frank2D._FT
        self._Nx = self._Ny = Frank2D._N
        
        self._u_input = Frank2D.gridded_data['u']
        self._v_input = Frank2D.gridded_data['v']
        self._vis_input = Frank2D.gridded_data['vis']
        self._weights_input = Frank2D.gridded_data['weights']

        self._u_model = self._Frank2D.u_grid
        self._v_model = self._Frank2D.v_grid
        self._u_model_1d = self._Frank2D.u
        self._v_model_1d = self._Frank2D.v


        self._x_model = self._Frank2D.x_grid*rad_to_arcsec
        self._y_model = self._Frank2D.y_grid*rad_to_arcsec
        self._x_model_1d = self._Frank2D.x*rad_to_arcsec
        self._y_model_1d = self._Frank2D.y*rad_to_arcsec

        self._int_model_shifted = None

        self._f1d_sol = None
        self._MAPEstimator = None
        self._fits_file = None

        self._Rmax = Frank2D.Rmax

        self._results = {}
        self._verbose = verbose
        self.show = Logger(self._verbose)

    def set_result(self, key, value):
        """
        Set a result in the results dictionary.
        Parameters
        ----------
        key : str
            The key for the result.
        value : any
            The value of the result.
        """
        self._results[key] = value
        
    def visibility(self,
                   kind = 'model',
                   title=r'$V_{model}^{F2D}$',
                   fig_size = 6, zoom = 1,
                   vmin = -10, vmax = -2,
                   phase_shift = True, deproject = False,
                   ax = None , label_size = 14, title_size = 20,
                   tick_label_size = 13, nbins_ticks = 5,
                   save_dir = None, xlims = None, ylims = None):
        """
        Plot the visibility in the uv-plane.
        Parameters
        ----------
        kind : str, optional
            The type of visibility to plot.
            It can be 'input' or 'model'. Default is 'model'.
        title : str, optional
            The title of the plot. Default is r'$V_{model}^{F2D}$'.
        fig_size : int, optional
            The size of the figure. Default is 6.
         zoom : float, optional
            The zoom factor for the plot. Default is 1.
        vmin : float, optional
            The minimum value for the color scale in logaritmic scale. Default is -10.
        vmax : float, optional
            The maximum value for the color scale in logaritmic scale. Default is -2.
        phase_shift : bool, optional
            Whether to apply the phase shift to the visibilities. Default is True.
        deproject : bool, optional
            Whether to deproject the visibilities. Default is False.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes will be created. Default is None.
        label_size : int, optional
            The size of the axis labels. Default is 14.
        title_size : int, optional
            The size of the title. Default is 20.
        tick_label_size : int, optional
            The size of the tick labels. Default is 13.
        nbins_ticks : int, optional
            The number of bins for the ticks. Default is 5.
        save_dir : str, optional
            The directory to save the figure. If None, the figure will not be saved. Default is None.
        xlims : tuple, optional
            The limits for the x-axis. If None, it will be set automatically. Default is None.
        ylims : tuple, optional
            The limits for the y-axis. If None, it will be set automatically. Default is None.
        """
        if zoom <= 0:
            self.show.error("zoom must be > 0")
        
        Nx, Ny = self._Nx, self._Ny
        geom = self._Geometry
        vis = None
        f2d = self._Frank2D

        if kind == 'input':
            vis = self._vis_input.reshape((Ny, Nx), order='C')
            u = self._u_input.reshape((Ny, Nx), order='C')
            v = self._v_input.reshape((Ny, Nx), order='C')
        elif kind == 'model':
            vis = f2d.visibility_model
            u, v = self._u_model, self._v_model
        else:
            self.show.error("type must be 'input' or 'model'")
        
        # Shifted already.
        # The signs are because convention: East of North.
        u = -u
        v = -v
        if phase_shift:
            vis = geom.apply_phase_shift(u, v, vis)
        
        if deproject: 
            ud, vd, _ = geom.deproject(u, v)
            u, v = ud, vd

        if ax is None:
                fig = plt.figure(figsize = (fig_size, fig_size))
                ax = fig.add_subplot(111)
                show_plot = True
        else:
            show_plot = False

        self._visibility2d = vis
        
        mesh = ax.pcolormesh(u,
                            v,
                            np.log(np.abs(self._visibility2d)),
                            cmap="magma",
                            vmin=vmin, vmax=vmax,
                            rasterized=True,
                            shading='auto')
        
        ax.set_xlabel(r'u [$\lambda$]', size=label_size)
        ax.set_ylabel(r'v [$\lambda$]', size=label_size)
        ax.set_title(title, size=title_size)
        
        cmap = plt.colorbar(mesh, ax=ax, shrink=0.8)
        cmap.set_label(r'log$\|V\|$ [Jy]', size=label_size)
        cmap.ax.tick_params(labelsize=tick_label_size)

        if xlims is not None:
            ax.set_xlim(xlims)
        else:
            ax.set_xlim(u.max()/zoom, u.min()/zoom)
        if ylims is not None:
            ax.set_ylim(ylims)
        else:
            ax.set_ylim(v.max()/zoom, v.min()/zoom)

        ax.tick_params(axis='both',
               which='both',
               labelsize=tick_label_size,
               length=5,
               width=2)
    
        ax.locator_params(axis='both', nbins=nbins_ticks)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
        ax.set_aspect(1)
        ax.invert_yaxis()

        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
    
    def intensity(self,
                  title= r'$I_{model}^{F2D}$',
                  fig_size = 6, zoom = 1,
                  vmin = 0, vmax = 4e10, gamma = 0.45, norm = None,
                  phase_shift = True, deproject = False,
                  ax = None , label_size = 14, title_size = 20,
                  tick_label_size = 13, nbins_ticks = 5,
                  save_dir = None):
        """
        Plot the intensity in the xy-plane.
        Parameters
        ----------
        title : str, optional
            The title of the plot. Default is r'$I_{model}^{F2D}$'.
        fig_size : int, optional
            The size of the figure. Default is 6.
        zoom : float, optional
            The zoom factor for the plot. Default is 1.
        vmin : float, optional
            The minimum value for the color scale. Default is 0.
        vmax : float, optional
            The maximum value for the color scale. Default is 4e10.
        gamma : float, optional
            The gamma value for the PowerNorm. Default is 0.45.
        norm : matplotlib.colors.Normalize
            The normalization for the color scale. If None, a PowerNorm will be used. Default is None.
        phase_shift : bool, optional
            Whether to apply the phase shift to the visibilities. Default is True.
        deproject : bool, optional
            Whether to deproject the coordinates. Default is False.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure and axes will be created. Default is None.
        label_size : int, optional
            The size of the axis labels. Default is 14.
        title_size : int, optional
            The size of the title. Default is 20.
        tick_label_size : int, optional
            The size of the tick labels. Default is 13.
        nbins_ticks : int, optional
            The number of bins for the ticks. Default is 5.
        save_dir : str, optional
            The directory to save the figure. If None, the figure will not be saved. Default is None.
        """
        
        if zoom <= 0:
            self.show.error("zoom must be > 0")
        
        Nx, Ny = self._Nx, self._Ny
        f2d = self._Frank2D
        geom = self._Geometry
        
        I = f2d.intensity_model.real

        u_grid = self._u_model #(N, N)
        v_grid = self._v_model
        if phase_shift:
            # Only in this scheme (East of North) makes sense to do the phase shifting.
            vis_ = f2d.visibility_model
            vis = geom.apply_phase_shift(-u_grid, -v_grid, vis_)
            I = f2d.transform(vis).real

        self._intensity2d = I
        
        # The signs are because the convention: East of North.
        x = -self._x_model
        y = -self._y_model
        if deproject:
            xd, yd = geom.deproject_xy(x, y)
            x, y = xd, yd

        if ax is None:
            fig = plt.figure(figsize = (fig_size, fig_size))
            ax = fig.add_subplot(111)
            show_plot = True
        else:
            show_plot = False

        if norm is None:
            norm = colors.PowerNorm(gamma = gamma, vmin = vmin, vmax = vmax)
        elif not isinstance(norm, colors.Normalize):
            self.show.error("norm must be an instance of matplotlib.colors.Normalize")
        
        mesh = ax.pcolormesh(x,
                             y,
                             self._intensity2d ,
                             cmap="magma",
                             norm=norm,
                             rasterized=True,
                             shading='auto')
            
        ax.set_xlabel(r'RA ["]', size=label_size)
        ax.set_ylabel(r'Dec ["]', size=label_size)
        ax.set_title(title, size=title_size)
        
        cmap = plt.colorbar(mesh, ax=ax, shrink=0.8)
        cmap.set_label(r'I [Jy/sr]', size=label_size)
        cmap.ax.tick_params(labelsize=tick_label_size)
        
        ax.set_xlim(x.max()/zoom, x.min()/zoom)
        ax.set_ylim(y.max()/zoom, y.min()/zoom)
        ax.tick_params(axis='both',
               which='both',
               labelsize=tick_label_size,
               length=5,
               width=2)
        ax.locator_params(axis='both', nbins=nbins_ticks)
        ax.set_aspect(1)
        ax.invert_yaxis()

        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()

    def get_profile(self, x1, x2, f, bins, weighted = False, weights = None, fit_1d = False):
        """
        Get the 1D profile of a 2D function by binning the values in annuli.
        Parameters
        ----------
        x1 : array-like
            The x-coordinates of the points.
        x2 : array-like
            The y-coordinates of the points.
        f : array-like
            The values of the function at the points.
        bins : array-like
            The edges of the bins for the annuli.
        weighted : bool, optional
            Whether to weight the values by the weights. Default is False.
        weights : array-like, optional
            The weights to use if weighted is True. Default is None.
        fit_1d : bool, optional
            Whether to fit a 1D profile instead of binning. Default is False.
        Returns
        -------
        r_1d : array-like
            The radii of the bins.
        f_1d : array-like
            The values of the function in the bins.
        """
        from scipy.stats import binned_statistic
        
        if not fit_1d:
            r = np.hypot(x1, x2)
            r = r.ravel(order = 'C')
            f = f.ravel(order = 'C')
        else:
            r = x1

        if weighted:
            if weights is None:
                self.show.error("Add weights.")
            weights_gridded = self._weights_input
            F_W_binned, bin_edges, _ = binned_statistic(r, f*weights_gridded, 'sum', bins = bins)
            Weights_binned, bin_edges, _ = binned_statistic(r, weights_gridded, 'sum', bins = bins)

            f_1d = np.nan_to_num(F_W_binned, nan=0)
            f_1d = f_1d/Weights_binned
        else:
            f_1d, bin_edges, _ = binned_statistic(r, f, 'mean', bins = bins)
        
        r_1d = (bin_edges[:-1] + bin_edges[1:]) / 2
            
        return r_1d, f_1d

    def edges(self, x):
        """
        Get the edges of the bins for the binned statistic.
        Parameters
        ----------
        x : array-like
            The bin centers.
        Returns
        -------
        edges : array-like
            The edges of the bins.
        """
        mids = (x[1:] + x[:-1]) / 2.0
        first = x[0]  - (x[1] - x[0]) / 2.0
        last  = x[-1] + (x[-1] - x[-2]) / 2.0
        return np.r_[first, mids, last]

    def visibility_profile( self, 
                            title = r'Visibility Profile',
                            fig_size = (10,3),
                            input = None, frank1d = False, weighted = True,
                            bins = 300,
                            phase_shift = True, deproject = True):
        """
        Plot the visibility profile as a function of the baseline.
        Parameters
        ----------
        title : str, optional
            The title of the plot. Default is r'$Visibility_{Model}$'.
        fig_size : tuple, optional
            The size of the figure. Default is (10,3).
        input : dict, optional
            The input data to plot. It must contain the keys 'u', 'v', 'vis', 'weights'.
            If None, it will use the gridded data from the Frank2D object. Default is None.
        frank1d : bool, optional
            Whether to plot the Frank1D profile. Default is False.
        weighted : bool, optional
            Whether to weight the binned statistic by the input weights. Default is True.
        bins : int, optional
            The number of bins for the binned statistic. Default is 300.
        phase_shift : bool, optional
            Whether to apply the phase shift to the visibilities. Default is True.
        deproject : bool, optional
            Whether to deproject the coordinates. Default is True.
        """
        f2d = self._Frank2D
        geom = self._Geometry
        # model
        u_model = self._u_model
        v_model = self._v_model
        vis_model = f2d.visibility_model
        
        u_model_1d = self._u_model_1d
        v_model_1d = self._v_model_1d

        # input
        u_input_g = self._u_input
        v_input_g = self._v_input
        vis_input_g = self._vis_input
        weights_input_g = self._weights_input
        
        if input is not None:
            u_input = input['u']
            v_input = input['v']
            vis_input = input['vis']
            weights_input = input['weights']
        else:
            u_input = u_input_g
            v_input = v_input_g
            vis_input = vis_input_g
            weights_input = weights_input_g
            
        if phase_shift:
            # is necessary to be done in the East of North convention.
            vis_input = geom.apply_phase_shift(-u_input, -v_input, vis_input)
            vis_model = geom.apply_phase_shift(-u_model, -v_model, vis_model)
            
        if deproject:
            u_model, v_model, _ = geom.deproject(u_model, v_model)
            u_model_1d, v_model_1d, _ = geom.deproject(u_model_1d, v_model_1d)
            u_input, v_input, _ = geom.deproject(u_input, v_input)

        # collocation points for the plot.
        q = np.unique(np.hypot(u_model_1d, v_model_1d))
        q = np.sort(q)[1:]
        q = np.geomspace(q[0], q[-1], bins)
        edges = self.edges(q)

        # f1d
        if frank1d:
            if self._f1d_sol is None:
                self.show.warning("Frank1D profile not set. "
                                  " Using gridded visibilities as input. "
                                  " And default parameters of frank1d method of Frank2D Class.\n "
                                  " Use the 'set_f1d_solution' method to set the Frank1D solution you want to be plotted."
                                  )

                data = {
                    'u': u_input_g,
                    'v': v_input_g,
                    'vis': vis_input_g,
                    'weights': weights_input_g
                }
                
                sol = f2d.frank1d(data = data, n_pts = self._Nx, geom = geom)
                self.set_f1d_solution(sol)

            if deproject:
                vis_f1d = self._f1d_sol.predict_deprojected(q = q)
            else:
                vis_f1d = self._f1d_sol._vis_map.predict_visibilities(self.f1d_sol.mean, q, q*0, geometry=geom )


        q_model_1d, vis_model_1d = self.get_profile(u_model, v_model, vis_model,
                                                    bins = edges,
                                                    weighted = weighted, weights = weights_input)

        # ----------------
        baselines = np.hypot(u_input, v_input)         
        grid = np.logspace(np.log10(min(baselines.min(), baselines[0])),
                                   np.log10(max(baselines.max(), baselines[-1])),
                                   10**4)
        
        
        plt.figure(figsize=fig_size)
        # Raw visibilities ---------------------------------------------------------
        cs, ms = ['#a4a4a4', 'k'], ['.', 'x']
        bin_widths = [1e3, 1e5]

        binned_vis0 = UVDataBinner(baselines, vis_input, weights_input, bin_widths[0])
        plt.plot(binned_vis0.uv, np.abs(binned_vis0.V), c=cs[0],
                     marker=ms[0], ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[0]/1e3))
        
        binned_vis1 = UVDataBinner(baselines, vis_input, weights_input, bin_widths[1])
        plt.plot(binned_vis1.uv, np.abs(binned_vis1.V), c=cs[1],
                     marker=ms[1], ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[1]/1e3))
    
        # frank1D -------------------------------------------------------------------
        if frank1d:
            plt.plot(q, np.abs(vis_f1d), color = "red", label = r'frank1d')
    
        # frank2D -------------------------------------------------------------------
        plt.plot(q, np.abs(vis_model_1d), color = 'blue', ls ='--', label = r'frank2d')
    
        plt.xlabel(r'baseline [$\lambda$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-5, 0)
        plt.xlim(1e5, 6e6)
        plt.ylabel(r'$\|V\|$ [Jy]', size = 10)
        plt.title(title+', N = ' + str(self._Nx))
        plt.legend(fontsize= 10, loc = 'best')
        plt.show()

    def calculate_resolution(self, clean_beam, intrinsic_resolution):
        """
        Calculate the beam to convolve intrinsic resolution with,
        to obtain the CLEAN beam desired final resolution.
        Parameters
        ----------
        clean_beam : dict
            Dictionary containing the CLEAN beam parameters: 'bmaj', 'bmin', 'beam_pa'.
        intrinsic_resolution : float
            The intrinsic resolution to convolve with the CLEAN beam, in arcseconds.
        Returns
        -------
        final_resolution : dict
            Dictionary containing the final beam parameters: 'bmaj', 'bmin', 'beam_pa'.
        """
        bmaj_as = clean_beam['bmaj']
        bmin_as = clean_beam['bmin']
        final_bmaj_as = np.sqrt(bmaj_as**2 - intrinsic_resolution**2)
        final_bmin_as = np.sqrt(bmin_as**2 - intrinsic_resolution**2)
        final_resolution = {'bmaj': final_bmaj_as, 'bmin': final_bmin_as, 'beam_pa': clean_beam['beam_pa']}
        return final_resolution

    def set_f1d_solution(self, sol):
        """Setter the Frank1D solution to be plotted in the intensity profile.
        Parameters
        ----------
        sol : Frank1D solution
            The Frank1D solution to be plotted in the intensity profile.
        """
        self._f1d_sol = sol
    
    def set_fits_file(self, fits_file):
        """
        Setter the FITS file to be used for the CLEAN profile in the intensity profile.
        Parameters
        ----------
        fits_file : str
            The path to the FITS file to be used for the CLEAN profile in the intensity profile.
        """
        self._fits_file = fits_file

    def intensity_profile(self, title= r'Brightness profile', 
                          clean = False,
                          frank1d = False,
                          bins = 300, Rmax = None,
                          f2d_fwhm = 0, f1d_fwhm = 0, 
                          log_scale = True, fig_size = (10,3),
                          x_lims = None, y_lims = None, 
                          save_fig = False):
        """
        Plot the intensity profile as a function of the radius.
        Parameters
        ----------
        title : str, optional
            The title of the plot. Default is r'Brightness profile'.
        clean : bool, optional
            Whether to plot the CLEAN profile. Default is False.
        frank1d : bool, optional
            Whether to plot the Frank1D profile. Default is False.
        bins : int, optional
            The number of bins for the binned statistic. Default is 300.
        Rmax : float, optional
            The maximum radius to plot. If None, it will be set to half of the maximum radius in the model. Default is None.
        f2d_fwhm : float, optional
            The intrinsic resolution of the Frank2D model to convolve with the CLEAN beam, in arcseconds. Default is 0 (no convolution).
        f1d_fwhm : float, optional
            The intrinsic resolution of the Frank1D model to convolve with the CLEAN beam, in arcseconds. Default is 0 (no convolution).
        log_scale : bool, optional
            Whether to plot the y-axis in log scale. Default is True.
        fig_size : tuple, optional
            The size of the figure. Default is (10,3).
        x_lims : tuple, optional
            The limits for the x-axis. If None, it will be set to (0, 0.8*Rmax). Default is None.
        y_lims : tuple, optional
            The limits for the y-axis. If None, it will be set to (1e8, 1e11) for the non-clean case, and (1e-6, 5e-3) for the clean case. Default is None.
        save_fig : bool, optional
            Whether to save the figure as 'intensity_profile.png'. Default is False.
        """

        f2d = self._Frank2D
        geom = self._Geometry
        inc, pa, dra, ddec = geom.inc, geom.pa, geom.dra, geom.ddec

        if clean:
            if self._fits_file is None:
                self.show.error(
                    "CLEAN: Data location for the FITS file not provided.\n"
                    "Use the 'set_fits_file' method.")
            else:
                cube_1mm = imagecube(self._fits_file)
                x_clean, y_clean, dy_clean = cube_1mm.radial_profile(inc= inc, PA=pa, x0=dra, y0=ddec)
                
                fits_image = fits.open(self._fits_file)
                header = fits_image[0].header
                bmaj_deg = float(header["BMAJ"])
                bmin_deg = float(header["BMIN"])
                bmaj_as = bmaj_deg * 3600.0
                bmin_as = bmin_deg * 3600.0
                bpa_deg = header['BPA']
                
                clean_beam = {'bmaj': bmaj_as, 'bmin': bmin_as, 'beam_pa': bpa_deg}
                clean_area = clean_beam['bmaj']*clean_beam['bmin']*np.pi/4./np.log(2.)*(1/rad_to_arcsec)**2
                self.show.info(f"+ FWHM CLEAN beam: {bmaj_as} arcsec x {bmin_as} arcsec.")

                # Decide the beam to convolve with.
                if f2d_fwhm > 0:
                    self.show.info(f"+ FWHM frank2d: {f2d_fwhm} arcsec.")
                    beam_for_f2d = self.calculate_resolution(clean_beam, f2d_fwhm)
                else: 
                    beam_for_f2d = clean_beam
                
                if f1d_fwhm > 0 and frank1d:
                    self.show.info(f"+ FWHM frank1d: {f1d_fwhm} arcsec.")
                    beam_for_f1d = self.calculate_resolution(clean_beam, f1d_fwhm)
                else:
                    beam_for_f1d = clean_beam

        # model
        u_model = self._u_model
        v_model = self._v_model
        vis_model = f2d.visibility_model
        
        u_model_1d = self._u_model_1d
        v_model_1d = self._v_model_1d

        x_model = self._x_model
        y_model = self._y_model

        x_model_1d = self._x_model_1d
        y_model_1d = self._y_model_1d

        u_input_g = self._u_input
        v_input_g = self._v_input
        vis_input_g = self._vis_input
        weights_input_g = self._weights_input
        
        # phase shift
        vis = geom.apply_phase_shift(-u_model, -v_model, vis_model) # the East of North convention.
        I = f2d.transform(vis).real
        vis_model = f2d.visibility_model

        # deproject
        x_model, y_model = geom.deproject_xy(x_model, y_model)
        x_model_1d_d, y_model_1d_d = geom.deproject_xy(x_model_1d, y_model_1d)

        # collocation points for the plot.
        r = np.unique(np.hypot(x_model_1d_d, y_model_1d_d))
        r = np.linspace(r.min(), r.max(), bins)
        edges = self.edges(r)

        # frank1d
        if frank1d:
            if self._f1d_sol is None:
                self.show.warning(
                    "Frank1D profile not set."
                    "Using gridded visibilities as input"
                    "And default parameters of frank1d method of Frank2D Class.\n"
                    "Use the 'set_f1d_solution' method to set the Frank1D solution to be plotted.")

                data = {
                    'u': u_input_g,
                    'v': v_input_g,
                    'vis': vis_input_g,
                    'weights': weights_input_g
                    }
            
                sol = f2d.frank1d(data = data, n_pts = self._Nx, geom = geom)
                self.set_f1d_solution(sol)

            r_f1d, I_f1d = self._f1d_sol.r, self._f1d_sol.mean

        r_model_1d, I_model_1d = self.get_profile(x_model, y_model, I, bins = edges)
        I_model_1d = np.nan_to_num(I_model_1d, nan=0)
        
        Rmax = self._Rmax/2 if Rmax is None else Rmax

        if x_lims is None:
            x_lims = (0, 0.8*Rmax)
        if y_lims is None:
            y_lims = (1e8, 1e11)
            if clean:
                y_lims = (1e-6, 5e-3)

        plt.figure(figsize=fig_size)
        if clean:
            area = clean_area
            #  CLEAN --------------------------------------------------------------------
            plt.plot(x_clean, y_clean, "black", label = "CLEAN")
            plt.fill_between(x_clean, y_clean-dy_clean, y_clean+dy_clean, alpha=0.7)
            # frank2D -------------------------------------------------------------------
            I_model_1d_convolved = convolve_profile(r_model_1d, I_model_1d, inc, pa, beam_for_f2d)
            I_model_1d_convolved *= area # Jy/beam
            plt.plot(r_model_1d, I_model_1d_convolved, color = 'blue',ls ='--', label = r'frank2d')
            
            if frank1d:
                # frank1D ---------------------------------------------------------------
                I_f1d_convolved = convolve_profile(r_f1d, I_f1d, inc, pa, beam_for_f1d)
                I_f1d_convolved *= area # Jy/beam
                plt.plot(r_f1d, I_f1d_convolved, color = "red", label = r'frank1d')
                self.set_result('f1d_profile', {'r': r_f1d, 'I': I_f1d_convolved})
        
            self.set_result('clean_profile', {'r': x_clean, 'I': y_clean, 'dI': dy_clean})
            self.set_result('f2d_profile', {'r': r_model_1d, 'I': I_model_1d_convolved})

            plt.ylabel(r'log($I_\nu$) [Jy/beam]', size=10)
            plt.xlabel('Radius ["]', size=10)
            plt.legend(fontsize=12)
            if log_scale:
                plt.yscale('log')
            plt.ylim(y_lims)
            plt.xlim(x_lims)
            if save_fig:
                plt.savefig('intensity_profile.png')
            plt.show()
        else:
            # frank1D -------------------------------------------------------------------
            if frank1d:
                plt.plot(r_f1d, I_f1d, color = "red", label = r'frank1d')
                self.set_result('f1d_profile', {'r': r_f1d, 'I': I_f1d})
    
            # frank2D -------------------------------------------------------------------
            plt.plot(r_model_1d, I_model_1d, color = 'blue', ls ='--', label = r'frank2d')
            self.set_result('f2d_profile', {'r': r_model_1d, 'I': I_model_1d})

            plt.xlabel('Radius ["]', size = 10)
            plt.ylabel(r'log($I_\nu$) [$10^{10}$ Jy sr$^{-1}$]', size = 10)
            plt.title(title)
            plt.xlim(x_lims)
            plt.ylim(y_lims)
            if log_scale:
                plt.yscale('log')
            plt.legend()
            if save_fig:
                plt.savefig('intensity_profile.png')
            plt.show()


    def stats_optimization(self, MAP_estimator = None):
        r"""
        Plot the statistics of the posterior optimization.
        Params
        ------
        MAP_estimator: MAPEstimator object, optional
            The MAPEstimator object used in the optimization.
            If not provided, it will use the one from the Frank2D object.
        """
        f2d = self._Frank2D
        if MAP_estimator is None:
            if f2d._MAPEstimator is None:
                self.show.error("MAPEstimator object not provided.")
            self._MAPEstimator = f2d._MAPEstimator
        else:
            if isinstance(MAP_estimator, MAPEstimator) is False:
                self.show.error("MAP_estimator must be an instance of MAPEstimator class.")
            self._MAPEstimator = MAP_estimator
        
        ME = self._MAPEstimator
        MAP = ME.MAP
        m = MAP['m']
        c = MAP['c']
        l = MAP['l']

        iterations = np.arange(1, len(ME._minus_log_posteriors) + 1)

        fig, axs_grid = plt.subplots(3, 3, figsize=(8, 7))
        gs = axs_grid[0, 0].get_gridspec()

        for ax in axs_grid[0, :]:
            ax.remove()

        ax_main = fig.add_subplot(gs[0, :])

        axs = [ax_main] + list(axs_grid[1, :]) + list(axs_grid[2, :])

        axs[0].plot(iterations, ME._minus_log_posteriors)
        axs[0].set_title(r"- logP($\theta | V_{obs}$)")
        axs[0].grid()
        axs[0].set_xlabel("Iterations")

        axs[1].plot(iterations, ME._jDjs)
        axs[1].set_title(r"-$j^T$Dj")
        axs[1].grid()
        axs[1].set_xlabel("Iterations")

        axs[2].plot(iterations, ME._logdetDs)
        axs[2].set_title(r"-log$|D|$")
        axs[2].grid()
        axs[2].set_xlabel("Iterations")

        axs[3].plot(iterations, ME._logdetSs)
        axs[3].set_title(r"log$|S|$")
        axs[3].grid()
        axs[3].set_xlabel("Iterations")

        axs[4].plot(iterations, ME._ms, label="m")
        axs[4].set_title("m")
        axs[4].axhline(m, color='red', ls='--', label='MAP')
        axs[4].grid()
        axs[4].set_xlabel("Iterations")
        axs[4].legend(loc="best")

        axs[5].plot(iterations, ME._cs, label="c")
        axs[5].set_title("log(c)")
        axs[5].set_xlabel("Iterations")
        axs[5].grid()
        axs[5].axhline(c, color='red', ls='--', label='MAP')
        axs[5].set_yscale("log")
        axs[5].legend(loc="best")

        axs[6].plot(iterations, ME._ls, label="l")
        axs[6].set_title("log(l)")
        axs[6].set_xlabel("Iterations")
        axs[6].axhline(l, color='red', ls='--', label='MAP')
        axs[6].grid()
        axs[6].set_yscale("log")
        axs[6].legend(loc="best")

        for ax in axs[7:]:
            fig.delaxes(ax)

        fig.suptitle('Optimization stats', fontsize=15)

        fig.tight_layout()
        plt.show()
    
    def power_spectrum(self, data, MAP_estimator = None, m = None, c = None,
                        fig_size = (7,2), title = "Power spectrum", title_size = 10,
                        ylim_log = (-9, 0), xlim_log = (5, 6.5)):
        r"""
        Plot the power spectrum of the best parameters found in the posterior optimization.
        Params
        ------
        data: dict
            Dictionary with the observed data. 
            It must contain the keys 'u', 'v', 'vis', 'weights'.
        MAP_estimator: MAPEstimator object, optional
            The MAPEstimator object used in the optimization.
            If not provided, it will use the one from the Frank2D object.
        m: float, optional
            The slope of the power spectrum.
        c: float, optional
            The normalization of the power spectrum.
        returns
        -------
        A plot of the power spectrum.
        """
        f2d = self._Frank2D

        if m is None and c is None:
            if MAP_estimator is None:
                if f2d._MAPEstimator is None:
                    self.show.error("MAPEstimator object not provided.")
                self._MAPEstimator = f2d._MAPEstimator
            else:
                if isinstance(MAP_estimator, MAPEstimator) is False:
                    self.show.error("MAP_estimator must be an instance of MAPEstimator class.")
                self._MAPEstimator = MAP_estimator
                m = self._MAPEstimator.MAP['m']
                c = self._MAPEstimator.MAP['c']

        # Plot visibilities gridded.
        from frank.utilities import UVDataBinner       
        bin_widths=[1e3, 1e5]

        u, v = data['u'], data['v']
        Vis = data['vis']
        Weights = data['weights']

        baselines = np.hypot(u, v)
        binned_vis = UVDataBinner(baselines, Vis, Weights, bin_widths[0])
        logx = np.log10(binned_vis.uv)
        logy = np.log10(np.abs(binned_vis.V)**2)

        binned_vis2 = UVDataBinner(baselines, Vis, Weights, bin_widths[1])
        logx2 = np.log10(binned_vis2.uv)
        logy2 = np.log10(np.abs(binned_vis2.V)**2)

        def linear_power_spectrum(logx, m, c):
            return m*logx + np.log10(c)

        def power_spectrum(baselines, m, c):
            return c * baselines**m

        P = power_spectrum(baselines, m, c)

        lgx = np.log10(np.geomspace(np.sort(baselines)[1], baselines.max(), 1000))
        lgy = linear_power_spectrum(lgx, m, c)

        cs, ms = ['#a4a4a4', 'k'], ['.', 'x']

        plt.figure(figsize=fig_size)
        plt.plot(logx, logy, c=cs[0], marker=ms[0], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[0]/1e3))
        plt.plot(logx2, logy2, c=cs[1], marker=ms[1], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[1]/1e3))
        plt.plot(lgx, lgy, ls='-', color= 'r', label='MAP, m={:.2f}, log(c)={}'.format(m, np.log10(c)))
        plt.xlabel(r'log Baseline [$\lambda$]', size = 10)
        plt.ylabel(r'log $|Vis_{Obs}|^{2}$ [Jy]', size = 10)
        plt.legend(loc = 'best', fontsize = 'x-small')
        plt.title(title, size = title_size)
        plt.ylim(ylim_log)
        if xlim_log is not None:
            plt.xlim(xlim_log)
        else:
            plt.xlim(np.min(logx), np.max(logx))
        plt.show()

    def cg_tolerance(self, fig_size = (7,2)):
        r"""
        Plot the conjugate gradient tolerance over iterations.
        """
        f2d = self._Frank2D
        tols = f2d.solver.fit_data['tols']

        iterations = range(len(tols))
        tols = np.array(tols)

        plt.figure(figsize = fig_size)
        plt.plot(iterations, tols)
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('log tolerance')
        plt.show()

    @property
    def I_shifted(self):
        """Get the intensity model with the phase shift applied."""
        if self._int_model_shifted is None:
            f2d = self._Frank2D
            geom = self._Geometry

            u_model = self._u_model
            v_model = self._v_model
            vis_model = f2d.visibility_model

            vis = geom.apply_phase_shift(-u_model, -v_model, vis_model) # the East of North convention.
            self._int_model_shifted = f2d.transform(vis).real
        return self._int_model_shifted
    
    @property
    def x_1d(self):
        """Getter for the 1D x-coordinates of the model."""
        return self._x_model_1d
    
    @property
    def y_1d(self):
        """Getter for the 1D y-coordinates of the model."""
        return self._y_model_1d
    
    @property
    def x_2d(self):
        """Getter for the 2D x-coordinates of the model."""
        return self._x_models
    
    @property
    def y_2d(self):
        """Getter for the 2D y-coordinates of the model."""
        return self._y_model

    @property
    def Nx(self):
        """Getter for the number of x-coordinates in the model."""
        return self._Nx
       
    @property
    def Ny(self):
        """Getter for the number of y-coordinates in the model."""
        return self._Ny

    @property
    def results(self):
        """Getter for the results of the plots."""
        return self._results