import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import matplotlib.colors as colors

from .constants import rad_to_arcsec, deg_to_rad

# frank1d utilities
from frank.utilities import UVDataBinner
from frank.utilities import convolve_profile

from astropy.io import fits
from gofish import imagecube
from .posterior_optimization_gpu import MAPEstimator

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
    def __init__(self, Frank2D, Geometry):
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
        
        self._u_input = Frank2D.gridded_data['u'].get()
        self._v_input = Frank2D.gridded_data['v'].get()
        self._vis_input = Frank2D.gridded_data['vis'].get()
        self._weights_input = Frank2D.gridded_data['weights'].get()

        self._u_model = Frank2D.u_grid.get()
        self._v_model = Frank2D.v_grid.get()
        self._u_model_1d = Frank2D.u.get()
        self._v_model_1d = Frank2D.v.get()
        self._vis_model = Frank2D.visibility_model.get()

        self._x_model = (Frank2D.x_grid*rad_to_arcsec).get()
        self._y_model = (Frank2D.y_grid*rad_to_arcsec).get()
        self._x_model_1d = (Frank2D.x*rad_to_arcsec).get()
        self._y_model_1d = (Frank2D.y*rad_to_arcsec).get()
        self._int_model = Frank2D.intensity_model.real.get()

        self._f1d_profile = None
        self._MAP_estimator = None
        self._fits_file = None

        self._Rmax = Frank2D.Rmax

        self._results = {}

    def to_cpu(self, array):
        if isinstance(array, np.ndarray):
            return array
        else:
            return array.get()

    def set_result(self, key, value):
        self._results[key] = value
        
    def visibility(self,
                   kind = 'model',
                   title=r'$V_{model}^{F2D}$',
                   fig_size = 6, zoom = 1,
                   vmin = -10, vmax = -2,
                   phase_shift = True, deproject = False,
                   ax = None , label_size = 14, title_size = 20,
                   tick_label_size = 13):
        
        if zoom <= 0:
            raise ValueError("zoom must be > 0")
        
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
            raise ValueError("type must be 'input' or 'model'")
        
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
                            np.log(np.abs(vis)),
                            cmap="magma",
                            vmin=vmin, vmax=vmax)
        
        ax.set_xlabel(r'u [$\lambda$]', size=label_size)
        ax.set_ylabel(r'v [$\lambda$]', size=label_size)
        ax.set_title(title, size=title_size)
        
        cmap = plt.colorbar(mesh, ax=ax, shrink=0.8)
        cmap.set_label(r'log$\|V\|$ [Jy]', size=label_size)
        cmap.ax.tick_params(labelsize=tick_label_size)

        
        ax.set_xlim(u.max()/zoom, u.min()/zoom)
        ax.set_ylim(v.max()/zoom, v.min()/zoom)

        ax.tick_params(axis='both',
               which='major',
               labelsize=tick_label_size,
               length=5,
               width=2)
        
        ax.set_aspect(1)
        ax.invert_yaxis()
        
        if show_plot:
            plt.show()
    
    def intensity(self,
                  title= r'$I_{model}^{F2D}$',
                  fig_size = 6, zoom = 1,
                  vmin = 0, vmax = 4e10, gamma = 0.45,
                  phase_shift = True, deproject = False,
                  ax = None , label_size = 14, title_size = 20,
                  tick_label_size = 13):
        
        if zoom <= 0:
            raise ValueError("zoom must be > 0")
        
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
        
        self._intensity2d = self.to_cpu(I)

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
        
        norm = colors.PowerNorm(gamma = gamma, vmin = vmin, vmax = vmax)

        mesh = ax.pcolormesh(x,
                                y,
                                I,
                                cmap="magma",
                                norm=norm)
            
        ax.set_xlabel(r'RA ["]', size=label_size)
        ax.set_ylabel(r'Dec ["]', size=label_size)
        ax.set_title(title, size=title_size)
        
        cmap = plt.colorbar(mesh, ax=ax, shrink=0.8)
        cmap.set_label(r'I [Jy/sr]', size=label_size)
        cmap.ax.tick_params(labelsize=tick_label_size)
        
        ax.set_xlim(x.max()/zoom, x.min()/zoom)
        ax.set_ylim(y.max()/zoom, y.min()/zoom)

        ax.tick_params(axis='both',
               which='major',
               labelsize=tick_label_size,
               length=5,
               width=2)
        
        ax.set_aspect(1)
        ax.invert_yaxis()
        
        if show_plot:
            plt.show()

    def get_profile(self, x1, x2, f, bins, weighted = False, weights = None, fit_1d = False):
        from scipy.stats import binned_statistic
        
        if not fit_1d:
            r = np.hypot(x1, x2)
            r = r.ravel(order = 'C')
            f = f.ravel(order = 'C')
        else:
            r = x1

        if weighted:
            if weights is None:
                raise ValueError("Add weights.")
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
        mids = (x[1:] + x[:-1]) / 2.0
        first = x[0]  - (x[1] - x[0]) / 2.0
        last  = x[-1] + (x[-1] - x[-2]) / 2.0
        return np.r_[first, mids, last]

    def visibility_profile( self, 
                            title = r'$Visibility_{Model}$',
                            fig_size = (10,3),
                            input = None, frank1d = False, weighted = True,
                            bins = 300,
                            phase_shift = True, deproject = True):

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
           if self._f1d_profile is None:
            print("Frank1D profile not set." 
                  "Using gridded visibilities as input.")

            data = {
                'u': u_input_g,
                'v': v_input_g,
                'vis': vis_input_g,
                'weights': weights_input_g
            }
            
            sol = f2d.frank1d(data = data, n_pts = self._Nx)

            if deproject:
                vis_f1d = sol.predict_deprojected(q = q)
            else:
                vis_f1d = sol._vis_map.predict_visibilities(sol.mean, q, q*0, geometry=self._Geometry )


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
        bmaj_as = clean_beam['bmaj']
        bmin_as = clean_beam['bmin']
        final_bmaj_as = np.sqrt(bmaj_as**2 - intrinsic_resolution**2)
        final_bmin_as = np.sqrt(bmin_as**2 - intrinsic_resolution**2)
        print(f"-> Final beam to convolve with: {final_bmaj_as} x {final_bmin_as} arcsec")
        final_resolution = {'bmaj': final_bmaj_as, 'bmin': final_bmin_as, 'beam_pa': clean_beam['beam_pa']}
        return final_resolution

    def set_f1d_solution(self, sol):
        self._f1d_profile = sol
    
    def set_fits_file(self, fits_file):
        self._fits_file = fits_file

    def intensity_profile(self, title= r'Brightness profile', 
                          clean = False,
                          frank1d = False,
                          bins = 300, Rmax = None,
                          resol_f2d = 0, resol_f1d = 0, 
                          log_scale = True, fig_size = (10,3),
                          x_lims = None, y_lims = None, 
                          save_fig = False):

        # TODO: add option to change figsize.
        f2d = self._Frank2D
        geom = self._Geometry
        inc, pa, dra, ddec = geom.inc, geom.pa, geom.dra, geom.ddec

        if clean:
            if self._fits_file is None:
                raise ValueError(
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
                print(f"Clean beam: {bmaj_as} x {bmin_as} arcsec")

                # Decide the beam to convolve with.
                if resol_f2d > 0:
                    print(f"FWHM for frank2d: {resol_f2d} arcsec")
                    beam_for_f2d = self.calculate_resolution(clean_beam, resol_f2d)
                else: 
                    beam_for_f2d = clean_beam
                
                if resol_f1d > 0 and frank1d:
                    print(f"FWHM for frank1d: {resol_f1d} arcsec")
                    beam_for_f1d = self.calculate_resolution(clean_beam, resol_f1d)
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
        
        # phase shift
        vis = geom.apply_phase_shift(-u_model, -v_model, vis_model) # the East of North convention.
        I = f2d.transform(vis).real
        I = self.to_cpu(I)
        self._int_model_shifted = I

        # deproject
        x_model, y_model = geom.deproject_xy(x_model, y_model)
        x_model_1d_d, y_model_1d_d = geom.deproject_xy(x_model_1d, y_model_1d)

        # collocation points for the plot.
        r = np.unique(np.hypot(x_model_1d_d, y_model_1d_d))
        r = np.linspace(r.min(), r.max(), bins)
        edges = self.edges(r)

        # frank1d
        if frank1d:
            if self._f1d_profile is None:
                raise ValueError(
                    "Set your Frank1D profile first. \n"
                    "Use the 'set_f1d_solution' method."
                    )
            r_f1d, I_f1d = self._f1d_profile.r, self._f1d_profile.mean

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
            if f2d._MAP_estimator is None:
                raise ValueError("MAPEstimator object not provided.")
            self._MAP_estimator = f2d._MAP_estimator
        else:
            if isinstance(MAP_estimator, MAPEstimator) is False:
                raise ValueError("MAP_estimator must be an instance of MAPEstimator class.")
            self._MAP_estimator = MAP_estimator
        
        ME = self._MAP_estimator

        iterations = np.arange(1, len(ME._minus_log_posteriors) + 1)

        fig, axs = plt.subplots(2, 4, figsize=(10, 5))
        axs = axs.ravel()

        axs[0].plot(iterations, ME._minus_log_posteriors)
        axs[0].set_title("- log posterior variation")
        axs[0].set_xlabel("Iterations")

        axs[1].plot(iterations, ME._jDjs)
        axs[1].set_title("-jDj term variation")
        axs[1].set_xlabel("Iterations")

        axs[2].plot(iterations, ME._logdetDs)
        axs[2].set_title("-Log|D| term variation")
        axs[2].set_xlabel("Iterations")

        axs[3].plot(iterations, ME._logdetSs)
        axs[3].set_title("Log|S| variation")
        axs[3].set_xlabel("Iterations")

        axs[4].plot(iterations, ME._ms, label="m")
        axs[4].set_title(f"m")
        axs[4].set_xlabel("Iterations")

        axs[5].plot(iterations, ME._cs, label="c")
        axs[5].set_title(f"log(c)")
        axs[5].set_xlabel("Iterations")
        axs[5].set_yscale("log")

        axs[6].plot(iterations, ME._ls, label="l")
        axs[6].set_title(f"log(l)")
        axs[6].set_xlabel("Iterations")
        axs[6].set_yscale("log")

        for ax in axs[7:]:
            fig.delaxes(ax)

        fig.suptitle(f'Optimization stats', fontsize=10)

        fig.tight_layout()
        plt.show()
    
    def power_spectrum(self, data, MAP_estimator = None, m = -2, c = 1e8):
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
                if f2d._MAP_estimator is None:
                    raise ValueError("MAPEstimator object not provided.")
                self._MAP_estimator = f2d._MAP_estimator
            else:
                if isinstance(MAP_estimator, MAPEstimator) is False:
                    raise ValueError("MAP_estimator must be an instance of MAPEstimator class.")
                self._MAP_estimator = MAP_estimator
                m = ME.MAP['m']
                c = ME.MAP['c']

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

        plt.figure(figsize=(7, 3))
        plt.plot(logx, logy, c=cs[0], marker=ms[0], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[0]/1e3))
        plt.plot(logx2, logy2, c=cs[1], marker=ms[1], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[1]/1e3))
        plt.plot(lgx, lgy, ls='-', color= 'r', label='MAP, m={:.2f}, log(c)={}'.format(m, np.log10(c)))
        plt.xlabel(r'log Baseline [$\lambda$]', size = 10)
        plt.ylabel(r'log $|Vis_{Obs}|^{2}$ [Jy]', size = 10)
        plt.legend(loc = 'best', fontsize = 'x-small')
        plt.title("Power spectrum")
        plt.ylim(-9, 0)
        plt.xlim(5, 6.5)
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
        return self._x_model_1d
    
    @property
    def y_1d(self):
        return self._y_model_1d
    
    @property
    def x_2d(self):
        return self._x_models
    
    @property
    def y_2d(self):
        return self._y_model

    @property
    def Nx(self):
        return self._Nx
       
    @property
    def Ny(self):
        return self._Ny

    @property
    def results(self):
        return self._results