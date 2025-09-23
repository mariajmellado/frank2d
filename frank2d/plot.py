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

"""
This module contains classes for plotting the results of Frank's 2D algorithm.
"""

class Plot(object):
    def __init__(self, Frank2D, Geometry):
        self._frank2d = Frank2D
        self._Geometry = Geometry
        self._FT = Frank2D._FT
        
        self._Nx = self._Ny = Frank2D._N
        
        self._u_input = Frank2D._gridded_data['u']
        self._v_input = Frank2D._gridded_data['v']
        self._vis_input = Frank2D._gridded_data['vis']
        self._weights_input = Frank2D._gridded_data['weights']

        self._u_model = self._frank2d.u_grid
        self._v_model = self._frank2d.v_grid
        self._u_model_1d = self._frank2d.u
        self._v_model_1d = self._frank2d.v
        self._vis_model = Frank2D.visibility_model

        self._x_model = self._frank2d.x_grid*rad_to_arcsec
        self._y_model = self._frank2d.y_grid*rad_to_arcsec
        self._x_model_1d = self._frank2d.x*rad_to_arcsec
        self._y_model_1d = self._frank2d.y*rad_to_arcsec
        self._int_model = Frank2D.intensity_model.real

        self._Rmax = Frank2D.Rmax
        
    def visibility(self, kind = 'model',
                   title=r'$log |Vis|$', fig_size = 3,
                   vmin = -9, vmax = -2,
                   phase_shift = True, deproject = False):
        
        Nx, Ny = self._Nx, self._Ny
        geom = self._Geometry
        vis = None

        if kind == 'input':
            vis = self._vis_input
        elif kind == 'model':
            vis = self._vis_model
        else:
            raise ValueError("type must be 'input' or 'model'")

        # Shifted already.
        u = self._u_model #(N, N)
        v = self._v_model
        if phase_shift:
            # Only in this scheme (East of North) makes sense to do the phase shifting.
            vis = geom.apply_phase_shift(-u, -v, vis)
        
        if deproject: 
            ud, vd, _ = geom.deproject(u, v)
            u, v = ud, vd

        
        plt.figure(figsize = (fig_size, fig_size))
        plt.pcolormesh(u,
                       v,
                       np.log(np.abs(vis)),
                       cmap="magma",
                       vmin=-10, vmax=-2)
        plt.xlabel(r'u [1e6 $\lambda$]')
        plt.ylabel(r'v [1e6 $\lambda$]')
        plt.title(title)
        cmap = plt.colorbar(shrink=0.8)
        cmap.set_label(r'log|Visibility model| [Jy]', size=10)
        
        # This impose the convention East of North.  
        plt.xlim(u.max(), u.min())
        plt.ylim(v.max(), v.min())
        
        plt.gca().set_aspect(1)
        plt.gca().invert_yaxis()
        plt.show()
    
    def intensity(self, title= r'I_{Model}', fig_size = 6, vmin = 0, vmax = 4e10,
                  phase_shift = True, deproject = False):
        
        Nx, Ny = self._Nx, self._Ny
        f2d = self._frank2d
        geom = self._Geometry
        
        I = self._int_model

        u_grid = self._u_model #(N, N)
        v_grid = self._v_model
        if phase_shift:
            # Only in this scheme (East of North) makes sense to do the phase shifting.
            vis_ = self._vis_model
            vis = geom.apply_phase_shift(-u_grid, -v_grid, vis_)
            I = f2d.transform(vis).real
        
        x = self._x_model
        y = self._y_model
        if deproject:
            xd, yd = geom.deproject_xy(x_grid, y_grid)
            x, y = xd, yd
            
        plt.figure(figsize = (fig_size, fig_size))
        norm = colors.PowerNorm(gamma=0.45, vmin=0, vmax=4e10)
        plt.pcolormesh(x,
                       y,
                       I,
                       cmap="magma",
                       norm=norm)
        plt.xlabel(r'x ["]')
        plt.ylabel(r'y ["]')
        cmap = plt.colorbar(shrink=0.8)
        plt.title(title)
        cmap.set_label(r'I [Jy/sr]', size=10)
        
        # This impose the convention East of North.  
        plt.xlim(x.max(), x.min())
        plt.ylim(y.max(), y.min())
        
        plt.gca().set_aspect(1)
        plt.gca().invert_yaxis()
        plt.show()

    def get_profile(self, u, v, vis, bins, weighted = False, weights = None):
        from scipy.stats import binned_statistic
        
        q = np.hypot(u, v)

        q = q.ravel(order = 'C')
        vis = vis.ravel(order = 'C')
        
        if weighted:
            if weights is None:
                raise ValueError("Add weights.")
            weights_gridded = self._weights_input
            Vis_Weights_binned, bin_edges, _ = binned_statistic(q, vis*weights_gridded, 'sum', bins = bins)
            Weights_binned, bin_edges, _ = binned_statistic(q, weights_gridded, 'sum', bins = bins)

            vis_1d = np.nan_to_num(Vis_Weights_binned, nan=0)
            vis_1d = vis_1d/Weights_binned
        else:
            vis_1d, bin_edges, _ = binned_statistic(q, vis, 'mean', bins = bins)
        
        q_1d = (bin_edges[:-1] + bin_edges[1:]) / 2
            
        return q_1d, vis_1d

    def edges(self, x):
        mids = (x[1:] + x[:-1]) / 2.0
        first = x[0]  - (x[1] - x[0]) / 2.0
        last  = x[-1] + (x[-1] - x[-2]) / 2.0
        return np.r_[first, mids, last]

    def  visibility_profile(self, 
                            input = None, frank1d = False, weighted = True,
                            bins = 300,
                            phase_shift = True, deproject = True):

        f2d = self._frank2d
        geom = self._Geometry
        # model
        u_model = self._u_model
        v_model = self._v_model
        vis_model = self._vis_model
        
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
        q = np.linspace(q.min(), q.max(), bins)
        edges = self.edges(q)

        # f1d
        if frank1d:
            #frank1d
            u_input_f1d = u_input_g
            v_input_f1d = v_input_g
            vis_input_f1d = vis_input_g
            weights_input_f1d = weights_input_g
            
            sol = f2d.frank1d(  u = u_input_f1d, v = v_input_f1d,
                                vis = vis_input_f1d, weights = weights_input_f1d,
                                n_pts = self._Nx
                            )

            if deproject:
                vis_f1d = sol.predict_deprojected(q = q)
            else:
                vis_f1d = sol._vis_map.predict_visibilities(sol.mean, q, q*0, geometry=self._Geometry )

        # rescale total flux
        def rescale_total_flux(vis, weights):
            vis = vis / np.cos(geom.inc * deg_to_rad)
            weights = weights * np.cos(geom.inc * deg_to_rad) ** 2
            return vis, weights

        vis_model_scaled, weights_scaled = rescale_total_flux(vis_model, weights_input)
        vis_input, weights_input = rescale_total_flux(vis_input, weights_input)

        q_model_1d, vis_model_1d = self.get_profile(u_model, v_model, vis_model_scaled,
                                                    bins = edges,
                                                    weighted = weighted, weights = weights_scaled)

        # ----------------
        baselines = np.hypot(u_input, v_input)         
        grid = np.logspace(np.log10(min(baselines.min(), baselines[0])),
                                   np.log10(max(baselines.max(), baselines[-1])),
                                   10**4)
        
        
        plt.figure(figsize=(10,4))
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
        plt.ylabel('|V| [Jy]', size = 10)
        plt.title(r'$Visibility_{Model}$ N = ' + str(self._Nx))
        plt.legend(fontsize= 10, loc = 'best')
        plt.show()

    def intensity_profile(self, title= r'Brightness profile', 
                          clean = False, fits_file = None,
                          frank1d = True, bins = 300,
                          phase_shift = True, deproject = True):
        f2d = self._frank2d
        geom = self._Geometry
        inc, pa, dra, ddec = geom.inc, geom.pa, geom.dra, geom.ddec

        if clean:
            if fits_file is None:
                raise ValueError("Data location for the FITS file to make the CLEAN profile is not provided.")
            else:
                cube_1mm = imagecube(fits_file)
                x_clean, y_clean, dy_clean = cube_1mm.radial_profile(inc= inc, PA=pa, x0=dra, y0=ddec)
                
                fits_image = fits.open(fits_file)
                header = fits_image[0].header
                bmaj_deg = float(header["BMAJ"])
                bmin_deg = float(header["BMIN"])
                bmaj_as = bmaj_deg * 3600.0
                bmin_as = bmin_deg * 3600.0
                bpa_deg = header['BPA']
                
                clean_beam = {'bmaj': bmaj_as, 'bmin': bmin_as, 'beam_pa': bpa_deg}
                area = clean_beam['bmaj']*clean_beam['bmin']*np.pi/4./np.log(2.)*(1/rad_to_arcsec)**2

        # input
        u_input_g = self._u_input
        v_input_g = self._v_input
        vis_input_g = self._vis_input
        weights_input_g = self._weights_input

        # model
        u_model = self._u_model
        v_model = self._v_model
        vis_model = self._vis_model
        
        u_model_1d = self._u_model_1d
        v_model_1d = self._v_model_1d

        x_model = self._x_model
        y_model = self._y_model

        x_model_1d = self._x_model_1d
        y_model_1d = self._y_model_1d
        
        # phase shift
        vis = geom.apply_phase_shift(-u_model, -v_model, vis_model) # the East of North convention.
        I = f2d.transform(vis).real

        # deproject
        x_model, y_model = geom.deproject_xy(x_model, y_model)
        x_model_1d, y_model_1d = geom.deproject_xy(x_model_1d, y_model_1d)

        # collocation points for the plot.
        r = np.unique(np.hypot(x_model_1d, y_model_1d))
        r = np.linspace(r.min(), r.max(), bins)
        edges = self.edges(r)

        # f1d
        if frank1d:
            #frank1d
            u_input_f1d = u_input_g
            v_input_f1d = v_input_g
            vis_input_f1d = vis_input_g
            weights_input_f1d = weights_input_g
            
            sol = f2d.frank1d(u = u_input_f1d, v = v_input_f1d,
                                        vis = vis_input_f1d, weights = weights_input_f1d,
                                        n_pts = self._Nx)

            I_f1d = sol.mean
            r_f1d = sol.r

        r_model_1d, I_model_1d = self.get_profile(x_model, y_model, I, bins = edges)
        I_model_1d = np.nan_to_num(I_model_1d, nan=0)/np.cos(inc*deg_to_rad)
        
        plt.figure(figsize=(10,4))
        if clean:
            #  CLEAN --------------------------------------------------------------------
            plt.plot(x_clean, y_clean, "black", label = "CLEAN")
            plt.fill_between(x_clean, y_clean-dy_clean, y_clean+dy_clean, alpha=0.7)
            # frank2D -------------------------------------------------------------------
            I_model_1d_convolved = convolve_profile(r_model_1d, I_model_1d, inc, pa, clean_beam)*area
            plt.plot(r_model_1d, I_model_1d_convolved, color = 'blue',ls ='--', label = r'frank2d')
            
            if frank1d:
                # frank1D ---------------------------------------------------------------
                I_f1d_convolved = convolve_profile(r_f1d, I_f1d, inc, pa, clean_beam)*area
                plt.plot(r_f1d, I_f1d_convolved, color = "red", label = r'frank1d')
        
            plt.ylabel(r'log($I_\nu$) [Jy/beam]', size=10)
            plt.xlabel('Radius ["]', size=10)
            plt.legend(fontsize=12)
            plt.yscale('log')
            plt.ylim(1e-6, 3e-3)
            plt.xlim(0,self._Rmax/2)
            plt.show()
        else:
            # frank1D -------------------------------------------------------------------
            if frank1d:
                plt.plot(r_f1d, I_f1d, color = "red", label = r'frank1d')
    
            # frank2D -------------------------------------------------------------------
            plt.plot(r_model_1d, I_model_1d, color = 'blue', ls ='--', label = r'frank2d')

            plt.xlabel('Radius ["]', size = 10)
            plt.ylabel(r'log($I_\nu$) [$10^{10}$ Jy sr$^{-1}$]', size = 10)
            plt.title(title)
            plt.xlim(0, self._Rmax/2)
            plt.yscale('log')
            plt.legend()
            plt.show()