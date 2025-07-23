import numpy as np
import matplotlib.pyplot as plt
from constants import rad_to_arcsec, deg_to_rad
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm

# frank1d utilities
from frank.geometry import SourceGeometry
from frank.radial_fitters import FrankFitter
from frank.utilities import UVDataBinner

__all__ = [
    "Plot",
]

class Plot(object):
    def __init__(self, Frank2D, Geometry):
        self._frank2d = Frank2D
        self._Geometry = Geometry
        self._u_gridded, self._v_gridded = self._frank2d._gridded_data['u'], self._frank2d._gridded_data['v']
        self._vis_gridded, self._weights_gridded = self._frank2d._gridded_data['vis'], self._frank2d._gridded_data['weights']
        self._Nx = Frank2D._Nx
        self._Ny = Frank2D._Ny

        
    def intensity_model(self, title="Frank2D intensity model", deproject = False, zoom = None, fig_size = 6, vmax = 4e10):
        frank2d = self._frank2d
        I = frank2d.sol_intensity
        x_ = frank2d._FT._Xn * rad_to_arcsec
        y_ = frank2d._FT._Yn * rad_to_arcsec
        Nx, Ny = self._Nx, self._Ny
        x, y = x_, y_

        # Deprojection.
        inc_r = self._Geometry._inc * deg_to_rad
        pa_r = self._Geometry._pa * deg_to_rad

        cos_i = np.cos(inc_r)
        cos_pa, sin_pa = np.cos(pa_r), np.sin(pa_r)
        
        if deproject:
            x = (x_ * cos_pa + y_ * sin_pa) / cos_i
            y = (x_ * -sin_pa + y_ * cos_pa)

        plt.figure(figsize=(fig_size, fig_size))

        X = x.reshape(Ny, Nx)
        Y = y.reshape(Ny, Nx)
        I = I.reshape(Ny, Nx)

        I_flip = np.fliplr(I)
        X_flip = -np.fliplr(X)

        plot = plt.pcolormesh(X_flip, Y, I_flip,
                              cmap='magma', vmin=0, vmax=vmax)
        plt.gca().invert_xaxis()
        cmap = plt.colorbar(plot, shrink=0.8)
        cmap.set_label(r'I [Jy $sr^{-1}$]', size=15)

        plt.title(title)
        plt.xlabel("dRa ['']")
        plt.ylabel("dDec ['']")

        plt.gca().set_aspect(1)  


        xlim = plt.xlim()
        ylim = plt.ylim()
        
        lims = [xlim[1], xlim[0]]

        if zoom is not None:
            plt.xlim(zoom, -zoom)
            plt.ylim(-zoom, zoom)
        else:
            plt.text(
            lims[0] + 0.8 * (lims[1] - lims[0]),  
            lims[0] + 0.1 * (lims[1] - lims[0]),  
            r'  $'+ str(Nx) + '^{2}$ pixels ',
            bbox={'facecolor': 'white', 'pad': 4, 'alpha': 0.8}
        )

        plt.show()


    def visibility_model(self, title="Frank2D visibility model", deproject = False):
            frank2d = self._frank2d
            Nx, Ny = self._Nx, self._Ny

            vis_model = frank2d.sol_visibility.reshape(Nx, Ny)
            
            u, v = frank2d._FT._Un, frank2d._FT._Vn

            u_shifted, v_shifted = np.fft.fftshift(u.reshape(Nx, Ny)), np.fft.fftshift(v.reshape(Nx, Ny))
            vis_shifted = np.fft.fftshift(vis_model)

            if deproject: 
                u_shifted, v_shifted, _ = self._Geometry.deproject(u_shifted.flatten(), v_shifted.flatten())

            plt.pcolormesh(v_shifted,
                           u_shifted,
                           np.log(np.abs(vis_shifted)),
                           cmap="viridis", vmin=-12, vmax=-2)
            plt.xlabel(r'u [ $\lambda$]')
            plt.ylabel(r'v [ $\lambda$]')
            plt.gca().set_aspect('equal') 
            cmap = plt.colorbar(shrink=0.8)
            cmap.set_label(r'V [Jy]', size=15)
            plt.title(r'log|$Vis_{model}$|')
            plt.show()


    def visibility_gridded_input(self, title="Frank2D gridded input", deproject = False):
        frank2d = self._frank2d
        Nx, Ny = self._Nx, self._Ny
        
        u_gridded, v_gridded = self._u_gridded, self._v_gridded
        vis_gridded, weights_gridded = self._vis_gridded, self._weights_gridded
        
        u_shifted, v_shifted = np.fft.fftshift(u_gridded.reshape(Nx, Ny)), np.fft.fftshift(v_gridded.reshape(Nx, Ny))
        vis_shifted = np.fft.fftshift(vis_gridded.reshape(Nx, Ny))

        if deproject: 
            u_shifted, v_shifted, _ = self._Geometry.deproject(u_shifted.flatten(), v_shifted.flatten())

        
        plt.pcolormesh(v_shifted,
                       u_shifted,
                       np.log(np.abs(vis_shifted)),
                       cmap="viridis", vmin=-12, vmax=-2)
        plt.xlabel(r'u [1e6 $\lambda$]')
        plt.ylabel(r'v [1e6 $\lambda$]')
        plt.title(r'log |$Vis_{Input}$|')
        cmap = plt.colorbar()
        cmap.set_label(r'V [Jy]', size=15)


    def  get_vis_profile(self, n_bins, range =  None, weighted = False, deprojected = False):
        u = self._frank2d._FT._Un
        v = self._frank2d._FT._Vn
    
        inc_r = self._frank2d._Geometry._inc*deg_to_rad
        pa_r = self._frank2d._Geometry._pa*deg_to_rad

        if deprojected:
            cos_i = np.cos(inc_r)
            cos_pa, sin_pa = np.cos(pa_r), np.sin(pa_r)
        
            u_ = u * cos_pa - v * sin_pa
            v_d = u * sin_pa + v * cos_pa
            u_d = u_ * cos_i
            u, v = u_d, v_d
        
        q = np.hypot(u, v)
        
        Vis = self._frank2d.sol_visibility
        q = q.flatten()
        Vis = Vis.flatten()
        
        weights_gridded = self._frank2d._gridded_data['weights']
        Vis_binned, bin_edges, _ = binned_statistic(q, Vis, 'mean', bins = n_bins, range = range)
        q_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if weighted:
            if range is None:
                Vis_Weights_binned, bin_edges, _ = binned_statistic(q, Vis*weights_gridded, 'sum', bins = n_bins)
                Weights_binned, bin_edges, _ = binned_statistic(q, weights_gridded, 'sum', bins = n_bins)
            else:
                Vis_Weights_binned, bin_edges, _ = binned_statistic(q, Vis*weights_gridded, 'sum', bins = n_bins, range = range)
                Weights_binned, bin_edges, _ = binned_statistic(q, weights_gridded, 'sum', bins = n_bins, range = range)
                
            Vis_Weights_binned = np.nan_to_num(Vis_Weights_binned, nan=0)
            return q_centers, Vis_Weights_binned/Weights_binned
    
        return q_centers, Vis_binned
        
    def frank1d(self, u, v, Vis, Weights, alpha = 1.3, w_smooth = 1e-3, n_pts = 300):
        Rout = self._frank2d._Rmax*rad_to_arcsec
        geom = self._Geometry
        inc, pa, dra, ddec = geom._inc, geom._pa, geom._dra, geom._ddec
        
        geom_f1d = SourceGeometry(inc= inc, PA= pa, dRA= dra, dDec= ddec)
        FF = FrankFitter(Rout, n_pts, geom_f1d, alpha = alpha, weights_smooth = w_smooth)
        sol = FF.fit(u, v, Vis, Weights)
        return sol
        
    def compare_vis_profile_frank1d(self, u, v, Vis, Weights):
        #  Plot variables.
        cs, ms = ['#a4a4a4', 'k'], ['.', 'x']
        bin_widths = [1e3, 1e5]

        u_deproj, v_deproj, _ = self._Geometry.deproject(u, v)

        
        baselines = np.hypot(u_deproj, v_deproj)         
        grid = np.logspace(np.log10(min(baselines.min(), baselines[0])),
                                   np.log10(max(baselines.max(), baselines[-1])),
                                   10**4)
        
        
        plt.figure(figsize=(14,5))
        
        # Raw visibilities -------------------------------------------------------------------
        binned_vis = UVDataBinner(baselines, Vis, Weights, bin_widths[0])
        plt.plot(binned_vis.uv, np.abs(binned_vis.V), c=cs[0],
                     marker=ms[0], ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[0]/1e3))
        
        binned_vis = UVDataBinner(baselines, Vis, Weights, bin_widths[1])
        plt.plot(binned_vis.uv, np.abs(binned_vis.V), c=cs[1],
                     marker=ms[1], ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[1]/1e3))

        # Frank1D -----------------------------------------------------------------------------
        Nbins = 1000
        sol = self.frank1d(u, v, Vis, Weights)
        baselines_edges = np.geomspace(sol.q[0], sol.q[-1], Nbins + 1 )
        baselines_centers = np.sqrt(baselines_edges[:-1] * baselines_edges[1:]) 
        range = (sol.q[0], sol.q[-1])
        
        vis_fit_1d = sol.predict_deprojected(baselines_centers)
        plt.plot(baselines_centers, np.abs(vis_fit_1d), color = "red", label = r'frank1d', ls ='--')

        # Frank2D (deprojected)------------------------------------------------------------------
        q, Vis_model = self.get_vis_profile(baselines_edges, range, weighted = True, deprojected = True)
        plt.plot(baselines_centers, np.abs(Vis_model), label = f'frank2d', color = 'blue')

        plt.xlabel(r'baseline [$\lambda$]')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-5, 10)
        plt.xlim(2e5, 6e6)
        plt.ylabel('|V| [Jy]', size = 10)
        plt.title(r'$Visibility_{Model}$ N = ' + str(self._Nx))
        plt.legend(fontsize= 10, loc = 'best')
        plt.show()