import numpy as np
import matplotlib.pyplot as plt
from constants import rad_to_arcsec, deg_to_rad
import time
from matplotlib.colors import LogNorm


class Plot():
    def __init__(self, Frank2D, Geometry):
        self._frank2d = Frank2D
        self._Geometry = Geometry
        self._Nx = Frank2D._Nx
        self._Ny = Frank2D._Ny

        
    def intensity_model(self, title="Frank2D intensity model", deproject = False, fig_size = 7):
        frank2d = self._frank2d
        I = frank2d.sol_intensity
        x_ = frank2d._FT._Xn * rad_to_arcsec
        y_ = frank2d._FT._Yn * rad_to_arcsec
        Nx, Ny = self._Nx, self._Ny
        x, y = x_, y_

        inc_r = self._Geometry._inc * deg_to_rad
        pa_r = self._Geometry._pa * deg_to_rad

        cos_i = np.cos(inc_r)
        cos_pa, sin_pa = np.cos(pa_r), np.sin(pa_r)
        
        if deproject:
            x = (x_ * cos_pa + y_ * sin_pa) / cos_i
            y = (x_ * -sin_pa + y_ * cos_pa)

        plt.figure(figsize=(fig_size, fig_size))

        plot = plt.pcolormesh(y.reshape(Nx, Ny), x.reshape(Nx, Ny), I.reshape(Nx, Ny),
                              cmap='magma', vmin=0, vmax=4e10)
        plt.gca().invert_xaxis()
        cmap = plt.colorbar(plot, shrink=0.8)
        cmap.set_label(r'I [Jy $sr^{-1}$]', size=15)

        plt.title(title)
        plt.xlabel("dRa ['']")
        plt.ylabel("dDec ['']")

        # Asegurar misma cantidad de elementos en ambos ejes
        plt.gca().set_aspect(1)  # Fija la relación de aspecto a 1:1

        xlim = plt.xlim()
        ylim = plt.ylim()
        
        # Forzar límites iguales en ambos ejes para evitar distorsión
        lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        plt.xlim(lims)
        plt.ylim(lims)

        plt.text(
            lims[0] + 0.1 * (lims[1] - lims[0]),  
            lims[0] + 0.1 * (lims[1] - lims[0]),  
            r'  $N^{2}$ pixels,  N = ' + str(Nx) + '  ',
            bbox={'facecolor': 'white', 'pad': 4, 'alpha': 0.8}
        )

        plt.show()


    def visibility_model(self, title="Frank2D visibility model", deproject = False, fig_size = 7):
            frank2d = self._frank2d
            Nx, Ny = self._Nx, self._Ny

            vis_model = frank2d.sol_visibility.reshape(Nx, Ny)
            
            u, v = frank2d._FT._Un, frank2d._FT._Vn

            u_shifted, v_shifted = np.fft.fftshift(u.reshape(Nx, Ny)), np.fft.fftshift(v.reshape(Nx, Ny))
            vis_shifted = np.fft.fftshift(vis_model)

            if deproject: 
                u_shifted, v_shifted, _ = self._Geometry.deproject(u_shifted.flatten(), v_shifted.flatten())

            plt.pcolormesh(v_shifted.reshape(Nx, Ny), u_shifted.reshape(Nx, Ny), np.log(np.abs(vis_shifted)), cmap="viridis", vmin=-12, vmax=-2)
            plt.xlabel(r'u [ $\lambda$]')
            plt.ylabel(r'v [ $\lambda$]')
            plt.gca().set_aspect('equal') 
            cmap = plt.colorbar(shrink=0.8)
            cmap.set_label(r'V [Jy]', size=15)
            plt.title(r'log|$Vis_{model}$|')
            plt.show()


    def visibility_gridded_input(self, title="Frank2D gridded input", fig_size = 7):
        frank2d = self._frank2d
        u_gridded, v_gridded = frank2d._gridded_data['u'], frank2d._gridded_data['v']
        vis_gridded, weights_gridded = frank2d._gridded_data['vis'], frank2d._gridded_data['weights']

        FT = frank2d._FT
        x_labels = np.fft.fftshift(FT._u/1e6)
        y_labels = np.fft.fftshift(FT._v/1e6)

        num_ticks = 5
        x_ticks_to_show = np.linspace(0, len(x_labels) - 1, num_ticks).astype(int)
        y_ticks_to_show = np.linspace(0, len(y_labels) - 1, num_ticks).astype(int)

        def format_labels(x):
            return f"{int(x):d}" if abs(x) > 1e-8 else "0"
        
        plt.imshow(np.log(np.abs(np.fft.fftshift(vis_gridded.reshape(self._Nx, self._Ny)))), origin='lower', vmin=-12, vmax=-2)
        plt.xticks(ticks=x_ticks_to_show, labels=[format_labels(x) for x in np.array(x_labels)[x_ticks_to_show]])
        plt.yticks(ticks=y_ticks_to_show, labels=[format_labels(y) for y in np.array(y_labels)[y_ticks_to_show]])
        plt.xlabel(r'u [1e6 $\lambda$]')
        plt.ylabel(r'v [1e6 $\lambda$]')
        plt.title(r'log |$Vis_{Input}$|')
        cmap = plt.colorbar()
        cmap.set_label(r'V [Jy]', size=15)




        



    