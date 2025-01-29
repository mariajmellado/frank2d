import numpy as np
import matplotlib.pyplot as plt
from constants import rad_to_arcsec, deg_to_rad
import time
from matplotlib.colors import LogNorm


class Plot():
    def __init__(self, Frank2D, Geometry):
        self._frank2d = Frank2D
        self._geometry = Geometry
        
    def get_image_intensity(self, title = "Model", fig_size = 7,  add_fourier_resolution = False, log_norm = False, deprojected = False):
        frank2d = self._frank2d
        Nx = frank2d._Nx
        Ny = frank2d._Ny
        dx = frank2d._FT._dx*rad_to_arcsec
        dy = frank2d._FT._dy*rad_to_arcsec
        Rout = frank2d._Rmax*rad_to_arcsec
        x = (frank2d._FT._Xn*rad_to_arcsec).reshape(Nx, Ny)
        y = (frank2d._FT._Yn*rad_to_arcsec).reshape(Nx, Ny)
        I = frank2d.sol_intensity.reshape(Nx, Ny)

        if deprojected:
            inc_r = self._geometry._inc*deg_to_rad
            pa_r = self._geometry._pa*deg_to_rad

            cos_i = np.cos(inc_r)
            cos_pa, sin_pa = np.cos(pa_r), np.sin(pa_r)

            x = (x * cos_pa + y * sin_pa) / cos_i
            y = (x *-sin_pa + y * cos_pa)
    
        # Coordenadas del pixel que quieres mostrar
        pixel_x, pixel_y = Nx//2, Ny//2
        pixel_value = I[pixel_y, pixel_x]


        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        axs = [ax]
        
        norm = LogNorm() if log_norm else None

        # Primer subplot.
        plot = axs[0].pcolormesh(x, y, I, cmap='magma', norm=norm)
        axs[0].invert_xaxis()
        cmap = plt.colorbar(plot, ax=axs[0])
        cmap.set_label(r'I [Jy $sr^{-1}$]', size=15)

        axs[0].set_title(title)
        axs[0].set_xlabel("dRa ['']")
        axs[0].set_ylabel("dDec ['']")

        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()

        axs[0].text(
            xlim[0] + 0.1 * (xlim[1] - xlim[0]),  # 10% desde el borde izquierdo
            ylim[0] + 0.1 * (ylim[1] - ylim[0]),  # 10% desde el borde inferior
            r'  $N^{2}$ pixels,  N = ' + str(Nx) + '  ',
            bbox={'facecolor': 'white', 'pad': 4, 'alpha': 0.8}
        )


        plt.gca().set_aspect('equal')

        plt.show()


    def get_several_images(self, data, params, n_plots = 9, rtol = 1e-7, transform = "2fft"):
        u, v, Vis, Weights = data
        m, c, l = params

        I_array = []
        Vis_array = []
        frank2d = self._frank2d

        for i in range(0, n_plots):
            start_time = time.time()
            #----------------------
            print("fit : "+ str(i))
            frank2d.fit(u, v, Vis, Weights, kernel_params = [m[i], c[i], l[i]],
                         rtol = rtol, transform = transform)

            Vis_array.append(frank2d.sol_visibility)
            I_array.append(frank2d.sol_intensity)
            #----------------------
            end_time = time.time()
            execution_time = end_time - start_time
            print(f'total time = {execution_time/60 :.2f}  min | {execution_time: .2f} seconds')
            print(' with ' + f' m = {m[i]:.1f}, c = {c[i]:.1f}, l = {l[i]:.1e} ') 
            print(".....................")


        n_plots = int(np.sqrt(n_plots))

        fig, axs = plt.subplots(n_plots, n_plots, figsize=(18, 14))

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        x, y = frank2d._FT._x*const.rad_to_arcsec, frank2d._FT._y*const.rad_to_arcsec

        k = 0
        for i in range(n_plots):
            for j in range(n_plots):
                I = I_array[k].reshape(frank2d._Nx, frank2d._Ny).T
                plot = axs[i, j].pcolormesh(y, x, I, cmap='magma')
                
                axs[i, j].invert_xaxis()
                
                cmap = plt.colorbar(plot, ax=axs[i, j])

                if i == j and i == (0):
                    axs[i, j].set_title(f'AS209 at 1mm')
                    axs[i, j].set_xlabel("dRa ['']")
                    axs[i, j].set_ylabel("dDec ['']")
                    cmap.set_label(r'I [Jy $sr^{-1}$]', size=10)

                    Rout =  frank2d._Rmax*const.rad_to_arcsec

                    axs[i, j].text(1.7, -1.5, f' N = {frank2d._Nx}, FOV: {2*Rout:.2f} ', 
                                bbox={'facecolor': 'white', 'pad': 2, 'alpha': 0.8})

                axs[i, j].text(1.7, -1.8, f' m = {m[k]:.1f}, c = {c[k]:.1f}, l = {l[k]:.1e}', 
                            bbox={'facecolor': 'white', 'pad': 2, 'alpha': 0.8})

                axs[i, j].text(1.7, 1.6, f'{k}', bbox={'facecolor': 'white', 'pad': 2, 'alpha': 1})
                
                k += 1

        plt.show()
        
        return Vis_array

        



    