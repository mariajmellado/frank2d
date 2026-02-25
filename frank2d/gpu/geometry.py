import numpy as np
from .constants import rad_to_arcsec, deg_to_rad

class Geometry(object):
    def __init__(self, inc, pa, dra, ddec):
        """
        Class to handle geometric transformations of visibilities (CuPy backend).
        Parameters
        ----------
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        dra : float
            Right ascension offset in arcseconds.
        ddec : float
            Declination offset in arcseconds.
        """
        self._inc = inc
        self._pa = pa
        self._dra = dra
        self._ddec = ddec

    def apply_correction(self, u, v, V, use3D=False):
        Vp = self.apply_phase_shift(u, v, V, inverse=True)
        up, vp, wp = self.deproject(u, v)
        return up, vp, Vp

    def deproject(self, u, v, inverse=False):
        """
        Deproject the (u,v) coordinates to account for inclination and position angle.
        """
        inc = self._inc * deg_to_rad
        pa = self._pa * deg_to_rad

        cos_t = np.cos(pa)
        sin_t = np.sin(pa)

        if inverse:
            sin_t *= -1
            u = u / np.cos(inc)

        up = u * cos_t - v * sin_t
        vp = u * sin_t + v * cos_t

        if inverse:
            return up, vp
        else:
            # Deproject
            wp = up * np.sin(inc)
            up = up * np.cos(inc)
            return up, vp, wp

    def deproject_xy(self, x, y, inverse=False):
        """
        Deproject the (x,y) coordinates to account for inclination and position angle.
        """
        inc = self._inc * deg_to_rad
        pa = self._pa * deg_to_rad

        cos_i = np.cos(inc)
        cos_pa, sin_pa = np.cos(pa), np.sin(pa)

        x_d = (x * cos_pa + y * sin_pa)
        y_d = (x * -sin_pa + y * cos_pa)

        x_d = x_d / cos_i
        return x_d, y_d

    def apply_phase_shift(self, u, v, V, inverse=False):
        """
        Apply a phase shift to the visibilities to account for position offsets.
        """
        dra = self._dra * 2.0 * np.pi / rad_to_arcsec
        ddec = self._ddec * 2.0 * np.pi / rad_to_arcsec

        phi = u * dra + v * ddec

        if inverse:
            shifted_vis = V / (np.cos(phi) + 1j * np.sin(phi))
            # alternativamente: shifted_vis = V * np.exp(-1j * phi)
        else:
            shifted_vis = V * (np.cos(phi) + 1j * np.sin(phi))
            # alternativamente: shifted_vis = V * np.exp(1j * phi)

        return shifted_vis

    def apply_phase_shift_xy(self, x, y, I, inverse=False):
        """
        Apply a phase shift to the image to account for position offsets.
        (Nota: el original solo calcula phi y no devuelve nada; se mantiene igual.)
        """
        dra = self._dra * 2.0 * np.pi / rad_to_arcsec
        ddec = self._ddec * 2.0 * np.pi / rad_to_arcsec

        phi = x * dra + y * ddec
        # El código original no usa 'phi' ni retorna un valor.

    @property
    def inc(self):
        """ Inclination angle in degrees"""
        return self._inc

    @property
    def pa(self):
        """ Position angle in degrees"""
        return self._pa

    @property
    def dra(self):
        """ Right ascension offset in arcseconds"""
        return self._dra

    @property
    def ddec(self):
        """ Declination offset in arcseconds"""
        return self._ddec