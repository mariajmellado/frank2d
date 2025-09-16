
import numpy as np
from .constants import rad_to_arcsec, deg_to_rad

class Geometry(object):
    def __init__(self, inc, pa, dra, ddec, deproject = False):
        self._inc = inc
        self._pa = pa
        self._dra = dra
        self._ddec = ddec
        self._deproject = deproject

    def apply_correction(self, u, v, V, use3D=False):
        Vp = self.apply_phase_shift(u, v, V, inverse=True)
        up, vp, wp = self.deproject(u, v)

        return up, vp, Vp

    def deproject(self, u, v, inverse=False):
        self._inc *= deg_to_rad
        self._pa *= deg_to_rad

        cos_t = np.cos(self._pa)
        sin_t = np.sin(self._pa)

        if inverse:
            sin_t *= -1
            u = u / np.cos(self._inc)

        up = u * cos_t - v * sin_t
        vp = u * sin_t + v * cos_t

        if inverse:
            return up, vp
        else:
        #   Deproject
            wp = up * np.sin(self._inc)
            up = up * np.cos(self._inc)

            return up, vp, wp

    def apply_phase_shift(self, u, v, V, inverse=False):
        self._dra *= 2. * np.pi / rad_to_arcsec
        self._ddec *= 2. * np.pi / rad_to_arcsec

        phi = u * self._dra + v * self._ddec

        if inverse:
            shifted_vis = V / (np.cos(phi) + 1j * np.sin(phi))
        else:
            shifted_vis = V * (np.cos(phi) + 1j * np.sin(phi))

        return shifted_vis