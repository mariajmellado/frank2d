
import numpy as np
import constants as const
import cupy as cp

class Geometry_GPU(object):
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
        self._inc *= const.deg_to_rad
        self._pa *= const.deg_to_rad

        cos_t = cp.cos(self._pa)
        sin_t = cp.sin(self._pa)

        if inverse:
            sin_t *= -1
            u = u / cp.cos(self._inc)

        up = u * cos_t - v * sin_t
        vp = u * sin_t + v * cos_t

        if inverse:
            return up, vp
        else:
        #   Deproject
            wp = up * cp.sin(self._inc)
            up = up * cp.cos(self._inc)

            return up, vp, wp

    def apply_phase_shift(self, u, v, V, inverse=False):
        self._dra *= 2. * cp.pi / const.rad_to_arcsec
        self._ddec *= 2. * cp.pi / const.rad_to_arcsec

        phi = u * self._dra + v * self._ddec

        if inverse:
            shifted_vis = V / (cp.cos(phi) + 1j * cp.sin(phi))
        else:
            shifted_vis = V * (cp.cos(phi) + 1j * cp.sin(phi))

        return shifted_vis