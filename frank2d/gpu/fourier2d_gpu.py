import cupy as cp

class FourierTransform2D(object):
    def __init__(self, Rmax, N):
        """
        Fourier Transform in 2D using DFT and FFT (CuPy backend).
        Parameters
        ----------
        Rmax : float
            Maximum value of the x and y coordinates in rad.
        N : int
            Number of collocation points in each direction.
        """
        self._Xmax = Rmax  # radians
        self._Ymax = Rmax
        self._N = N
        self._N2 = self.size = self._N * self._N  # Number of points we want to use in the 2D-DFT.

        # Real space collocation points.
        self._x = cp.linspace(-self._Xmax, self._Xmax, self._N, endpoint=False)  # rad
        self._y = cp.linspace(-self._Ymax, self._Ymax, self._N, endpoint=False)  # rad
        x_, y_ = cp.meshgrid(self._x, self._y)
        x_n, y_n = x_.reshape(-1), y_.reshape(-1)  # x_n.shape = (N2,1)
        self._dx = 2 * self._Xmax / self._N  # rad.
        self._dy = 2 * self._Ymax / self._N

        self._Xn = x_n
        self._Yn = y_n

        # Frequency space collocation points.
        self._u = cp.fft.fftfreq(self._N, d=self._dx)  # unshifted
        self._v = cp.fft.fftfreq(self._N, d=self._dy)  # unshifted

        # Shifted points (so that zero frequency is at the center of the array).
        self._u_shifted = cp.fft.fftshift(self._u)
        self._v_shifted = cp.fft.fftshift(self._v)

        # Default shifted points.
        u_shifted, v_shifted = cp.meshgrid(self._u_shifted, self._v_shifted)

        # u is positive to the right, v is positive downwards.
        # convention is u to left, v upwards. East and North.
        u_shifted_convention = (-u_shifted).ravel(order="C")
        v_shifted_convention = (-v_shifted).ravel(order="C")

        u_shifted = u_shifted.ravel(order="C") # shape = (N2,1)
        v_shifted = v_shifted.ravel(order="C")

        self._Un = u_shifted
        self._Vn = v_shifted

        self._Un_convention = u_shifted_convention
        self._Vn_convention = v_shifted_convention

        self._Un_unshifted = None
        self._Vn_unshifted = None

        self._in = True

    def coefficients(self, u=None, v=None, direction="forward"):
        """
        Compute the coefficients of the 2D-DFT matrix.
        Parameters
        ----------
        u : 1D array_like
            Frequency space collocation points along u-axis.
        v : 1D array_like
            Frequency space collocation points along v-axis.
        direction : str
            Direction of the transform. Can be 'forward' or 'backward'.
            - forward: from image space to frequency space.
            - backward: from frequency space to image space.
        Returns
        -------
        H : 2D array_like, shape = (len(u), N2) or (len(v), N2)
            Coefficient matrix of the 2D-DFT.
        """
        if direction == 'forward':
            # Normalization is dx*dy since we the DFT to be an approximation
            # of the integral (which depends on the area).
            norm = 4 * self._Xmax * self._Ymax / self._N2
            factor = -2j * cp.pi

            X, Y = self._Xn, self._Yn
            if u is None:
                u = self._Un
                v = self._Vn
        elif direction == 'backward':
            norm = 1 / (4 * self._Xmax * self._Ymax)
            factor = 2j * cp.pi

            X, Y = self._Un, self._Vn
            if u is None:
                u = self._Xn
                v = self._Yn
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))

        H = norm * cp.exp(factor * (cp.outer(u, X) + cp.outer(v, Y)))

        return H

    def fast_transform(self, obj, direction='forward'):
        """
        Compute the 2D-FFT of an element.
        Parameters
        ----------
        obj : 2D array_like, shape = (Nx, Ny)
            Object to be transformed. Is assumed to be in a shifted form, i.e.,
            with the zero frequency at the center.
        direction : str
            Direction of the transform. Can be 'forward' or 'backward'.
            Where:
                - forward: from image space to frequency space.
                - backward: from frequency space to image space.
        """
        # cp.fft.fft2 assumes the zero frequency is at the [0,0] index.
        # so we need to unshift the object first.
        obj_unshifted = cp.fft.ifftshift(obj)

        if direction == 'forward':
            return cp.fft.fftshift(cp.fft.fft2(obj_unshifted)) * (self._dx * self._dy)
        elif direction == 'backward':
            return cp.fft.fftshift(cp.fft.ifft2(obj_unshifted)) / (self._dx * self._dy)
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))

    def direct_transform(self, obj, direction='forward'):
        """
        Compute the 2D-DFT of an element.
        Parameters
        ----------
        obj : 1D array_like
            Object to be transformed.
        direction : str
            Direction of the transform. Can be 'forward' or 'backward'.
                - forward: from image space to frequency space.
                - backward: from frequency space to image space.
        """
        if direction == 'forward':
            F = self.coefficients(direction='forward')
            return F @ obj
        elif direction == 'backward':
            F_inverse = self.coefficients(direction='backward')
            return F_inverse @ obj
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))

    @property
    def q(self):
        """ Radial collocation points in the frequency plane"""
        return cp.hypot(self._Un, self._Vn)

    @property
    def r(self):
        """ Radial collocation points in the image plane"""
        return cp.hypot(self._Xn, self._Yn)

    @property
    def Rmax(self):
        """ Maximum value of the x coordinate in rad"""
        return self._Xmax

    @property
    def Qmax(self):
        """ Maximum value of the u coordinate in rad^-1"""
        return cp.max(self.q)
    
    @property
    def u(self):
        """ Collocation points in the frequency plane (0-centered)"""
        return self._u_shifted

    @property
    def v(self):
        """ Collocation points in the frequency plane (0-centered)"""
        return self._v_shifted

    @property
    def x(self):
        """ Collocation points x-axis in the image plane"""
        return self._x

    @property
    def y(self):
        """ Collocation points y-axis in the image plane"""
        return self._y
    
    @property
    def dx(self):
        """ Sampling interval in the image plane along x-axis in rad"""
        return self._dx
    
    @property
    def dy(self):
        """ Sampling interval in the image plane along y-axis in rad"""
        return self._dy

    @property
    def uv_points_convention(self):
        """ Collocation points in the frequency plane with u to left, v upwards"""
        if self._Un_convention is None:
            u_convention = cp.flip(self._Un, axis=1).ravel(order="C")
            v_convention = cp.flip(self._Vn, axis=0).ravel(order="C")
            self._Un_convention, self._Vn_convention = u_convention, v_convention
        return self._Un_convention, self._Vn_convention
    @property
    def uv_points_unshifted(self):
        """ Unshifted collocation points in the frequency plane"""
        if self._Un_unshifted is None:
            u, v = cp.meshgrid(self._u, self._v)
            u, v = u.ravel(order="C"), v.ravel(order="C")
            self._Un_unshifted, self._Vn_unshifted = u, v

        return self._Un_unshifted, self._Vn_unshifted
    
    @property
    def uv_points(self):
        """ Collocation points in the frequency plane"""
        return self._Un, self._Vn

    @property
    def collocation_points(self):
        return cp.array([self._Xn, self._Yn]), cp.array([self._Un, self._Vn])