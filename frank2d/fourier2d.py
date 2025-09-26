import numpy as np

class FourierTransform2D(object):
    def __init__(self, Rmax, N, Geometry):
        """
        Fourier Transform in 2D using DFT and FFT.
        Parameters
        ----------
        Rmax : float
            Maximum value of the x and y coordinates in rad.
        N : int
            Number of collocation points in each direction.
        Geometry : str
            Geometry of the problem.
        """
        self._Xmax = Rmax # radians
        self._Ymax = Rmax
        self._Nx = N 
        self._Ny = N
        self._N2 = self.size = self._Nx*self._Ny # Number of points we want to use in the 2D-DFT.

        # Real space collocation points.
        self._x = np.linspace(-self._Xmax, self._Xmax, self._Nx, endpoint=False) # rad
        self._y = np.linspace(-self._Ymax, self._Ymax, self._Ny, endpoint=False) # rad
        x_, y_ = np.meshgrid(self._x, self._y)
        x_n, y_n = x_.reshape(-1), y_.reshape(-1) # x_n.shape = (N2,1)
        self._dx = 2*self._Xmax/self._Nx # rad.
        self._dy = 2*self._Ymax/self._Ny

        self._Xn = x_n
        self._Yn = y_n

        # Frequency space collocation points.
        self._u = np.fft.fftfreq(self._Nx, d = self._dx) # unshifted
        self._v = np.fft.fftfreq(self._Ny, d = self._dy) # unshifted


        # Shifted points (so that zero frequency is at the center of the array).
        self._u_shifted = np.fft.fftshift(self._u)
        self._v_shifted = np.fft.fftshift(self._v)
        
        # Default shifted points.
        u_shifted, v_shifted = np.meshgrid(self._u_shifted, self._v_shifted)

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

    def get_collocation_points(self):
        return np.array([self._Xn, self._Yn]), np.array([self._Un, self._Vn])

    def coefficients(self, u = None, v = None, x = None, y = None, direction="forward"):
        """
        Compute the coefficients of the 2D-DFT matrix.
        Parameters
        ----------
        """
        if direction == 'forward':
            # Normalization is dx*dy since we the DFT to be an approximation
            # of the integral (which depends on the area).
            norm = 4*self._Xmax*self._Ymax/self._N2
            factor = -2j*np.pi
            
            X, Y = self._Xn, self._Yn
            if u is None:
                u = self._Un
                v = self._Vn
        elif direction == 'backward':
            norm = 1 / (4*self._Xmax*self._Ymax)
            factor = 2j*np.pi
            
            X, Y = self._Un, self._Vn
            if u is None:
                u = self._Xn
                v = self._Yn
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
                     
        H = norm * np.exp(factor*(np.outer(u, X) + np.outer(v, Y)))

        return H

    def fast_transform(self, obj, direction = 'forward'):
        """
        Compute the 2D-FFT of an element.
        Parameters
        ----------
        obj : 2D array_like, shape = (Nx, Ny)
            Object to be transformed. Is assumed to be in a shifted form, i.e.,
            with the zero frequency at the center.
        direction : str
            Direction of the transform. Can be 'forward' or 'backward'.
        """
        # np.fft.fft2 assumes the zero frequency is at the [0,0] index.
        # so we need to unshift the object first.
        obj_unshifted = np.fft.ifftshift(obj)

        if direction == 'forward':
            return np.fft.fftshift(np.fft.fft2(obj_unshifted))*(self._dx * self._dy)
        elif direction == 'backward':
            return np.fft.fftshift(np.fft.ifft2(obj_unshifted))/(self._dx * self._dy)
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
          
    def direct_transform(self, obj, direction = 'forward'):
        """
        Compute the 2D-DFT of an element.
        Parameters
        ----------
        obj : 1D array_like
            Object to be transformed.
        direction : str
            Direction of the transform. Can be 'forward' or 'backward'.
        """
        if direction == 'forward':
            F = self.coefficients(direction = 'forward')
            return F@obj
        elif direction == 'backward':
            F_inverse = self.coefficients(direction = 'backward')
            return F_inverse@obj
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))

    @property
    def q(self):
        """ Radial collocation points in the frequency plane"""
        return np.hypot(self._Un, self._Vn)

    @property
    def r(self):
        """ Radial collocation points in the image plane"""
        return np.hypot(self._Xn, self._Yn)
    
    @property
    def Rmax(self):
        """ Maximum value of the x coordinate in rad"""
        return self._Xmax
    
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
    def uv_points_convention(self):
        """ Collocation points in the frequency plane with u to left, v upwards"""
        if self._Un_convention is None:
            u_convention = np.flip(self._Un, axis = 1).ravel(order="C")
            v_convention = np.flip(self._Vn, axis = 0).ravel(order="C")
            self._Un_convention, self._Vn_convention = u_convention, v_convention

    @property
    def uv_points_unshifted(self):
        """ Unshifted collocation points in the frequency plane"""
        if self._Un_unshifted is None:
            u, v = np.meshgrid(self._u, self._v)
            u, v = u.ravel(order="C"), v.ravel(order="C")
            self._Un_unshifted, self._Vn_unshifted = u, v

        return self._Un_unshifted, self._Vn_unshifted

    @property
    def uv_points(self):
        """ Collocation points in the frequency plane"""
        return self._Un, self._Vn