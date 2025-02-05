import numpy as np

class FourierTransform2D(object):
    def __init__(self, Rmax, N, Geometry):
        # Remember that now N is to create N**2 points in image plane.
        self._Xmax = Rmax #Rad
        self._Ymax = Rmax
        self._Nx = N 
        self._Ny = N
        self._N2 = self._Nx*self._Ny # Number of points we want to use in the 2D-DFT.

        # Real space collocation points
        self._x = np.linspace(-self._Xmax, self._Xmax, self._Nx, endpoint=False) # rad
        self._y = np.linspace(-self._Ymax, self._Ymax, self._Ny, endpoint=False) # rad
        x_, y_ = np.meshgrid(self._x, self._y, indexing='ij')
        # x_n.shape = N**2 X 1, so now, we have N**2 collocation points in the image plane.
        x_n, y_n = x_.reshape(-1), y_.reshape(-1)

        self._dx = 2*self._Xmax/self._Nx
        self._dy = 2*self._Ymax/self._Ny

        # Frequency space collocation points.
        self._u = np.fft.fftfreq(self._Nx, d = self._dx)# unshifted
        self._v = np.fft.fftfreq(self._Ny, d = self._dy) # unshifted
        self._u_shifted = np.fft.fftshift(self._u)
        self._v_shifted = np.fft.fftshift(self._v)
        
        u_, v_ = np.meshgrid(self._u, self._v, indexing='ij') 
        # u_n.shape = N**2 X 1, so now, we have N**2 collocation points.
        u_n, v_n = u_.reshape(-1), v_.reshape(-1)

        self._Xn = x_n
        self._Yn = y_n
        self._Un = u_n
        self._Vn = v_n

    def get_collocation_points(self):        
        return np.array([self._Xn, self._Yn]), np.array([self._Un, self._Vn])

    def coefficients(self, u = None, v = None, x = None, y = None, direction="forward"):
        #start_time = time.time()
        if direction == 'forward':
            ## Normalization is dx*dy since we the DFT to be an approximation
            ## of the integral (which depends on the area)
            norm = 4*self._Xmax*self._Ymax/self._N2
            factor = -2j*np.pi
            
            X, Y = self._Xn, self._Yn
            if u is None:
                u = self._Un
                v = self._Vn
        elif direction == 'backward':
            ## Correcting for the normalization above 1/N is replaced by this:
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

    def fast_transform(self, element, direction = 'forward'):
        if direction == 'forward':
            return np.fft.fftshift(np.fft.fft2(element.reshape(self._Nx, self._Ny)).real)*(self._dx * self._dy)
        elif direction == 'backward':
            return np.fft.fftshift(np.fft.ifft2(element.reshape(self._Nx, self._Ny)).real)/(self._dx * self._dy)
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
          
    def transform(self, element, direction = 'forward'):
        if direction == 'forward':
            F = self.coefficients(direction = 'forward')
            return F@element
        elif direction == 'backward':
            F_inverse = self.coefficients(direction = 'backward')
            return F_inverse@element
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
    @property
    def size(self):
        """Number of points used in the 2D-DFT"""
        return self._N2

    @property
    def q(self):
        """Frequency points"""
        return np.hypot(self._Un, self._Vn)
    
    @property
    def Rmax(self):
        """ Maximum value of the x coordinate in rad"""
        return self._Xmax
    
    @property
    def resolution(self):
        """ Resolution of the grid in the x coordinate in rad"""
        return self._dx
    
    @property
    def xy_points(self):
        """ Collocation points in the image plane"""
        return self._Xn, self._Yn
    
    @property
    def uv_points(self):
        """u and v  collocation points"""
        return self._Un, self._Vn