import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

class Minimizer():
    def __init__(self, fun, guess, sol):
        """
        Params
        ------
        fun : callable
            The function to minimize. Should take a 1D array and return a scalar.
        guess : array-like
            Initial guess for the minimum.
        """
        self._fun = fun 
        self._x = np.atleast_1d(guess)
        self._sol = sol
    
    @property
    def solution(self):
        """
        The solution after running the minimizer.
        """
        return self._sol

class Powell(Minimizer):
    """Based on Powell's method from Numerical Recipes"""
    def __init__(self, fun, guess, eps=0.1):
        """
        Params
        ------
        fun : callable
            The function to minimize. Should take a 1D array and return a scalar.
        guess : array-like
            Initial guess for the minimum.
        eps : float
            Step size for line minimization.
        """
        super().__init__(fun, guess, guess)
        self._f0 = fun(guess)

        self._dir = np.eye(len(guess)) 
        self._eps = eps

    def line_min(self, x, direction):
        """
        Minimize along a direction from x.
        Uses scipy's minimize_scalar with a bracketed interval.
        Params
        ------
        x : array-like
            Starting point.
        direction : array-like
            Direction to minimize along.
        Returns
        -------
        dx : array-like
            The step taken along the direction.
        f : float
            The function value at the new point.
        """
        def f(dx):
            return self._fun(x + direction*dx)
        
        res = minimize_scalar(f, tol=3e-3, bracket=(0.0, self._eps))

        return res.x*direction, res.fun

    def step(self):
        """
        Perform one iteration of the conjugate direction minimization.
        """
        N = len(self._x)

        # Minimize along each search direction.
        x = self._x.copy()
        f = np.full(N+1, self._f0)
        for i in range(N):
            dx, fi = self.line_min(x, self._dir[i])

            x += dx
            f[i+1] = fi

        # Update the search directions.
        f0, fN = f[0], f[-1]
        fE = self._fun(2*x-self._x)
        df = -np.diff(f)
        i = np.argmax(df)
        if fE >= f0:
            pass
        elif 2*(f0 - 2*fN + fE)*((f0-fN)-df[i])**2 >= df[i]*(f0-fE)**2:
            pass
        else:
            p = x - self._x
            p = p/(p**2).sum()**0.5
            
            # Minimize along new direction
            dx, fi = self.line_min(x, p)
            x += dx

            # update directions, putting new one at the back.
            self._dir[i] = self._dir[-1]
            self._dir[-1] = p

        
        # Save the new guess for x
        self._x =  x
        self._f0 = fi
    
    def run(self):
        """
        Run the minimizer until convergence.
        """
        xs = [self.x]
        for i in range(10):
            self.step()
            dx = self.x - xs[-1]            
            xs.append(self.x)

            if (dx**2).sum()**0.5 < 0.01:
                break

        self._sol = self.x
    
    @property
    def x(self):
        """
        Current guess for the minimum.
        """
        return self._x
    
        
class Scipy(Minimizer):
    """Wrapper around scipy's minimize function."""
    def __init__(self, fun, guess, bounds = None):
        """
        Params
        ------
        fun : callable
            The function to minimize. Should take a 1D array and return a scalar.
        guess : array-like
            Initial guess for the minimum.
        """
        super().__init__(fun, guess, guess)
        self._bounds = bounds

    def run(self):
        """
        Run the minimizer until convergence.
        """
        res = minimize(self._fun, self._x, method='Nelder-Mead', tol=1e-2, bounds = self._bounds)
        sucess = res.success
        print("Minimization success:", sucess)
        self._sol = res.x