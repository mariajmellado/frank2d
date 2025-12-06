import warnings
import sys

# -----------------------------------------------------------------------------
# Expose CPU.
# We import the actual classes directly. This ensures that IDEs (VS Code, PyCharm)
# can read the method signatures, docstrings, and provide full autocomplete.
# -----------------------------------------------------------------------------
try:
    from .frank2d import Geometry
    from .frank2d import Gridding
    from .frank2d import IterativeSolverMethod
    from .frank2d import FourierTransform2D
    from .frank2d import Plot
    from .frank2d import MAPEstimator
    from .frank2d import rad_to_arcsec, deg_to_rad
except ImportError as e:
    warnings.warn(f"Warning: Could not import default CPU modules: {e}")

# -----------------------------------------------------------------------------
# Expose GPU.
# -----------------------------------------------------------------------------
try:
    import cupy
    # We import the gpu package and alias it to 'gpu' for clean access
    from . import frank2d_gpu as gpu
except ImportError:
    # If cupy is missing or the package is not found, we set it to None
    # so we can check availability later without crashing.
    gpu = None 

# -----------------------------------------------------------------------------
# Frank2D Factory Function.
# -----------------------------------------------------------------------------
from .frank2d import Frank2D as _Frank2D_CPU

def Frank2D(*args, use_gpu=False, **kwargs):
    """
    Initializes the Frank2D algorithm.

    Parameters
    ----------
    use_gpu : bool, optional
        If True, attempts to use the CUDA-accelerated implementation.
        Defaults to False.
    *args, **kwargs :
        Arguments passed to the underlying Frank2D constructor.
    """
    if use_gpu:
        if gpu is None:
            warnings.warn("GPU mode requested but 'cupy' is not installed or 'frank2d_gpu' is missing. Falling back to CPU.")
        else:
            print("INFO: Initializing Frank2D in GPU mode 🚀")
            # We access the class dynamically from the 'gpu' namespace loaded above
            return gpu.Frank2D(*args, **kwargs)

    print("INFO: Initializing Frank2D in CPU mode 💻")
    return _Frank2D_CPU(*args, **kwargs)

# -----------------------------------------------------------------------------
# Module Exports.
# -----------------------------------------------------------------------------
__all__ = [
    "Frank2D",
    "Geometry",
    "Gridding",
    "IterativeSolverMethod",
    "FourierTransform2D",
    "Plot",
    "MAPEstimator",
    "rad_to_arcsec", 
    "deg_to_rad",
    "gpu"  # Exports the namespace so users can do 'from frank2d import gpu'
]