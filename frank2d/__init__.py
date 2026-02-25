"""
Frank2D: High-performance visibility fitting with Gaussian Processes.
Unified API for dynamic CPU/GPU strategy selection.
"""
import warnings
import numpy as np

# =============================================================================
# CPU CORE COMPONENTS (Internal prefixed imports)
# =============================================================================
from .cpu import (
    Geometry as _Geometry_CPU,
    Gridding as _Gridding_CPU,
    IterativeSolverMethod as _IterativeSolverMethod_CPU,
    FourierTransform2D as _FourierTransform2D_CPU,
    Plot as _Plot_CPU,
    MAPEstimator as _MAPEstimator_CPU,
    rad_to_arcsec,
    deg_to_rad,
    Frank2D as _Frank2D_CPU
)

# =============================================================================
# GPU ACCELERATION SUPPORT (CUDA)
# =============================================================================
try:
    import cupy
    from . import gpu as gpu_module
    HAS_GPU = True
except ImportError:
    gpu_module = None
    HAS_GPU = False

# =============================================================================
# COMPONENT FACTORIES (Strategy Routing)
# =============================================================================

def Frank2D(*args, use_gpu=False, **kwargs):
    """Main Mediator for the Frank2D algorithm."""
    if use_gpu and HAS_GPU:
        return gpu_module.Frank2D(*args, **kwargs)
    return _Frank2D_CPU(*args, **kwargs)

def Geometry(*args, use_gpu=False, **kwargs):
    """Factory for disc geometry configuration."""
    if use_gpu and HAS_GPU:
        return gpu_module.Geometry(*args, **kwargs)
    return _Geometry_CPU(*args, **kwargs)

def Gridding(*args, use_gpu=False, **kwargs):
    """Factory for visibility gridding strategies."""
    if use_gpu and HAS_GPU:
        return gpu_module.Gridding(*args, **kwargs)
    return _Gridding_CPU(*args, **kwargs)

def IterativeSolverMethod(*args, use_gpu=False, **kwargs):
    """Factory for the Conjugate Gradient solver."""
    if use_gpu and HAS_GPU:
        return gpu_module.IterativeSolverMethod(*args, **kwargs)
    return _IterativeSolverMethod_CPU(*args, **kwargs)

def FourierTransform2D(*args, use_gpu=False, **kwargs):
    """Factory for the 2D Fourier engine."""
    if use_gpu and HAS_GPU:
        print("[DEBUG] Instantiating GPU-accelerated FourierTransform2D")
        return gpu_module.FourierTransform2D(*args, **kwargs)
    return _FourierTransform2D_CPU(*args, **kwargs)

def Plot(*args, use_gpu=False, **kwargs):
    """Factory for image and visibility visualization."""
    if use_gpu and HAS_GPU:
        return gpu_module.Plot(*args, **kwargs)
    return _Plot_CPU(*args, **kwargs)

def MAPEstimator(*args, use_gpu=False, **kwargs):
    """Factory for posterior optimization and hyperparameter tuning."""
    if use_gpu and HAS_GPU:
        return gpu_module.MAPEstimator(*args, **kwargs)
    return _MAPEstimator_CPU(*args, **kwargs)

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================
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
]