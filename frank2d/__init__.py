"""
Frank2D: High-performance visibility fitting with Gaussian Processes.
Unified API for dynamic CPU/GPU strategy selection.
"""
import warnings
import numpy as np
import importlib.util

# Common utilities and constants.
from .constants import rad_to_arcsec, deg_to_rad
from .geometry import Geometry
from .minimizer import Powell
from .logger import Logger

# =============================================================================
# CPU CORE COMPONENTS (Internal prefixed imports)
# =============================================================================
from .cpu import (
    Gridding as _Gridding_CPU,
    IterativeSolverMethod as _IterativeSolverMethod_CPU,
    FourierTransform2D as _FourierTransform2D_CPU,
    Plot as _Plot_CPU,
    MAPEstimator as _MAPEstimator_CPU,
    Frank2D as _Frank2D_CPU,
    Wendland as _Wendland_CPU,
    SquaredExponential as _SquaredExponential_CPU,
    linear_operator as _linear_operator_CPU,
    get_optimal_N as _get_optimal_N_CPU,
)

# =============================================================================
# GPU ACCELERATION SUPPORT (CUDA)
# =============================================================================

import os
import warnings

def _detect_gpu_support():
    """
    Check for CuPy, physical hardware, and local module integrity.
    """
    if os.environ.get("FRANK_NO_GPU"):
        return False, None
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            return False, None
    except (ImportError, Exception):
        return False, None
    try:
        from . import gpu as gpu_module
        return True, gpu_module
    except ImportError as e:
        # Aquí es donde atrapamos los errores de sintaxis o circulares
        warnings.warn(f"CuPy found, but Frank2D GPU module failed to load: {e}")
        return False, None

HAS_CUPY = importlib.util.find_spec("cupy") is not None

if HAS_CUPY:
    try:
        import cupy as cp
        # Solo intentamos cargar el módulo si hay hardware
        if cp.cuda.runtime.getDeviceCount() > 0:
            from . import gpu as gpu_module
            HAS_GPU = True
        else:
            HAS_GPU = False
    except Exception as e:
        print(f"[DEBUG] Error loading GPU logic: {e}")
        HAS_GPU = False
else:
    HAS_GPU = False

# =============================================================================
# COMPONENT FACTORIES (Strategy Routing)
# =============================================================================

def Frank2D(*args, use_gpu=False, **kwargs):
    """Main Mediator for the Frank2D algorithm."""
    if use_gpu and HAS_GPU:
        return gpu_module.Frank2D(*args, **kwargs)
    return _Frank2D_CPU(*args, **kwargs)

def Gridding(*args, use_gpu=False, **kwargs):
    # Factory for visibility gridding strategies.
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

def SquaredExponential(*args, use_gpu=False, **kwargs):
    """Factory for the Squared Exponential kernel."""
    if use_gpu and HAS_GPU:
        return gpu_module.SquaredExponential(*args, **kwargs)
    return _SquaredExponential_CPU(*args, **kwargs)

def Wendland(*args, use_gpu=False, **kwargs):
    """Factory for the Wendland kernel."""
    if use_gpu and HAS_GPU:
        return gpu_module.Wendland(*args, **kwargs)
    return _Wendland_CPU(*args, **kwargs)

def linear_operator(*args, use_gpu=False, **kwargs):
    """Factory for linear operator construction."""
    if use_gpu and HAS_GPU:
        return gpu_module.linear_operator(*args, **kwargs)
    return _linear_operator_CPU(*args, **kwargs)

def get_optimal_N(*args, use_gpu=False, **kwargs):
    """Factory for optimal pixel count calculation."""
    if use_gpu and HAS_GPU:
        return gpu_module.get_optimal_N(*args, **kwargs)
    return _get_optimal_N_CPU(*args, **kwargs)


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
    "Powell",
    "MAPEstimator",
    "rad_to_arcsec",
    "deg_to_rad",
    "get_optimal_N",
    "Logger",
]