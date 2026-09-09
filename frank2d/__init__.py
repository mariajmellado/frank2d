"""
Frank2D: high-fidelity non-axisymmetric visibility fitting with Gaussian Processes.

Public entry point of the package. Exposes Frank2D, Gridding, FourierTransform2D,
MAPEstimator and the GP kernels, and routes each of them to the CPU backend
(frank2d.cpu) or the CUDA/CuPy backend (frank2d.gpu).

The backend is chosen once. Pass use_gpu to the first component you build
(usually Frank2D or Gridding) and every later call reuses it; passing the
opposite value afterwards raises RuntimeError. If the first call omits use_gpu,
the GPU backend is used when a CUDA device is available and the CPU backend
otherwise. Call reset_backend to start over, set FRANK_NO_GPU=1 to disable the
GPU backend, or FRANK2D_REQUIRE_GPU=1 to turn the CPU fallback into an error.
Use backend_info to inspect what was detected at import time.
"""

import os
import inspect
import warnings
import importlib.util

from .constants import rad_to_arcsec, deg_to_rad
from .geometry import Geometry
from .minimizer import Powell
from .logger import Logger

from . import cpu as _cpu_backend
from .cpu import linear_operator, get_optimal_N

# ---------------------------------------------------------------------------
# GPU backend detection (CUDA / CuPy).
#
# Loaded lazily and defensively so a missing or broken GPU stack never stops
# the CPU backend from working. Every failure mode is recorded as a short
# string that backend_info can report.
# ---------------------------------------------------------------------------

def _detect_gpu_backend():
    """
    Try to load the GPU backend.

    Returns
    -------
    module : module or None
        The frank2d.gpu package on success, None otherwise.
    reason : str or None
        Human-readable explanation, set only when module is None.
    """
    if os.environ.get("FRANK_NO_GPU"):
        return None, "disabled by the FRANK_NO_GPU environment variable"

    if importlib.util.find_spec("cupy") is None:
        return None, "cupy is not installed (pip install 'frank2d[gpu]')"

    try:
        import cupy as cp
    except Exception as exc:
        return None, f"'import cupy' failed: {exc!r}"

    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return None, f"CUDA runtime query failed: {exc!r}"
    if n_devices == 0:
        return None, "no CUDA-capable device detected"

    try:
        from . import gpu as gpu_module
    except Exception as exc:
        return None, f"cupy is available but frank2d.gpu failed to import: {exc!r}"

    return gpu_module, None


_GPU_MODULE, _GPU_DISABLED_REASON = _detect_gpu_backend()

HAS_CUPY = importlib.util.find_spec("cupy") is not None
HAS_GPU = _GPU_MODULE is not None

_REQUIRE_GPU = bool(os.environ.get("FRANK2D_REQUIRE_GPU"))

# Backend fixed by the first component the user builds: "cpu", "gpu" or None
# while nothing has been chosen yet. _gpu_denied records that a GPU request had
# to fall back to CPU, so a repeated GPU request is not flagged as a conflict.
_committed = None
_gpu_denied = False


def _gpu_unavailable_message():
    return (f"GPU backend requested but unavailable ({_GPU_DISABLED_REASON}); "
            f"using the CPU backend. See frank2d.backend_info().")


def _module_for(choice):
    """Return the backend module for choice, which is always usable here."""
    return _GPU_MODULE if choice == "gpu" else _cpu_backend


def _commit(use_gpu):
    """
    Resolve use_gpu against the committed backend and return the backend module.

    Parameters
    ----------
    use_gpu : bool or None
        True or False fixes the backend on the first call and must match it on
        every later call. None means "use the committed backend", auto-detecting
        (GPU when available) if nothing has been committed yet.

    Raises
    ------
    RuntimeError
        If use_gpu contradicts a backend that was already committed, or if a GPU
        backend is unavailable and FRANK2D_REQUIRE_GPU is set.
    """
    global _committed, _gpu_denied

    if use_gpu is None:
        if _committed is None:
            _committed = "cpu"
        return _module_for(_committed)

    want = "gpu" if use_gpu else "cpu"

    if want == "gpu" and not HAS_GPU:
        if _REQUIRE_GPU:
            raise RuntimeError(_gpu_unavailable_message())
        if not _gpu_denied:
            warnings.warn(_gpu_unavailable_message(), RuntimeWarning, stacklevel=3)
        _gpu_denied = True
        want = "cpu"

    if _committed is None:
        _committed = want
    elif want != _committed:
        raise RuntimeError(
            f"Frank2D is already running on the {_committed!r} backend, fixed by "
            f"the first component you built; this call asks for {want!r}. A single "
            f"pipeline cannot mix CPU and GPU components. Call "
            f"frank2d.reset_backend() first if the switch is intentional.")

    return _module_for(_committed)


def set_backend(use_gpu):
    """
    Fix the backend explicitly, before building any component.

    Equivalent to passing use_gpu to the first component, but clearer when a
    script states its choice up front.

    Parameters
    ----------
    use_gpu : bool
        True for the GPU backend, False for the CPU backend.

    Raises
    ------
    RuntimeError
        If a different backend was already committed.
    """
    _commit(bool(use_gpu))


def reset_backend():
    """
    Forget the committed backend so the next component can choose again.

    Useful in notebooks and tests that switch between CPU and GPU. Components
    already built are not affected.
    """
    global _committed, _gpu_denied
    _committed = None
    _gpu_denied = False


def active_backend():
    """
    Return the committed backend as "cpu" or "gpu", or None if none was chosen.
    """
    return _committed


def get_backend(use_gpu=None):
    """
    Return a backend module whose components are mutually consistent.

    Use this to assemble a pipeline by hand without repeating use_gpu on every
    call. The choice is committed exactly as if it had been passed to a factory.

    Parameters
    ----------
    use_gpu : bool or None, optional
        True for the GPU backend, False for CPU, None (default) to use the
        committed backend, auto-detecting (GPU when available) if none was
        committed yet.

    Returns
    -------
    module
        Either frank2d.cpu or frank2d.gpu.

    Examples
    --------
    B = frank2d.get_backend(use_gpu=True)
    ft = B.FourierTransform2D(Rmax, N)
    grid = B.Gridding(Rmax, ft)
    """
    return _commit(use_gpu)


# ---------------------------------------------------------------------------
# Dual-backend factories.
# ---------------------------------------------------------------------------

def _factory(name):
    """
    Build the public factory for name: a wrapper that instantiates the class
    from whichever backend is committed. It keeps the CPU class name and
    docstring and exposes the class as __wrapped__, so help() and
    inspect.signature() still show the real constructor.
    """
    cpu_impl = getattr(_cpu_backend, name)

    def factory(*args, use_gpu=None, **kwargs):
        return getattr(_commit(use_gpu), name)(*args, **kwargs)

    factory.__name__ = name
    factory.__qualname__ = name
    factory.__wrapped__ = cpu_impl
    doc = inspect.getdoc(cpu_impl.__init__) or inspect.getdoc(cpu_impl) or ""
    factory.__doc__ = doc + (
        "\n\n"
        "Additional parameters\n"
        "---------------------\n"
        "use_gpu : bool or None, optional\n"
        "    Backend for this call. The first component built fixes the backend;\n"
        "    later calls must agree or pass None. See frank2d.set_backend and\n"
        "    frank2d.reset_backend.\n")
    return factory


Frank2D = _factory("Frank2D")
Gridding = _factory("Gridding")
IterativeSolverMethod = _factory("IterativeSolverMethod")
FourierTransform2D = _factory("FourierTransform2D")
Plot = _factory("Plot")
MAPEstimator = _factory("MAPEstimator")
SquaredExponential = _factory("SquaredExponential")
Wendland = _factory("Wendland")


__all__ = [
    "Frank2D",
    "Gridding",
    "IterativeSolverMethod",
    "FourierTransform2D",
    "MAPEstimator",
    "Plot",
    "SquaredExponential",
    "Wendland",
    "linear_operator",
    "get_optimal_N",
    "Geometry",
    "Powell",
    "Logger",
    "rad_to_arcsec",
    "deg_to_rad",
    "set_backend",
    "reset_backend",
    "active_backend",
    "get_backend",
    "backend_info",
    "HAS_GPU",
    "HAS_CUPY",
]


def backend_info():
    """
    Report the active backend and, when the GPU backend is unavailable, why.

    Returns
    -------
    info : dict
        has_cupy : bool
            Whether the cupy package is importable.
        has_gpu : bool
            Whether the GPU backend is usable.
        gpu_disabled_reason : str or None
            Why the GPU backend is off, set only when has_gpu is False.
        committed_backend : str or None
            The backend fixed by the first component built, or None.
        require_gpu : bool
            Whether FRANK2D_REQUIRE_GPU is set.
        cupy_version : str or None
        cuda_device_count : int or None
    """
    info = {
        "has_cupy": HAS_CUPY,
        "has_gpu": HAS_GPU,
        "gpu_disabled_reason": _GPU_DISABLED_REASON,
        "committed_backend": _committed,
        "require_gpu": _REQUIRE_GPU,
        "cupy_version": None,
        "cuda_device_count": None,
    }
    if HAS_CUPY:
        try:
            import cupy as cp
            info["cupy_version"] = cp.__version__
            info["cuda_device_count"] = cp.cuda.runtime.getDeviceCount()
        except Exception:
            pass
    return info