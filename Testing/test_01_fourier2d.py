import pytest
import numpy as np
from frank2d import FourierTransform2D

try:
    import cupy as cp
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, Exception):
    HAS_GPU = False

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(params=[
    "cpu",
    pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="GPU/CuPy not available"))
])
def setup_fourier(request):
    """
    Unified fixture providing the Fourier engine through the Factory interface.
    Implements the Strategy pattern by passing use_gpu to the top-level API.
    """
    from frank2d import FourierTransform2D
    
    backend = request.param
    N = 16
    Rmax = 1.0 # rad
    xp = cp if backend == "gpu" else np
    
    # We pass the use_gpu flag so the __init__.py factory can route to the correct implementation.
    engine = FourierTransform2D(Rmax=Rmax, N=N, use_gpu=(backend == "gpu"))
    
    return engine, xp, backend

# =============================================================================
# ATOMIC TESTS
# =============================================================================

def test_resolution_consistency(setup_fourier):
    """
    Verify that spatial resolution (dx) is correctly derived from Rmax and N.
    Critical for absolute flux scaling in protoplanetary discs.
    """
    engine, xp, _ = setup_fourier
    expected_dx = (2 * engine.Rmax) / 16
    
    assert np.isclose(float(engine.dx), expected_dx), "Spatial resolution (dx) mismatch."

def test_device_leakage(setup_fourier):
    """
    Ensure that data remains on the requested hardware device.
    Prevents silent and slow transfers between RAM and VRAM.
    """
    engine, xp, backend = setup_fourier
    
    # Check internal grid storage
    assert type(engine._Xn).__module__.startswith(xp.__name__), \
        f"Memory leakage detected: Data is not on {backend} device."

def test_fourier_equivalence(setup_fourier):
    """
    Mathematical check: Fast Fourier Transform (FFT) must yield the 
    same result as the Direct Fourier Transform (DFT) matrix.
    """
    engine, xp, _ = setup_fourier
    
    # Create a synthetic Gaussian source on the active device
    x_axis = engine.x
    y_axis = engine.y
    print("[DEBUG] ", engine, xp, type(x_axis), type(y_axis))
    X, Y = xp.meshgrid(x_axis, y_axis)
    img_2d = xp.exp(-(X**2 + Y**2) / 0.1) 
    img_1d = img_2d.ravel()

    # Execute both strategies
    vis_fft = engine.fast_transform(img_2d, direction='forward')
    vis_direct = engine.direct_transform(img_1d, direction='forward')
    
    # Check numerical convergence
    assert xp.allclose(vis_fft.ravel(), vis_direct, rtol=1e-5), \
        "DFT and FFT results diverged numerically."

def test_unitary_roundtrip(setup_fourier):
    """
    Forward + Backward transform must recover the original signal.
    Validates normalization factors and energy conservation.
    """
    engine, xp, _ = setup_fourier
    N = 16
    
    # Random signal check
    original_signal = xp.random.rand(N, N)
    
    vis = engine.fast_transform(original_signal, direction='forward')
    recovered_signal = engine.fast_transform(vis, direction='backward')
    
    # Verify reconstruction within machine precision
    assert xp.allclose(original_signal, xp.abs(recovered_signal)), \
        "Unitary recovery failed: Energy was not conserved during transforms."

def test_property_dimensions(setup_fourier):
    """
    Ensure the Fourier engine provides correctly shaped data for the 
    Conjugate Gradient solver.
    """
    engine, _, _ = setup_fourier
    N2 = 16 * 16
    
    u, v = engine.uv_points
    assert u.shape == (N2,) and v.shape == (N2,), "UV coordinates shape mismatch."
    assert engine.q.shape == (N2,), "Radial frequency vector (q) shape mismatch."