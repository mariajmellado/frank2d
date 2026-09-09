import pytest
import numpy as np

from frank2d import FourierTransform2D

try:
    import cupy as cp
except Exception:
    cp = None


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(params=[
    pytest.param("cpu", id="cpu"),
    pytest.param("gpu", id="gpu", marks=pytest.mark.gpu),
])
def setup_fourier(request):
    """Fourier engine built through the top-level factory, CPU or GPU."""
    backend = request.param
    xp = cp if backend == "gpu" else np
    N = 16
    Rmax = 1.0  # radians
    engine = FourierTransform2D(Rmax=Rmax, N=N, use_gpu=(backend == "gpu"))
    return engine, xp, backend, N

# =============================================================================
# ATOMIC TESTS
# =============================================================================

def test_resolution_consistency(setup_fourier):
    """dx must equal 2*Rmax / N."""
    engine, xp, _, N = setup_fourier
    expected_dx = (2 * engine.Rmax) / N
    assert np.isclose(float(engine.dx), expected_dx), "Spatial resolution (dx) mismatch."

def test_device_leakage(setup_fourier):
    """Grid data must stay on the requested device (no silent host<->GPU copies)."""
    engine, xp, backend, _ = setup_fourier
    x, y = engine.xy_points
    assert type(x).__module__.split(".")[0] == xp.__name__, \
        f"Data is not on the {backend} device."

def test_fourier_equivalence(setup_fourier):
    """FFT (fast_transform) must match the DFT matrix (direct_transform)."""
    engine, xp, _, _ = setup_fourier

    X, Y = xp.meshgrid(engine.x, engine.y)
    sigma = engine.Rmax / 4.0
    img_2d = xp.exp(-(X**2 + Y**2) / (2 * sigma**2))

    vis_fft = engine.fast_transform(img_2d, direction="forward")
    vis_direct = engine.direct_transform(img_2d.ravel(), direction="forward")

    assert xp.allclose(vis_fft.ravel(), vis_direct, rtol=1e-6, atol=1e-9), \
        "DFT and FFT results diverged numerically."

def test_unitary_roundtrip(setup_fourier):
    """
    Forward + Backward transform must recover the original signal.
    Validates normalization factors.
    """
    engine, xp, _, N = setup_fourier

    rng = xp.random.default_rng(0)
    original = rng.random((N, N))

    vis = engine.fast_transform(original, direction="forward")
    recovered = engine.fast_transform(vis, direction="backward")

    assert xp.allclose(recovered.real, original, atol=1e-10), "Signal not recovered."
    assert xp.allclose(recovered.imag, 0.0, atol=1e-10), "Spurious imaginary part."

def test_property_dimensions(setup_fourier):
    """uv_points and q must be flat vectors of length N*N for the solver."""
    engine, _, _, N = setup_fourier
    N2 = N * N
    u, v = engine.uv_points
    assert u.shape == (N2,) and v.shape == (N2,), "UV coordinates shape mismatch."
    assert engine.q.shape == (N2,), "Radial frequency vector (q) shape mismatch."