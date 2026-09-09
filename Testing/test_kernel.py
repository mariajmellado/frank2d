"""
Tests for the GP correlation-matrix kernels (frank2d.cpu.gaussian_process).

CPU only, small grid: the dense kernel matrix is built explicitly as the
reference, so N stays small (N=16 -> 256x256).
"""
import numpy as np
import pytest

from frank2d import rad_to_arcsec
from frank2d.cpu import FourierTransform2D, Wendland, SquaredExponential

KERNELS = [Wendland, SquaredExponential]
PARAMS = {"m": -2.0, "c": 1e8, "l": 1e5}


@pytest.fixture
def uv(small_grid):
    """1D collocation points (u, v) of a small square grid, in lambda."""
    Rmax_rad = small_grid["Rmax"] / rad_to_arcsec
    ft = FourierTransform2D(Rmax_rad, small_grid["N"])
    u, v = ft.uv_points
    return np.asarray(u), np.asarray(v)


def _dense(kernel):
    """Full kernel matrix as a NumPy array."""
    return kernel.sparse_matrix().toarray()


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Kernel", KERNELS)
def test_matrix_is_square_with_right_size(uv, Kernel):
    u, v = uv
    K = _dense(Kernel(PARAMS, u, v))
    assert K.shape == (u.size, u.size)


@pytest.mark.parametrize("Kernel", KERNELS)
def test_cross_covariance_has_shape_size2_by_size(uv, Kernel):
    """With u2, v2 given the matrix is (len(u2), len(u))  ->  S_12."""
    u, v = uv
    idx = np.arange(0, u.size, 25)        # 11 points spread across the grid
    u2, v2 = u[idx], v[idx]
    K12 = _dense(Kernel(PARAMS, u, v, u2=u2, v2=v2))
    assert K12.shape == (u2.size, u.size)


# ---------------------------------------------------------------------------
# Algebraic properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Kernel", KERNELS)
def test_matrix_is_symmetric(uv, Kernel):
    u, v = uv
    K = _dense(Kernel(PARAMS, u, v))
    assert np.allclose(K, K.T, rtol=1e-10, atol=1e-12 * np.abs(K).max())


@pytest.mark.parametrize("Kernel", KERNELS)
def test_matrix_is_positive_semidefinite(uv, Kernel):
    u, v = uv
    K = _dense(Kernel(PARAMS, u, v))
    K = 0.5 * (K + K.T)                       # remove float asymmetry before eigh
    eig_min = np.linalg.eigvalsh(K).min()
    assert eig_min > -1e-6 * np.abs(K).max(), f"min eigenvalue = {eig_min:.3e}"


# ---------------------------------------------------------------------------
# Matrix-free operator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Kernel", KERNELS)
def test_sparse_operator_matches_dense_matvec(uv, Kernel):
    u, v = uv
    k = Kernel(PARAMS, u, v)
    K = k.sparse_matrix().toarray()
    op = k.sparse()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(u.size)
    assert np.allclose(op.matvec(x), K @ x, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Power spectrum and parameters
# ---------------------------------------------------------------------------
def test_power_spectrum_is_a_power_law(uv):
    u, v = uv
    k = Wendland(PARAMS, u, v)
    q = np.array([1e4, 1e5, 1e6])
    P = k.power_spectrum(q.copy(), PARAMS["m"], PARAMS["c"])
    assert np.allclose(P, PARAMS["c"] * q ** PARAMS["m"], rtol=1e-12)
    slope = np.polyfit(np.log(q), np.log(P), 1)[0]
    assert np.isclose(slope, PARAMS["m"], rtol=1e-6)


@pytest.mark.parametrize("Kernel", KERNELS)
def test_amplitude_scales_linearly_with_c(uv, Kernel):
    u, v = uv
    k1 = _dense(Kernel({**PARAMS, "c": 1e8}, u, v))
    k2 = _dense(Kernel({**PARAMS, "c": 2e8}, u, v))
    nz = k1 != 0
    assert np.allclose(k2[nz] / k1[nz], 2.0, rtol=1e-6)


@pytest.mark.parametrize("Kernel", KERNELS)
def test_matrix_construction_is_deterministic(uv, Kernel):
    u, v = uv
    a = _dense(Kernel(PARAMS, u, v))
    b = _dense(Kernel(PARAMS, u, v))
    assert np.array_equal(a, b)