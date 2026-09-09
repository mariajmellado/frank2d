import pytest
import numpy as np
from scipy.interpolate import RectBivariateSpline

from frank2d import FourierTransform2D, rad_to_arcsec
from frank2d.cpu import FourierTransform2D

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Print a clear end-of-run banner.

    `terminalreporter` is injected by pytest into this hook; its `.stats` dict
    holds the test reports grouped by outcome.
    """
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    errors = len(terminalreporter.stats.get("error", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    print("\n" + "=" * 60)
    if failed or errors:
        print(f"❌ FRANK2D: {failed + errors} FAILED, {passed} passed, {skipped} skipped")
    elif skipped:
        print(f"❗️ FRANK2D: {passed} passed, but {skipped} SKIPPED "
              f"(run with -ra to see why)")
    else:
        print(f"🚀 FRANK2D: ALL {passed} TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


# =============================================================================
# GPU AVAILABILITY  (single source of truth for the whole suite)
# =============================================================================
try:
    import cupy as _cp
    HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_GPU = False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test needs a CUDA device")
    config.addinivalue_line("markers", "slow: long-running test")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords and not HAS_GPU:
            item.add_marker(skip_gpu)


@pytest.fixture
def has_gpu():
    """True when a CUDA device is available."""
    return HAS_GPU


@pytest.fixture(autouse=True)
def _reset_frank2d_backend():
    """
    Clear the committed backend before and after every test so test order never
    leaks a CPU/GPU choice from one test into the next. No-op if the installed
    frank2d has no reset_backend yet.
    """
    import frank2d
    reset = getattr(frank2d, "reset_backend", lambda: None)
    reset()
    yield
    reset()

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
def disk_with_crescent(
    N=512,                           # Image size (N x N pixels)
    pixel_scale_arcsec=0.01,         # Pixel scale (arcsec)
    sigma_core=0.04,                 # Core Gaussian width (arcsec)
    R_rings=[0.4, 0.6, 0.8],         # Rings radii (arcsec)
    width_rings=[0.05, 0.02, 0.05],  # Rings Gaussian width (arcsec)
    amp_core=0.1,                    # Core amplitude (Jy/pixel)
    amp_rings=[0.1, 0.04, 0.03],     # Rings amplitude (Jy/pixel)
    m_modes=[1, 0, 0],               # Asymmetry mode (1 -> crescent)
    theta0=[np.pi, 0, 0],            # Angle of crescent max (rad)
    amp_asym=[0.5, 0, 0]             # Asymmetry factor
    ):
    r"""
    Generate a synthetic disc image with non-axisymmetric substructures.

    The model follows the intensity distribution:
    $$I(R, \phi) = I_{core}(R) + \sum I_{ring, i}(R, \phi)$$
    
    Where asymmetries are introduced via azimuthal modulation:
    $$1 + A_{asym} \cos(m(\phi - \theta_0))$$

    Parameters
    ----------
    N : int
        Number of pixels per side.
    pixel_scale_arcsec : float
        Angular size of each pixel.
    sigma_core : float
        Standard deviation of the central stellar/inner disc emission.
    R_rings, width_rings, amp_rings : list
        Radial parameters for the ring components.
    m_modes : list
        Order of azimuthal asymmetry (m=0 is symmetric, m=1 is a crescent).
    amp_asym : list
        Strength of the asymmetry relative to the ring's base amplitude.

    Returns
    -------
    ndarray
        2D array of shape (N, N) representing the disc brightness (Jy/pixel).
    """
    
    # --- 1. Setup Coordinate Grid ---
    x_arcsec = (np.arange(N) - N / 2) * pixel_scale_arcsec
    y_arcsec = (np.arange(N) - N / 2) * pixel_scale_arcsec
    
    X, Y = np.meshgrid(x_arcsec, y_arcsec)
    R_image = np.sqrt(X**2 + Y**2)
    Phi_image = np.arctan2(Y, X)
    
    I_disk_jy_per_pixel = np.zeros((N, N))
    
    # --- 2. Central Core Emission ---
    I_core = amp_core * np.exp(-0.5 * (R_image / sigma_core)**2)
    I_disk_jy_per_pixel += I_core
    
    # --- 3. Ring & Crescent Substructures ---
    # We iterate through each ring to apply radial profiles and 
    # optional azimuthal modulations (m-modes).
    for A, R, w, m, th0, Aasym in zip(amp_rings, R_rings,
                                      width_rings, m_modes, theta0, amp_asym):
        
        radial_profile = np.exp(-0.5 * ((R_image - R) / w)**2)
        
        if m != 0 and Aasym > 0:
            # Introducing non-axisymmetry (Strategy for spirals/vortices)
            angular_modulation = (1 + Aasym * np.cos(m * (Phi_image - th0)))
            I_component = A * radial_profile * angular_modulation
        else:
            # Standard symmetric ring
            I_component = A * radial_profile
            
        I_disk_jy_per_pixel += I_component
            
    return I_disk_jy_per_pixel

def add_vis_noise(vis, weights, seed=None):
    r"""
    Add Gaussian noise to visibilities proportional to their weights.

    Parameters
    ----------
    vis : array
        Visibilities to add noise to (complex or real).
    weights : array
        Weights on the visibilities ($1 / \sigma^2$).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    vis_noisy : array
        Visibilities with added noise.
    """
    rng = np.random.default_rng(seed)

    vis = np.asarray(vis)
    weights = np.asarray(weights)

    sigma = np.zeros_like(weights, dtype=float)
    good = weights > 0
    sigma[good] = weights[good] ** -0.5

    dim0 = 2 if np.iscomplexobj(vis) else 1
    noise = rng.standard_normal((dim0,) + vis.shape) * sigma

    vis_noisy = vis + noise[0]
    if np.iscomplexobj(vis):
        vis_noisy = vis_noisy + 1j * noise[1]

    return vis_noisy

def apply_radial_dropout(weights_1d, N, protection_radius=0.3, dropout_prob=0.4, seed=None):
    r"""
    Applies a radial dropout mask to the weights array.
    
    The center of the uv-plane is protected (kept at 1.0), while the 
    outer regions are randomly zeroed out to simulate flagged data.

    Parameters
    ----------
    weights_1d : ndarray
        The 1D flattened weights array of size N*N.
    N : int
        Grid size per side.
    protection_radius : float
        Fraction of the radius (0 to 1) that will be protected from dropouts.
    dropout_prob : float
        Probability of zeroing out a pixel in the outer region.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Flattened weights array with applied radial dropouts.
    """
    rng = np.random.default_rng(seed)

    weights_2d = np.array(weights_1d, dtype=float).reshape((N, N))

    val = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(val, val)
    R = np.sqrt(X**2 + Y**2)

    outer_mask = R > protection_radius
    dropout_mask = rng.random((N, N)) < dropout_prob

    weights_2d[outer_mask & dropout_mask] = 0.0
    return weights_2d.ravel()

# =============================================================================
# VISIBILITY SAMPLING (GALARIO-LIKE IMPLEMENTATION)
# =============================================================================

def py_sampleImage(reference_image, dxy, udat, vdat, dRA=0., dDec=0., PA=0., origin='upper'):
    r"""
    Original from ``galario``.
    Python implementation of sampleImage.
    
    Calculates the complex visibilities $V(u,v)$ by performing a 
    Fast Fourier Transform (FFT) and interpolating the results at 
    the requested baseline coordinates.

    Parameters
    ----------
    reference_image : ndarray
        The 2D brightness distribution (source model).
    dxy : float
        Pixel size in radians.
    udat, vdat : ndarray
        The (u,v) coordinates in wavelength units ($\lambda$).
    dRA, dDec : float, optional
        Positional offsets in arcsec.
    PA : float, optional
        Position Angle for image rotation (radians).
    origin : {'upper', 'lower'}, optional
        Coordinate system origin for the FFT shift.

    Returns
    -------
    complex ndarray
        Sampled complex visibilities corresponding to the input (u,v).
    """
    if origin == 'upper':
        v_origin = 1.
    elif origin == 'lower':
        v_origin = -1.

    nxy = reference_image.shape[0]
    dRA *= 2.*np.pi
    dDec *= 2.*np.pi
    du = 1. / (nxy*dxy)

    # Fourier Domain
    # Real-to-Complex FFT with appropriate shifting
    fft_r2c_shifted = np.fft.fftshift(
                        np.fft.rfft2(
                            np.fft.fftshift(reference_image)), axes=0)
    
    # Geometric Transformations
    cos_PA, sin_PA = np.cos(PA), np.sin(PA)
    urot = udat * cos_PA - vdat * sin_PA
    vrot = udat * sin_PA + vdat * cos_PA
    dRArot = dRA * cos_PA - dDec * sin_PA
    dDecrot = dRA * sin_PA + dDec * cos_PA

    # Grid Interpolation
    # Mapping baselines to FFT indices for linear interpolation
    uroti = np.abs(urot)/du
    vroti = nxy/2. + v_origin * vrot/du
    uneg = urot < 0.
    vroti[uneg] = nxy/2 - v_origin * vrot[uneg]/du

    u_axis = np.linspace(0., nxy // 2, nxy // 2 + 1)
    v_axis = np.linspace(0., nxy - 1, nxy)

    # Bi-linear interpolation via RectBivariateSpline
    f_re = RectBivariateSpline(v_axis, u_axis, fft_r2c_shifted.real, kx=1, ky=1, s=0)
    ReInt = f_re.ev(vroti, uroti)
    f_im = RectBivariateSpline(v_axis, u_axis, fft_r2c_shifted.imag, kx=1, ky=1, s=0)
    ImInt = f_im.ev(vroti, uroti)
    f_amp = RectBivariateSpline(v_axis, u_axis, np.abs(fft_r2c_shifted), kx=1, ky=1, s=0)
    AmpInt = f_amp.ev(vroti, uroti)

    # Phase Correction
    uneg = urot < 0.
    ImInt[uneg] *= -1.
    PhaseInt = np.angle(ReInt + 1j*ImInt)

    # Apply phase shifts for RA/Dec offsets
    theta = urot*dRArot + vrot*dDecrot
    vis = AmpInt * (np.cos(theta+PhaseInt) + 1j*np.sin(theta+PhaseInt))

    return vis
    

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def small_grid():
    """
    Tiny grid for tests that build the dense kernel / solver matrix explicitly.
    N**2 = 256, so a 256x256 reference matrix is cheap. Do NOT build a full
    Frank2D with this (it is below the Nyquist limit); use FourierTransform2D,
    the kernels and IterativeSolverMethod directly.
    """
    return {"N": 16, "Rmax": 1.0}       # Rmax in arcsec


@pytest.fixture
def big_grid():
    """Realistic-ish grid for gridding and full-pipeline tests."""
    return {"N": 50, "Rmax": 1.0}       # Rmax in arcsec


@pytest.fixture
def simulated_obs(big_grid):
    """
    Synthetic asymmetric-disc observation sampled on a uniform (u, v) grid.

    Runs the full forward model: image -> visibilities (py_sampleImage) ->
    radial flagging -> noise.

    Returns
    -------
    dict
        uvtable : dict with keys 'u', 'v', 'vis', 'weights'
        N       : grid size per side
        Rmax    : field of view in arcseconds
    """
    N, Rmax = big_grid["N"], big_grid["Rmax"]

    pix_scale = (2 * Rmax) / N
    image = disk_with_crescent(N=N, pixel_scale_arcsec=pix_scale)

    Rmax_rad = Rmax / rad_to_arcsec
    FT = FourierTransform2D(Rmax_rad, N)
    u, v = FT.uv_points
    dxy_rad = FT.dx

    vis_clean = py_sampleImage(image, dxy_rad, u, v)

    weights = np.ones_like(u)
    weights = apply_radial_dropout(
        weights, N,
        protection_radius=0.50,
        dropout_prob=0.5,
        seed=46,
    )
    vis_noisy = add_vis_noise(vis_clean, weights, seed=46)

    return {
        "uvtable": {"u": u, "v": v, "vis": vis_noisy, "weights": weights},
        "N": N,
        "Rmax": Rmax,
    }