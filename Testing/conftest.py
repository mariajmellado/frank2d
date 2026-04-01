import pytest
import numpy as np
from scipy.interpolate import RectBivariateSpline

from frank2d import FourierTransform2D, rad_to_arcsec
import matplotlib.pyplot as plt

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Hook to announce the successful completion of all atomic tests.
    This will only print if the exit status is 0 (all tests passed).
    """
    if exitstatus == 0:
        print("\n" + "="*50)
        print("🚀 FRANK2D: ALL ATOMIC TESTS PASSED SUCCESSFULLY!")
        print("="*50 + "\n")
    else:
        print("\n" + "!"*50)
        print("❌ SOME TESTS FAILED.")
        print("!"*50 + "\n")

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
    if seed is not None:
        np.random.seed(seed)

    dim0 = 2 if np.iscomplexobj(vis) else 1
    vis = np.array(vis)
    
    # Noise scaled by sigma (weights**-0.5)
    noise = np.random.standard_normal((dim0,) + vis.shape)
    noise *= weights ** -0.5

    vis_noisy = vis + noise[0]
    if np.iscomplexobj(vis):
        vis_noisy += 1j * noise[1]

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
    if seed is not None:
        np.random.seed(seed)

    # 1. Reshape to 2D for spatial operations
    weights_2d = weights_1d.reshape((N, N))
    
    # 2. Create normalized radial coordinate system
    val = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(val, val)
    R = np.sqrt(X**2 + Y**2)
    
    # 3. Define the region eligible for dropouts
    outer_mask = R > protection_radius
    
    # 4. Generate and apply the random dropout
    random_noise = np.random.random((N, N))
    dropout_mask = (random_noise < dropout_prob)
    
    # Only zero out if it is in the outer region AND selected by the probability
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

DISK_PARAMETRIC_FUNCTION = disk_with_crescent

@pytest.fixture
def simulated_obs(N = 50, Rmax = 2.0):
    """
    Fixture providing a synthetic asymmetric disc observation sampled 
    on a perfectly UNIFORM (u, v) grid.
    This fixture simulates the entire observation process, from image
    generation to visibility sampling, including noise addition.
    Parameters
    ----------
    N : int
        Number of pixels for the image and the (u, v) grid.
    Rmax : float
        Maximum baseline length in arcseconds (defining the field of view).
    Returns
    -------
    dict
        A dictionary containing:
        - 'u': Array of u coordinates (wavelength units).
        - 'v': Array of v coordinates (wavelength units).
        - 'vis': Complex visibilities with noise.
        - 'weights': Weights for each visibility point.
        - 'N_test': Grid size for the Frank2D solver.
        - 'Rmax_test': Maximum baseline length for testing (arcsec).
    """
    # Setup Simulation Parameters
    pix_scale = (2 * Rmax) / N
    image = DISK_PARAMETRIC_FUNCTION(N=N, pixel_scale_arcsec=pix_scale)

    #plt.imshow(image)
    
    # Generate Coordinate Grid
    # We use the project's own logic to define the (u, v) points
    # This ensures perfect alignment between testing and execution
    Rmax_rad = Rmax / rad_to_arcsec
    FT = FourierTransform2D(Rmax_rad, N)
    u, v = FT.uv_points
    dxy_rad = FT.dx

    # Sample Visibilities
    # The py_sampleImage function (Strategy) calculates V(u,v) from I(x,y)
    vis_clean = py_sampleImage(image, dxy_rad, u, v)
    
    # Add Noise & Weights
    weights = np.ones_like(u)
    weights = apply_radial_dropout(
        weights, 
        N, 
        protection_radius=0.50, # 25% inner core protected
        dropout_prob=0.5,       # 50% chance of flagging in outskirts
        seed=46
    )
    vis_noisy = add_vis_noise(vis_clean, weights, seed=46)

    #plt.imshow(np.log(np.abs(vis_noisy.reshape(N, N))))

    uvtable = {
        'u': u,
        'v': v,
        'vis': vis_noisy,
        'weights': weights
    }
    
    # Package for Frank2D
    return {
        'uvtable': uvtable,
        'N': N,
        'Rmax': Rmax
    }
