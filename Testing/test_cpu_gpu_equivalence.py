import pytest
import numpy as np
from frank2d import Frank2D

# --- Hardware Detection ---
try:
    import cupy as cp
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, Exception):
    HAS_GPU = False

# =============================================================================
# UTILS
# =============================================================================
def to_cpu(data):
    """
    Safely moves data from GPU (CuPy) to CPU (NumPy) if necessary.
    """
    if hasattr(data, 'get'):
        return data.get()
    return data

# =============================================================================
# EQUIVALENCE TESTS
# =============================================================================

@pytest.mark.skipif(not HAS_GPU, reason="GPU/CuPy not available for equivalence test")
def test_frank2d_logic_consistency(simulated_obs):
    """
    Validation of the Strategy pattern: CPU and GPU backends must 
    produce identical gridding and fitting results.
    """
    # --- 1. Extraction from Fixture ---
    uvtable = simulated_obs['uvtable']
    N = simulated_obs['N']
    Rmax = simulated_obs['Rmax']

    # --- 2. Initialization of both mediators ---
    # We pass use_gpu=False (CPU) and use_gpu=True (GPU)
    model_cpu = Frank2D(N, Rmax, use_gpu=False)
    model_gpu = Frank2D(N, Rmax, use_gpu=True)

    model_cpu.process_vis(uvtable)
    model_gpu.process_vis(uvtable)

    # --- 3. Step 1: Gridding Equivalence ---
    # Compare u, v, vis, and weights
    for key in ['u', 'v', 'vis', 'weights']:
        data_cpu = model_cpu.gridded_data[key]
        data_gpu = to_cpu(model_gpu.gridded_data[key])
        print(f"Comparing gridded component: {key}")
        print(f"CPU {key} stats: min={data_cpu.min()}, max={data_cpu.max()}, mean={data_cpu.mean()}")
        print(f"GPU {key} stats: min={data_gpu.min()}, max={data_gpu.max()}, mean={data_gpu.mean()}")
        
        # We use a strict tolerance for gridding since it's mostly indexing
        assert np.allclose(data_cpu, data_gpu, rtol=1e-12), \
            f"Gridding mismatch found in component: {key}"

    # --- 4. Step 2: Fitting Equivalence ---
    # Solving the system using the Conjugate Gradient Method
    # We use a very strict rtol as requested by the user
    model_cpu.fit(rtol=1e-10)
    model_gpu.fit(rtol=1e-10)

    # --- 5. Step 3: Model Result Comparison ---
    # Comparison of the Intensity Map (Image Plane)
    int_cpu = model_cpu.intensity_model
    int_gpu = to_cpu(model_gpu.intensity_model)

    # Comparison of the Model Visibilities (Fourier Plane)
    vis_cpu = model_cpu.visibility_model
    vis_gpu = to_cpu(model_gpu.visibility_model)

    # Assertions for final models
    # Note: GPU (float32/64) and CPU (float64) might have tiny epsilon differences 
    # due to floating point accumulation order in the solver.
    assert np.allclose(int_cpu, int_gpu, rtol=1e-8), \
        "Intensity model differs between CPU and GPU backends."
    
    assert np.allclose(vis_cpu, vis_gpu, rtol=1e-8), \
        "Visibility model differs between CPU and GPU backends."

    print("\n✅ CPU/GPU Equivalence Verified: Results are numerically consistent.")