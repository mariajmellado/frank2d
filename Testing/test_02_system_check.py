import pytest
import numpy as np
from frank2d import Frank2D

# --- Step 1: Handle CuPy availability ---
# We check if cupy is installed to avoid crashing the test suite
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# --- Step 2: Parametrized Test ---
# This runs the function twice: once with use_gpu=False and once with True
@pytest.mark.parametrize("use_gpu", [
    False, 
    pytest.param(True, marks=pytest.mark.skipif(not HAS_GPU, reason="CuPy not installed"))
])
def test_frank2d_factory_initialization(use_gpu):
    """
    Test if the Frank2D factory returns the correct class instance 
    based on the use_gpu flag.
    """
    N = 100
    Rmax = 1.0
    
    # Initialize the mediator
    model = Frank2D(N, Rmax, use_gpu=use_gpu)
    
    # Assertions: Basic metadata
    assert model.N == N
    assert model.Rmax == Rmax
    
    # Assertions: Class Type Check
    # We check if the object's name contains 'GPU' or 'CPU' (or your class names)
    class_name = type(model).__name__
    
    if use_gpu:
        # We check if 'gpu' is in the module path or class string
        # Path is 'frank2d.gpu.frank2d_gpu.Frank2D'
        full_class_path = str(type(model)).lower()
        assert "gpu" in full_class_path or "cupy" in full_class_path
    
    else:
        # For CPU, we just ensure it's not the GPU one
        assert "gpu" not in str(type(model)).lower()