import importlib
import pytest
import numpy as np

import frank2d
from frank2d import Frank2D

@pytest.mark.gpu
def test_factory_returns_gpu_class_when_requested():
    """use_gpu=True on a CUDA machine must return the GPU implementation."""
    model = Frank2D(100, 1.0, use_gpu=True)
    assert "gpu" in type(model).__module__.lower()


def test_factory_returns_cpu_class_by_default():
    """use_gpu=False must return the CPU implementation with correct metadata."""
    model = Frank2D(100, 1.0, use_gpu=False)
    assert "gpu" not in type(model).__module__.lower()
    assert model.N == 100
    assert np.isclose(model.Rmax, 1.0)


def test_use_gpu_true_without_gpu_falls_back_with_warning(monkeypatch):
    """
    use_gpu=True with no GPU must warn and fall back to CPU, never crash and
    never silently pretend it is on GPU. (This is the original bug.)
    """
    monkeypatch.setenv("FRANK_NO_GPU", "1")
    fresh = importlib.reload(frank2d)
    try:
        with pytest.warns(RuntimeWarning):
            model = fresh.Frank2D(100, 1.0, use_gpu=True)
        assert "gpu" not in type(model).__module__.lower()
    finally:
        monkeypatch.delenv("FRANK_NO_GPU", raising=False)
        importlib.reload(frank2d)


def test_require_gpu_raises_when_unavailable(monkeypatch):
    """FRANK2D_REQUIRE_GPU turns the CPU fallback into an error."""
    monkeypatch.setenv("FRANK_NO_GPU", "1")
    monkeypatch.setenv("FRANK2D_REQUIRE_GPU", "1")
    fresh = importlib.reload(frank2d)
    try:
        with pytest.raises(RuntimeError):
            fresh.Frank2D(100, 1.0, use_gpu=True)
    finally:
        monkeypatch.delenv("FRANK_NO_GPU", raising=False)
        monkeypatch.delenv("FRANK2D_REQUIRE_GPU", raising=False)
        importlib.reload(frank2d)


def test_backend_info_reports_state():
    info = frank2d.backend_info()
    assert {"has_cupy", "has_gpu", "gpu_disabled_reason"} <= set(info)