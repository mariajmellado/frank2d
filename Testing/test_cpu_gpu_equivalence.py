import pytest
import numpy as np


def _to_numpy(a):
    """CuPy or NumPy array -> NumPy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


@pytest.mark.gpu
def test_gridding_cpu_gpu_equivalence(simulated_obs):
    """Gridding must give the same u, v, vis, weights on both backends."""
    from frank2d.cpu import Frank2D as Frank2D_CPU
    from frank2d.gpu import Frank2D as Frank2D_GPU

    d = simulated_obs
    N, Rmax, uvtable = d["N"], d["Rmax"], d["uvtable"]

    m_cpu = Frank2D_CPU(N, Rmax)
    m_gpu = Frank2D_GPU(N, Rmax)
    m_cpu.process_vis(uvtable)
    m_gpu.process_vis(uvtable)

    for key in ("u", "v", "vis", "weights"):
        a = _to_numpy(m_cpu.gridded_data[key])
        b = _to_numpy(m_gpu.gridded_data[key])
        scale = float(np.abs(a).max()) or 1.0
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12 * scale), \
            f"Gridding mismatch in component {key!r}"


@pytest.mark.gpu
@pytest.mark.slow
def test_fit_cpu_gpu_equivalence(simulated_obs):
    """The fitted image and visibility model must agree between backends."""
    from frank2d.cpu import Frank2D as Frank2D_CPU
    from frank2d.gpu import Frank2D as Frank2D_GPU

    d = simulated_obs
    N, Rmax, uvtable = d["N"], d["Rmax"], d["uvtable"]

    m_cpu = Frank2D_CPU(N, Rmax)
    m_gpu = Frank2D_GPU(N, Rmax)
    m_cpu.fit(uvtable, rtol=1e-10)          # numpy uvtable -> also covers the
    m_gpu.fit(uvtable, rtol=1e-10)          # process_vis numpy-input bug

    img_cpu, img_gpu = _to_numpy(m_cpu.intensity_model), _to_numpy(m_gpu.intensity_model)
    vis_cpu, vis_gpu = _to_numpy(m_cpu.visibility_model), _to_numpy(m_gpu.visibility_model)

    img_scale = float(np.abs(img_cpu).max()) or 1.0
    vis_scale = float(np.abs(vis_cpu).max()) or 1.0
    assert np.allclose(img_cpu, img_gpu, rtol=1e-6, atol=1e-8 * img_scale), \
        "Intensity model differs between CPU and GPU."
    assert np.allclose(vis_cpu, vis_gpu, rtol=1e-6, atol=1e-8 * vis_scale), \
        "Visibility model differs between CPU and GPU."