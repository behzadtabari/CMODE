import numpy as np
import pytest

from odelab import cuda_available, solve_hopf_batch_cuda, solve_system_euler


@pytest.mark.parametrize("initial", [[0, 2], [[0, 2, 3]], [[], []]])
def test_initial_shape(initial):
    with pytest.raises(ValueError, match="shape"):
        solve_hopf_batch_cuda(initial, 10, 3.6, 0, 1, 10)


def test_cpu_build_error():
    from odelab import _core
    # A CUDA build without an accessible GPU is distinct from a CPU build.
    if cuda_available():
        pytest.skip("CUDA device available")
    with pytest.raises(RuntimeError):
        _core.solve_hopf_batch_cuda([0, 2], 10, 3.6, 0, 1, 10)


@pytest.mark.skipif(not cuda_available(), reason="CUDA backend/device unavailable")
def test_hopf_matches_cpu():
    initial = np.array([[0, 2], [0.1, 2], [2, 5]], dtype=float)
    def rhs(t, y):
        x, z = y
        d = 1 + x*x
        return [10 - x - 4*x*z/d, 3.6*x*(1-z/d)]
    gpu = solve_hopf_batch_cuda(initial, 10, 3.6, 0, 20, 2000)
    cpu = np.stack([solve_system_euler(rhs, row, 0, 20, 2000)[1] for row in initial])
    assert gpu.shape == (3, 2001, 2)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-8, atol=1e-8)
    assert solve_hopf_batch_cuda(np.empty((0, 2)), 10, 3.6, 0, 1, 10).shape == (0, 11, 2)


@pytest.mark.skipif(not cuda_available(), reason="CUDA backend/device unavailable")
@pytest.mark.parametrize("initial,steps", [([[0, 2]], 0), ([[np.nan, 2]], 10)])
def test_invalid_cuda_inputs(initial, steps):
    with pytest.raises(ValueError):
        solve_hopf_batch_cuda(initial, 10, 3.6, 0, 1, steps)
