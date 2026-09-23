import numpy as np
import pytest

from odelab import (Method, cuda_available, solve, solve_linear,
                    solve_linear_batch, solve_linear_batch_cuda)

METHODS = (Method.FORWARD_EULER, Method.BACKWARD_EULER, Method.TRAPEZOIDAL)


@pytest.mark.parametrize("method,expected", [
    (Method.FORWARD_EULER, -1.0),
    (Method.BACKWARD_EULER, 1.0 / 3.0),
    (Method.TRAPEZOIDAL, 0.0),
])
def test_one_linear_step(method, expected):
    t, y = solve_linear(-2.0, 0.0, 1.0, 0.0, 1.0, 1, method)
    np.testing.assert_allclose(t, [0.0, 1.0])
    assert y[-1] == pytest.approx(expected)


@pytest.mark.parametrize("method", METHODS)
def test_generic_callback_matches_compiled_linear(method):
    t, y = solve(lambda t, y: -2.0 * y + 0.5, 1.0, 0.0, 1.0, 20, method)
    t_ref, y_ref = solve_linear(-2.0, 0.5, 1.0, 0.0, 1.0, 20, method)
    np.testing.assert_allclose(t, t_ref)
    np.testing.assert_allclose(y, y_ref, rtol=1e-10, atol=1e-12)


def test_trapezoidal_converges_quadratically():
    def error(n):
        _, y = solve_linear(-1.0, 0.0, 1.0, 0.0, 1.0, n, Method.TRAPEZOIDAL)
        return abs(y[-1] - np.exp(-1.0))
    assert error(40) < error(20) / 3.5


def test_implicit_nonlinear_newton():
    _, y = solve(lambda t, y: -y * y, 1.0, 0.0, 1.0, 100,
                 Method.TRAPEZOIDAL)
    assert y[-1] == pytest.approx(0.5, abs=1e-4)


def test_batch_rows_and_invalid_step():
    matrix = solve_linear_batch([-1.0, -2.0], [0.0, 0.0], [1.0, 2.0],
                                0.0, 1.0, 5, Method.FORWARD_EULER)
    assert matrix.shape == (2, 6)
    assert matrix[1, -1] == pytest.approx(
        solve_linear(-2.0, 0.0, 2.0, 0.0, 1.0, 5, Method.FORWARD_EULER)[1][-1])
    with pytest.raises((ValueError, RuntimeError)):
        solve_linear(1.0, 0.0, 1.0, 0.0, 1.0, 1, Method.BACKWARD_EULER)


@pytest.mark.skipif(not cuda_available(), reason="CUDA backend/device unavailable")
@pytest.mark.parametrize("method", METHODS)
def test_cuda_matches_cpu(method):
    a, b, y0 = [-2.0, -1.0, -0.5], [0.0, 0.3, -0.2], [1.0, 2.0, -1.0]
    expected = solve_linear_batch(a, b, y0, 0.0, 2.0, 100, method)
    actual = solve_linear_batch_cuda(a, b, y0, 0.0, 2.0, 100, method)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)