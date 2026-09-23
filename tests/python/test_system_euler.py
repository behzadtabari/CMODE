import numpy as np
import pytest

from odelab import solve_system_euler


def test_coupled_step_uses_old_state():
    t, y = solve_system_euler(lambda t, y: [y[1], -y[0]], [1, 2], 0, 0.1, 1)
    np.testing.assert_allclose(t, [0, 0.1])
    np.testing.assert_allclose(y, [[1, 2], [1.2, 1.9]])


def test_time_dependent_rhs():
    t, y = solve_system_euler(lambda t, y: [t, 2 * t], [0, 0], 1, 2, 2)
    np.testing.assert_allclose(t, [1, 1.5, 2])
    np.testing.assert_allclose(y, [[0, 0], [0.5, 1], [1.25, 2.5]])


def test_euler_convergence():
    errors = []
    for steps in (100, 200):
        _, y = solve_system_euler(lambda t, y: [y[1], -y[0]], [1, 0], 0, 1, steps)
        errors.append(np.linalg.norm(y[-1] - [np.cos(1), -np.sin(1)]))
    assert 1.9 < errors[0] / errors[1] < 2.1


@pytest.mark.parametrize("y0,steps", [([], 1), ([np.nan], 1), ([1], 0)])
def test_invalid_inputs(y0, steps):
    with pytest.raises(ValueError):
        solve_system_euler(lambda t, y: y, y0, 0, 1, steps)


def test_invalid_rhs():
    with pytest.raises(ValueError, match="dimension"):
        solve_system_euler(lambda t, y: [1], [1, 2], 0, 1, 2)
    with pytest.raises(RuntimeError, match="non-finite"):
        solve_system_euler(lambda t, y: [np.inf], [1], 0, 1, 2)


def test_callback_exception():
    def rhs(t, y):
        raise ValueError("RHS failed")
    with pytest.raises(ValueError, match="RHS failed"):
        solve_system_euler(rhs, [1, 2], 0, 1, 2)
