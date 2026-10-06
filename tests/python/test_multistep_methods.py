"""The BDF binding solves nonlinear scalar IVPs, including Python callbacks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from odelab import MultiStepMethod, solve_linear_multi_step


def rhs(t, y):
    return -5*t*y*y + 5/t - 1/(t*t)


def derivative(t, y):
    return -10*t*y


@pytest.mark.parametrize("order", range(1, 7))
def test_bdf_orders_and_exact_solution(order):
    steps = 100
    startup = [1/(1+j/steps) for j in range(1, order)] if order > 2 else []
    t, y = solve_linear_multi_step(rhs, 1, 1, 2, steps, order=order,
                                    df_dy=derivative, startup_values=startup)
    np.testing.assert_allclose(t, np.linspace(1, 2, steps+1), atol=1e-15)
    assert len(y) == steps+1
    assert abs(y[-1]-.5) < {1: 2e-4, 2: 2e-6, 3: 4e-8, 4: 1e-9,
                            5: 1e-10, 6: 1e-10}[order]


def test_bdf2_converges_and_numerical_derivative_matches():
    errors = []
    for steps in (100, 200):
        analytic = solve_linear_multi_step(rhs, 1, 1, 2, steps, order=2, df_dy=derivative)
        numerical = solve_linear_multi_step(rhs, 1, 1, 2, steps, order=2)
        np.testing.assert_allclose(analytic.y, numerical.y, rtol=1e-9, atol=1e-10)
        errors.append(abs(analytic.y[-1]-.5))
    assert 3.7 < errors[0]/errors[1] < 4.3


def test_bdf_invalid_startup_and_newton_options():
    with pytest.raises(ValueError, match="startup"):
        solve_linear_multi_step(rhs, 1, 1, 2, 100, order=4)
    with pytest.raises(ValueError, match="startup"):
        solve_linear_multi_step(rhs, 1, 1, 2, 100, order=2, startup_values=[1, 2])
    with pytest.raises(ValueError, match="order"):
        solve_linear_multi_step(rhs, 1, 1, 2, 100, order=7)
    with pytest.raises(ValueError, match="Newton"):
        solve_linear_multi_step(rhs, 1, 1, 2, 100, newton_atol=0)
    with pytest.raises(TypeError):
        solve_linear_multi_step(rhs, 1, 1, 2, 100, df_dy=3)


def test_bdf_callback_failure_propagates():
    def failing(t, y):
        raise ValueError("RHS callback failed")
    with pytest.raises(ValueError, match="RHS callback failed"):
        solve_linear_multi_step(failing, 1, 1, 2, 10)


def test_bdf_benchmark_writes_metrics(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    root = Path(__file__).resolve().parents[2]
    script = root / "examples/chapter3_exercises/bdf_nonlinear_benchmark.py"
    command = [sys.executable, str(script), "--steps", "20", "40", "--repeats", "1",
               "--output", str(tmp_path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "max error=" in result.stdout
    report = json.loads((tmp_path / "results.json").read_text())
    assert len(report["rows"]) == 4
    assert {row["solver"] for row in report["rows"]} == {
        "odelab fixed BDF2", "SciPy adaptive BDF"}
    assert all(row["median_ms"] > 0 and row["max_abs_error"] > 0 for row in report["rows"])
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "comparison.png").is_file()
