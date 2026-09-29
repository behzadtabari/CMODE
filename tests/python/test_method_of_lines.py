import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/chapter3_exercises/method_of_line_1D_cpu_gpu.py"
spec = importlib.util.spec_from_file_location("mol_example", SCRIPT)
mol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mol)


def test_random_nodes_are_unrestricted_uniform_samples():
    expected = np.sort(np.random.default_rng(123).uniform(0, 1, 50))
    x = mol.random_grid(50, np.random.default_rng(123))
    np.testing.assert_array_equal(x[1:-1], expected)
    assert x[0] == 0 and x[-1] == 1
    assert np.all(np.diff(x) > 0)
    rng = np.random.default_rng(123)
    assert not np.array_equal(mol.random_grid(50, rng), mol.random_grid(50, rng))


def test_nonuniform_operator_is_exact_for_quadratic():
    x = np.array([0, 0.000001, 0.12, 0.121, 0.6, 0.99, 1])
    volumes, stiffness = mol.spatial_operator(x)
    u = x[1:-1] * (1 - x[1:-1])
    np.testing.assert_allclose(stiffness @ u, -2 * volumes, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(stiffness.toarray(), stiffness.toarray().T)


@pytest.mark.parametrize("solver", [mol.solve_cn, mol.solve_numpy_cn, mol.solve_native_cn])
def test_crank_nicolson_second_order_in_time(solver):
    count = 15
    x = np.linspace(0, 1, count + 2)
    u0 = mol.exact_solution(x, 0)[1:-1]
    # Exact solution of the semidiscrete system on a uniform grid.
    eigenvalue = -4 * (count + 1)**2 * np.sin(np.pi / (2 * (count + 1)))**2
    target = np.exp(eigenvalue * 0.1) * u0
    errors = []
    for steps in (10, 20):
        left, right = mol.cn_matrices(x, 0.1 / steps)
        result, _ = solver(left, right, u0, steps)
        errors.append(np.max(np.abs(result - target)))
    assert 3.9 < errors[0] / errors[1] < 4.1


@pytest.mark.parametrize("count", [1, 64, 256])
def test_random_grid_solution_and_accuracy(count):
    x = mol.random_grid(count, np.random.default_rng(17))
    left, right = mol.cn_matrices(x, 0.1 / 200)
    u, timing = mol.solve_cn(left, right, mol.exact_solution(x, 0)[1:-1], 200)
    assert np.all(np.isfinite(u))
    assert timing["total_ms"] >= timing["solve_ms"] > 0
    if count >= 64:
        assert np.max(np.abs(u - mol.exact_solution(x, 0.1)[1:-1])) < 0.003


def test_cli_report_and_grid_replay(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "run"
    command = [sys.executable, str(SCRIPT), "--cpu-only", "--nodes", "12",
               "--steps", "10", "--repeats", "1", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads((output / "results.json").read_text())
    assert {r["backend"] for r in report["rows"]} == {
        "C++ CPU (Thomas)", "NumPy/Python (Thomas)", "SciPy CPU (sparse LU)"}
    assert all(r["agrees_with_cpu"] for r in report["rows"])
    row = report["rows"][0]
    assert row["median_solve_ms"] > 0
    assert row["max_abs_error"] > 0
    assert not report["cuda_available"]
    assert "max error=" in result.stdout
    with np.load(output / "grids_and_solutions.npz") as data:
        np.testing.assert_array_equal(data["x_12"], mol.random_grid(12, np.random.default_rng(report["seed"])))
        assert data["cpu_12"][0] == data["cpu_12"][-1] == 0
    assert (output / "results.csv").is_file()
    assert (output / "comparison.png").is_file()


@pytest.mark.parametrize("count", [1, 2, 3, 7, 64, 65, 256])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_agrees_on_random_grid(count, backend):
    if backend == "cuda" and not mol.cuda_available():
        pytest.skip("Native CUDA backend/device unavailable")
    x = mol.random_grid(count, np.random.default_rng(19))
    left, right = mol.cn_matrices(x, 0.1 / 100)
    initial = mol.exact_solution(x, 0)[1:-1]
    reference, _ = mol.solve_cn(left, right, initial, 100)
    result, timing = mol.solve_native_cn(left, right, initial, 100, backend)
    np.testing.assert_allclose(result, reference, rtol=1e-7, atol=1e-9)
    assert timing["total_ms"] >= timing["solve_ms"] > 0


@pytest.mark.parametrize("count", [1, 8, 64])
def test_numpy_baseline_matches_scipy(count):
    x = mol.random_grid(count, np.random.default_rng(42))
    left, right = mol.cn_matrices(x, 0.001)
    initial = mol.exact_solution(x, 0)[1:-1]
    original = initial.copy()
    numpy, timing = mol.solve_numpy_cn(left, right, initial, 100)
    scipy, _ = mol.solve_cn(left, right, initial, 100)
    np.testing.assert_allclose(numpy, scipy, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(initial, original)
    assert timing["total_ms"] >= timing["solve_ms"] > 0


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_input_validation(backend):
    if backend == "cuda" and not mol.cuda_available():
        pytest.skip("Native CUDA backend/device unavailable")
    solve = mol.solve_tridiagonal_cn
    with pytest.raises(ValueError, match="matching"):
        solve([], [1, 1], [], [], [1, 1], [], [1, 2], 10, backend=backend)
    with pytest.raises(ValueError, match="steps"):
        solve([], [1], [], [], [1], [], [1], 0, backend=backend)
    with pytest.raises(ValueError, match="finite"):
        solve([], [1], [], [], [1], [], [np.nan], 1, backend=backend)
    with pytest.raises(ValueError, match="dominant"):
        solve([-2], [1, 1], [-2], [0], [1, 1], [0], [1, 2], 1, backend=backend)
    with pytest.raises(RuntimeError, match="pivot"):
        solve([-1], [1, 1], [-1], [0], [1, 1], [0], [1, 2], 1, backend=backend)


def test_native_right_matrix_and_initial_are_not_modified():
    x = mol.random_grid(8, np.random.default_rng(1))
    left, right = mol.cn_matrices(x, 0.01)
    before_left, before_right = left.toarray(), right.toarray()
    initial = mol.exact_solution(x, 0)[1:-1]
    original = initial.copy()
    mol.solve_native_cn(left, right, initial, 10)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(left.toarray(), before_left)
    np.testing.assert_array_equal(right.toarray(), before_right)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_general_coupled_linear_ode(backend):
    if backend == "cuda" and not mol.cuda_available():
        pytest.skip("Native CUDA backend/device unavailable")
    generator = np.array([[-2, 1, 0], [0.5, -1, 0.25], [0, 0.75, -3]])
    left = np.eye(3) - 0.025 * generator
    right = np.eye(3) + 0.025 * generator
    initial = np.array([1.0, -2.0, 3.0])
    expected = np.linalg.matrix_power(np.linalg.solve(left, right), 20) @ initial
    actual, _ = mol.solve_tridiagonal_cn(
        np.diag(left, -1), np.diag(left), np.diag(left, 1),
        np.diag(right, -1), np.diag(right), np.diag(right, 1),
        initial, 20, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
