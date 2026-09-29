import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_reports_times_and_reference_errors(tmp_path):
    pytest.importorskip("scipy")
    report = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    result = subprocess.run([
        sys.executable, str(ROOT / "examples/chapter3_exercises/hopfbi_gpu_cpu_benchmarking.py"),
        "--counts", "1", "2", "--steps", "20", "--repeats", "2",
        "--json", str(report), "--csv", str(csv_path),
    ], text=True, capture_output=True, check=True)
    data = json.loads(report.read_text())
    assert "max error vs reference=" in result.stdout
    assert "median=" in result.stdout
    for count in (1, 2):
        rows = [row for row in data["rows"] if row["batch"] == count]
        assert {row["backend"] for row in rows} >= {"C++ CPU (Python RHS)", "NumPy CPU"}
        cpp = rows[0]
        for row in rows:
            assert row["median_ms"] >= row["min_ms"] > 0
            assert row["speedup_vs_cpp"] == pytest.approx(cpp["median_ms"] / row["median_ms"])
            assert row["agrees_with_cpp"]
            assert row["max_abs_diff_cpp"] < 1e-8
            assert row["max_abs_error_ref"] >= row["rmse_ref"] > 1e-5
    assert "median_ms" in csv_path.read_text()


def test_reference_shape_and_euler_error_convergence():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location(
        "hopf_benchmark", ROOT / "examples/chapter3_exercises/hopfbi_gpu_cpu_benchmarking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = np.array([[0, 2], [2, 5]], dtype=float)
    ref = module.reference_solution(initial, 0.01, 20)
    assert ref.shape == (2, 21, 2)
    np.testing.assert_allclose(ref[:, 0], initial)
    np.testing.assert_allclose(ref[1], np.tile([2, 5], (21, 1)), atol=1e-12)
    coarse = module.numpy_batch(initial, 0.01, 20)
    fine = module.numpy_batch(initial, 0.005, 40)[:, ::2]
    assert np.max(np.abs(fine - ref)) < 0.6 * np.max(np.abs(coarse - ref))


def test_notebook_streams_stdout_stderr_and_failure(capsys):
    notebook = json.loads((ROOT / "notebooks/colab_quickstart.ipynb").read_text())
    source = "".join(notebook["cells"][1]["source"])
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook", "exec"), namespace)
    with pytest.raises(subprocess.CalledProcessError):
        namespace["run_logged"]([sys.executable, "-c",
            "import sys; print('timing row'); print('test error', file=sys.stderr); sys.exit(2)"])
    lines = capsys.readouterr().out.splitlines()
    assert "timing row" in lines
    assert "test error" in lines
