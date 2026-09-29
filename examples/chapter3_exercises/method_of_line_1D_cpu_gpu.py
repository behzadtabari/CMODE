"""Random-grid method of lines for u_t = u_xx on a unit rod.

Example conditions: u(0,t)=u(1,t)=0, u(x,0)=sin(pi*x), alpha=1.
Crank-Nicolson in time; conservative three-point differences in space.
The spatial discretization gives the ODE system M*u'=K*u.
Compare NumPy/Python Thomas, SciPy LU, and odelab native C++/CUDA solvers.
Each run draws fresh independent uniform interior nodes; both backends and
all timing repeats share the same grid. Use --seed only to reproduce a run.
"""
import argparse
import csv
import json
import secrets
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from odelab import cuda_available, solve_tridiagonal_cn

ALPHA = 1.0
LENGTH = 1.0


def random_grid(interior_nodes, rng):
    if interior_nodes < 1:
        raise ValueError("interior_nodes must be positive")
    x = np.concatenate(([0.0], np.sort(rng.uniform(0.0, LENGTH, interior_nodes)), [LENGTH]))
    if np.any(np.diff(x) <= 0):
        raise ValueError("Random draw produced duplicate nodes; rerun with a new seed")
    return x


def exact_solution(x, t):
    values = np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)
    values = np.array(values, dtype=np.float64)
    values[[0, -1]] = 0.0
    return values


def spatial_operator(x):
    """M u' = K u, with zero Dirichlet boundary values eliminated.

    M_ii=(h_left+h_right)/2; K_ii=-(1/h_left+1/h_right).
    Hence (M^-1 K u)_i = 2[(u_{i+1}-u_i)/h_right
                                     -(u_i-u_{i-1})/h_left]/(h_left+h_right).
    The symmetric mass form avoids explicitly scaling rows by tiny cell widths.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional grid with interior nodes")
    gaps = np.diff(x)
    if x[0] != 0 or x[-1] != LENGTH or np.any(gaps <= 0):
        raise ValueError("x must increase strictly from 0 to 1")
    volumes = (gaps[:-1] + gaps[1:]) / 2
    stiffness = diags(
        [ALPHA / gaps[1:-1], -ALPHA * (1 / gaps[:-1] + 1 / gaps[1:]),
         ALPHA / gaps[1:-1]], [-1, 0, 1], format="csc")
    return volumes, stiffness


def cn_matrices(x, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    volumes, stiffness = spatial_operator(x)
    mass = diags(volumes, format="csc")
    return mass - (dt / 2) * stiffness, mass + (dt / 2) * stiffness


def solve_numpy_cn(left, right, initial, steps):
    """NumPy/Python baseline: factor a tridiagonal matrix once, then reuse it.

    Uses the Thomas algorithm with NumPy arrays and Python elimination loops.
    SciPy supplies the shared matrix representation, but does not solve here.
    This exploits the same tridiagonal structure as the planned C++ CPU solver.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    total_start = perf_counter()
    lower = left.diagonal(-1).copy()
    diagonal = left.diagonal().copy()
    upper = left.diagonal(1).copy()
    right_lower = right.diagonal(-1)
    right_diagonal = right.diagonal()
    right_upper = right.diagonal(1)
    u = initial.copy()
    for i in range(1, len(diagonal)):
        lower[i - 1] /= diagonal[i - 1]
        diagonal[i] -= lower[i - 1] * upper[i - 1]
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise RuntimeError("Crank-Nicolson factorization produced invalid pivots")
    setup_end = perf_counter()
    for _ in range(steps):
        rhs = right_diagonal * u
        rhs[1:] += right_lower * u[:-1]
        rhs[:-1] += right_upper * u[1:]
        for i in range(1, len(u)):
            rhs[i] -= lower[i - 1] * rhs[i - 1]
        u[-1] = rhs[-1] / diagonal[-1]
        for i in range(len(u) - 2, -1, -1):
            u[i] = (rhs[i] - upper[i] * u[i + 1]) / diagonal[i]
    end = perf_counter()
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite NumPy Crank-Nicolson solution")
    return u, {
        "setup_ms": 1000 * (setup_end - total_start),
        "solve_ms": 1000 * (end - setup_end),
        "total_ms": 1000 * (end - total_start),
    }


def solve_cn(left, right, initial, steps):
    """SciPy baseline: factor once, then reuse sparse LU in the ODE loop."""
    if steps < 1:
        raise ValueError("steps must be positive")
    start = perf_counter()
    factor = splu(left, permc_spec="NATURAL")
    operator = right.tocsr()
    u = initial.copy()
    ready = perf_counter()
    for _ in range(steps):
        u = factor.solve(operator @ u)
    end = perf_counter()
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite SciPy solution")
    return u, {"setup_ms": 1000 * (ready-start), "solve_ms": 1000 * (end-ready),
               "total_ms": 1000 * (end-start)}


def solve_native_cn(left, right, initial, steps, backend="cpu"):
    start = perf_counter()
    u, timing = solve_tridiagonal_cn(
        left.diagonal(-1), left.diagonal(), left.diagonal(1),
        right.diagonal(-1), right.diagonal(), right.diagonal(1),
        initial, steps, backend=backend,
    )
    timing["total_ms"] = 1000 * (perf_counter() - start)
    return u, timing


def measure(call, repeats):
    call()  # Warm up every backend before collecting elapsed times.
    samples = []
    for _ in range(repeats):
        solution, timing = call()
        samples.append(timing)
    summary = {"median_" + key: float(np.median([s[key] for s in samples]))
               for key in samples[0]}
    summary["min_solve_ms"] = min(s["solve_ms"] for s in samples)
    summary["min_total_ms"] = min(s["total_ms"] for s in samples)
    return solution, summary


def benchmark(nodes, steps, final_time, repeats, rng, use_cuda=False):
    rows, arrays = [], {}
    for count in nodes:
        x = random_grid(count, rng)
        dt = final_time / steps
        left, right = cn_matrices(x, dt)
        initial, exact = exact_solution(x, 0), exact_solution(x, final_time)
        volumes, _ = spatial_operator(x)
        gaps = np.diff(x)
        print(f"\n{count} interior nodes: min dx={gaps.min():.3e}, max dx={gaps.max():.3e}", flush=True)
        arrays[f"x_{count}"] = x
        arrays[f"exact_{count}"] = exact
        cpu = None
        cpu_timing = None
        calls = [
            ("C++ CPU (Thomas)", "cpu", lambda: solve_native_cn(left, right, initial[1:-1], steps)),
            ("NumPy/Python (Thomas)", "numpy", lambda: solve_numpy_cn(left, right, initial[1:-1], steps)),
            ("SciPy CPU (sparse LU)", "scipy", lambda: solve_cn(left, right, initial[1:-1], steps)),
        ]
        if use_cuda:
            calls.append(("C++ CUDA (PCR)", "gpu", lambda: solve_native_cn(
                left, right, initial[1:-1], steps, "cuda")))
        for name, key, call in calls:
            interior, timing = measure(call, repeats)
            solution = np.concatenate(([0.0], interior, [0.0]))
            if cpu is None:
                cpu, cpu_timing = solution, timing
            error = solution - exact
            row = {
                "interior_nodes": count, "backend": name,
                "min_dx": float(gaps.min()), "max_dx": float(gaps.max()),
                **timing,
                "max_abs_error": float(np.max(np.abs(error))),
                "weighted_l2_error": float(np.sqrt(np.sum(volumes * error[1:-1]**2))),
                "max_abs_diff_cpu": float(np.max(np.abs(solution - cpu))),
                "solve_speedup_vs_cpu": cpu_timing["median_solve_ms"] / timing["median_solve_ms"],
                "total_speedup_vs_cpu": cpu_timing["median_total_ms"] / timing["median_total_ms"],
                "agrees_with_cpu": bool(np.allclose(solution, cpu, rtol=1e-7, atol=1e-9)),
            }
            rows.append(row)
            arrays[f"{key}_{count}"] = solution
            print(f"  {name}: solve={row['median_solve_ms']:.3f} ms, "
                  f"setup={row['median_setup_ms']:.3f} ms, total={row['median_total_ms']:.3f} ms\n"
                  f"    max error={row['max_abs_error']:.6e}, weighted L2={row['weighted_l2_error']:.6e}, "
                  f"difference vs CPU={row['max_abs_diff_cpu']:.3e}, "
                  f"solve speedup={row['solve_speedup_vs_cpu']:.2f}x, "
                  f"agreement={'PASS' if row['agrees_with_cpu'] else 'FAIL'}", flush=True)
    return rows, arrays


def save_plot(output, rows, arrays, nodes, final_time):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    count = nodes[0]
    x = arrays[f"x_{count}"]
    smooth_x = np.linspace(0, 1, 600)
    axes[0, 0].plot(smooth_x, exact_solution(smooth_x, final_time), "k-", label="Exact")
    axes[0, 0].plot(x, arrays[f"cpu_{count}"], "o", fillstyle="none", label="C++ CPU")
    if f"gpu_{count}" in arrays:
        axes[0, 0].plot(x, arrays[f"gpu_{count}"], "+", label="C++ CUDA")
    for key, label, marker in [("numpy", "NumPy/Python", "x"), ("scipy", "SciPy", ".")]:
        axes[0, 0].plot(x, arrays[f"{key}_{count}"], marker, label=label)
    axes[0, 0].set(xlabel="x", ylabel="u(x,T)", title=f"Solution at T={final_time:g} ({count} interior nodes)")
    axes[0, 1].plot(x, np.zeros_like(x), "|", markersize=20)
    axes[0, 1].set(xlabel="x", yticks=[], title="Independent uniform random nodes (same grid for CPU/GPU)")
    for backend in dict.fromkeys(row["backend"] for row in rows):
        series = [r for r in rows if r["backend"] == backend]
        sizes = [r["interior_nodes"] for r in series]
        axes[1, 0].plot(sizes, [r["median_solve_ms"] for r in series], "o-", label=backend)
        axes[1, 1].plot(sizes, [r["max_abs_error"] for r in series], "o--", label=backend)
    axes[1, 0].set(xlabel="Interior nodes", ylabel="Median solve time (ms)", xscale="log", yscale="log", title="Crank–Nicolson time stepping")
    axes[1, 1].set(xlabel="Interior nodes", ylabel="Maximum final-time error", xscale="log", title="Error against exact PDE solution")
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[64, 256, 1024], help="Number of interior nodes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, help="Optional seed for replay; default uses fresh entropy")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("build/method_of_line_1D"))
    args = parser.parse_args()
    if min(args.nodes) < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("nodes, steps and repeats must be positive")
    if len(set(args.nodes)) != len(args.nodes):
        parser.error("node counts must be distinct")
    if not np.isfinite(args.final_time) or args.final_time <= 0 or args.final_time / args.steps == 0:
        parser.error("final time and step size must be positive and finite")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.require_cuda and args.cpu_only:
        parser.error("--cpu-only and --require-cuda cannot be combined")
    use_cuda = not args.cpu_only and cuda_available()
    if args.require_cuda and not use_cuda:
        parser.error("Native CUDA unavailable. Rebuild odelab with ODELAB_ENABLE_CUDA=ON in a GPU runtime.")
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    print(__doc__, flush=True)
    print(f"Seed: {seed}; T={args.final_time:g}; dt={args.final_time / args.steps:g}", flush=True)
    print("Errors are measured at final time. Timings exclude grid assembly, warm-up and plotting.", flush=True)
    print("Native CUDA uses precomputed parallel cyclic reduction; solve time includes synchronized native launches.", flush=True)
    if not use_cuda:
        print("Native CUDA skipped: CPU-only requested or no compiled CUDA backend/device.", flush=True)
    rows, arrays = benchmark(args.nodes, args.steps, args.final_time, args.repeats,
                             np.random.default_rng(seed), use_cuda)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"equation": "u_t = u_xx", "alpha": ALPHA, "length": LENGTH,
              "boundary": "u(0,t)=u(1,t)=0", "initial": "sin(pi*x)",
              "exact": "exp(-pi^2*t)*sin(pi*x)", "seed": seed,
              "steps": args.steps, "final_time": args.final_time, "repeats": args.repeats,
              "cuda_available": use_cuda, "rows": rows}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(args.output / "grids_and_solutions.npz", **arrays)
    save_plot(args.output, rows, arrays, args.nodes, args.final_time)
    print(f"Saved table, grids, solutions and plot in {args.output}", flush=True)
    if not all(row["agrees_with_cpu"] for row in rows):
        raise RuntimeError("CPU/GPU disagreement; inspect the saved errors and random grid")


if __name__ == "__main__":
    main()
