"""Benchmark odelab's fixed-step BDF against SciPy's adaptive BDF.

Problem: y'=-5*t*y**2+5/t-1/t**2, y(1)=1, 1<=t<=2.
The exact solution y(t)=1/t makes both timing and accuracy visible.
SciPy uses adaptive step sizes constrained by max_step=h and samples the
same output grid. It is not the same fixed-order algorithm as odelab.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from odelab import solve_linear_multi_step


def rhs(t, y):
    return -5.0*t*y*y + 5.0/t - 1.0/(t*t)


def derivative(t, y):
    return -10.0*t*y


def timed(call, repeats):
    call()  # Exclude one warm-up call for each backend.
    samples = []
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = call()
        samples.append(1000.0*(perf_counter()-start))
    return value, float(np.median(samples)), float(min(samples))


def compare(steps, order, repeats):
    h = 1.0/steps
    grid = np.linspace(1.0, 2.0, steps+1)
    # Higher fixed orders require accurate history; use known exact values.
    startup = [1.0/grid[j] for j in range(1, order)] if order > 2 else []
    own, own_median, own_min = timed(
        lambda: solve_linear_multi_step(
            rhs, 1.0, 1.0, 2.0, steps, order=order,
            df_dy=derivative, startup_values=startup), repeats)

    def scipy_call():
        result = solve_ivp(
            lambda t, y: [rhs(t, y[0])], (1.0, 2.0), [1.0],
            method="BDF", jac=lambda t, y: [[derivative(t, y[0])]],
            t_eval=grid, max_step=h, rtol=1e-10, atol=1e-12)
        if not result.success:
            raise RuntimeError(f"SciPy BDF failed: {result.message}")
        return result

    other, scipy_median, scipy_min = timed(scipy_call, repeats)
    np.testing.assert_allclose(own.t, grid, rtol=0, atol=2e-15)
    exact = 1.0/grid
    own_y, scipy_y = np.asarray(own.y), other.y[0]
    rows = []
    for name, y, median, minimum, evaluations in [
        (f"odelab fixed BDF{order}", own_y, own_median, own_min, None),
        ("SciPy adaptive BDF", scipy_y, scipy_median, scipy_min, int(other.nfev)),
    ]:
        rows.append({
            "steps": steps, "solver": name, "median_ms": median,
            "min_ms": minimum, "max_abs_error": float(np.max(np.abs(y-exact))),
            "final_abs_error": float(abs(y[-1]-exact[-1])),
            "max_abs_difference": float(np.max(np.abs(own_y-scipy_y))),
            "rhs_evaluations": evaluations,
        })
    return rows, {"t": grid, "exact": exact, "odelab": own_y, "scipy": scipy_y}


def save_plot(output, rows, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    grid = trajectories["t"]
    axes[0].plot(grid, trajectories["exact"], "k-", label="Exact 1/t")
    axes[0].plot(grid, trajectories["odelab"], "--", label="odelab")
    axes[0].plot(grid, trajectories["scipy"], ":", label="SciPy")
    axes[0].set(xlabel="t", ylabel="y(t)", title="Solution on the finest grid")
    for name in dict.fromkeys(row["solver"] for row in rows):
        selected = [row for row in rows if row["solver"] == name]
        x = [row["steps"] for row in selected]
        axes[1].plot(x, [row["median_ms"] for row in selected], "o-", label=name)
        axes[2].plot(x, [row["max_abs_error"] for row in selected], "o-", label=name)
    axes[1].set(xlabel="Output steps", ylabel="Median elapsed ms", title="Runtime", xscale="log", yscale="log")
    axes[2].set(xlabel="Output steps", ylabel="Maximum absolute error", title="Error vs exact solution", xscale="log", yscale="log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("build/bdf_nonlinear"))
    args = parser.parse_args()
    if args.order not in range(1, 7) or min(args.steps) < args.order or args.repeats < 1:
        parser.error("order must be 1..6; steps >= order and repeats >= 1")
    if len(set(args.steps)) != len(args.steps):
        parser.error("step counts must be distinct")
    rows = []
    trajectories = None
    for steps in args.steps:
        more, current = compare(steps, args.order, args.repeats)
        rows.extend(more)
        if trajectories is None or steps > len(trajectories["t"])-1:
            trajectories = current
        print(f"\n{steps} output steps, h={1/steps:g}", flush=True)
        for row in more:
            print(f"  {row['solver']}: median={row['median_ms']:.3f} ms, "
                  f"min={row['min_ms']:.3f} ms, "
                  f"max error={row['max_abs_error']:.6e}, "
                  f"final error={row['final_abs_error']:.6e}, "
                  f"difference={row['max_abs_difference']:.6e}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"equation": "y'=-5*t*y^2+5/t-1/t^2", "initial": "y(1)=1",
              "interval": [1, 2], "exact": "1/t", "order": args.order,
              "repeats": args.repeats, "scipy_rtol": 1e-10, "scipy_atol": 1e-12,
              "rows": rows}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(args.output, rows, trajectories)
    print(f"Saved report and plot in {args.output}", flush=True)


if __name__ == "__main__":
    main()
