"""Benchmark Hopf forward Euler on CPU and GPU, with reference errors.

Elapsed times include the whole solver call. C++ CPU calls a Python RHS;
CUDA evaluates the RHS on-device and includes allocation and transfers.
Reference integration, warm-up, error calculation, and reporting are not timed.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from odelab import cuda_available, solve_hopf_batch_cuda, solve_system_euler


def rhs(t, y):
    # Also supports (batch, 2) states for reference integration.
    y = np.asarray(y)
    y1, y2 = y[..., 0], y[..., 1]
    denominator = 1.0 + y1**2
    return np.stack([
        10.0 - y1 - 4.0 * y1 * y2 / denominator,
        3.6 * y1 * (1.0 - y2 / denominator),
    ], axis=-1)


def numpy_batch(initial, h, steps):
    out = np.empty((len(initial), steps + 1, 2))
    for j, y0 in enumerate(initial):
        out[j, 0] = y0
        for n in range(steps):
            out[j, n + 1] = out[j, n] + h * rhs(n * h, out[j, n])
    return out


def reference_solution(initial, h, steps):
    """DOP853 numerical reference, not an exact solution."""
    t = np.linspace(0.0, h * steps, steps + 1)
    result = solve_ivp(
        lambda t, y: rhs(t, y.reshape(-1, 2)).ravel(),
        (t[0], t[-1]), initial.ravel(), t_eval=t,
        method="DOP853", rtol=1e-11, atol=1e-13,
    )
    if not result.success:
        raise RuntimeError(f"Reference integration failed: {result.message}")
    return result.y.T.reshape(steps + 1, len(initial), 2).transpose(1, 0, 2)


def timed(call, repeats):
    call()  # Warm every backend, including the CUDA context, outside timings.
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append((perf_counter() - start) * 1000)
    return result, samples


def benchmark(counts, steps, h, repeats, use_cuda):
    rows = []
    for count in counts:
        print(f"\nBatch {count}: {steps} steps, h={h:g}, {repeats} timed repeats", flush=True)
        initial = np.column_stack((np.linspace(0.0, 0.1, count), np.full(count, 2.0)))
        reference = reference_solution(initial, h, steps)
        calls = [
            ("C++ CPU (Python RHS)", lambda: np.stack([
                solve_system_euler(rhs, row.tolist(), 0.0, h * steps, steps)[1]
                for row in initial
            ])),
            ("NumPy CPU", lambda: numpy_batch(initial, h, steps)),
        ]
        if use_cuda:
            calls.append(("CUDA GPU (total)", lambda: solve_hopf_batch_cuda(
                initial, 10.0, 3.6, 0.0, h * steps, steps)))
        cpp, cpp_ms = None, None
        for backend, call in calls:
            result, samples = timed(call, repeats)
            elapsed = float(np.median(samples))
            if cpp is None:
                cpp, cpp_ms = result, elapsed
            delta = result - reference
            row = {
                "batch": count,
                "backend": backend,
                "median_ms": elapsed,
                "min_ms": float(min(samples)),
                "speedup_vs_cpp": cpp_ms / elapsed,
                "max_abs_error_ref": float(np.max(np.abs(delta))),
                "rmse_ref": float(np.sqrt(np.mean(delta**2))),
                "max_abs_diff_cpp": float(np.max(np.abs(result - cpp))),
                "agrees_with_cpp": bool(np.allclose(result, cpp, rtol=1e-8, atol=1e-8)),
            }
            rows.append(row)
            print(
                f"  {backend:22s} median={elapsed:10.3f} ms  min={row['min_ms']:10.3f} ms"
                f"  speedup={row['speedup_vs_cpp']:.2f}x\n"
                f"    max error vs reference={row['max_abs_error_ref']:.6e}"
                f"  RMSE={row['rmse_ref']:.6e}"
                f"  max difference vs C++={row['max_abs_diff_cpp']:.6e}"
                f"  agreement={'PASS' if row['agrees_with_cpp'] else 'FAIL'}",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", type=Path, help="Save results for notebook display")
    parser.add_argument("--csv", type=Path, help="Save a benchmark table")
    args = parser.parse_args()
    if min(args.counts) < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("counts, steps, and repeats must be positive")
    if not np.isfinite(args.h) or args.h <= 0 or not np.isfinite(args.h * args.steps):
        parser.error("h and final time must be positive and finite")
    use_cuda = cuda_available()
    if args.require_cuda and not use_cuda:
        parser.error("CUDA unavailable: select a GPU runtime and install with ODELAB_ENABLE_CUDA=ON")
    print(__doc__, flush=True)
    print("Reference: SciPy DOP853, rtol=1e-11, atol=1e-13; errors cover all samples.", flush=True)
    if not use_cuda:
        print("CUDA unavailable; GPU results are omitted, not measured as zero.", flush=True)
    rows = benchmark(args.counts, args.steps, args.h, args.repeats, use_cuda)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "steps": args.steps, "h": args.h, "repeats": args.repeats,
            "cuda_available": use_cuda, "reference": "DOP853 (rtol=1e-11, atol=1e-13)",
            "rows": rows,
        }, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.json}", flush=True)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.csv}", flush=True)
    if not all(row["agrees_with_cpp"] for row in rows):
        raise RuntimeError("Backend agreement failed; see the reported differences and saved results")


if __name__ == "__main__":
    main()
