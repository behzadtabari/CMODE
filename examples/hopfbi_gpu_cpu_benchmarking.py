"""Compare Euler implementations; CUDA timing includes transfers and allocation.

The CPU C++ loop calls a Python RHS; CUDA evaluates the RHS on the GPU.
This is an end-to-end comparison, not a pure C++ versus CUDA kernel benchmark.
"""
import argparse

from time import perf_counter

import numpy as np
from odelab import (
    cuda_available,
    solve_hopf_batch_cuda,
    solve_system_euler,
)

alpha, beta = 10.0, 3.6
t0, h, steps = 0.0, 0.01, 2000
t1 = t0 + h * steps


def f(t, y):
    y1, y2 = y
    denominator = 1.0 + y1**2
    return np.array([
        alpha - y1 - 4.0 * y1 * y2 / denominator,
        beta * y1 * (1.0 - y2 / denominator),
    ])


def numpy_euler(y0):
    y = np.empty((steps + 1, 2))
    y[0] = y0
    for n in range(steps):
        y[n + 1] = y[n] + h * f(t0 + n * h, y[n])
    return y


def cpp_batch(initial):
    return np.stack([
        solve_system_euler(f, row.tolist(), t0, t1, steps)[1]
        for row in initial
    ])


def numpy_batch(initial):
    return np.stack([numpy_euler(row) for row in initial])


def timed(call, repeats=3):
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append((perf_counter() - start) * 1000)
    return result, float(np.median(samples))


def main():
    global steps, t1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if min(args.counts) < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("counts, steps, and repeats must be positive")
    steps = args.steps
    t1 = t0 + h * steps
    use_cuda = cuda_available()
    if args.require_cuda and not use_cuda:
        parser.error("CUDA unavailable: select a GPU runtime and install with ODELAB_ENABLE_CUDA=ON")
    print("C++ uses a Python RHS; CUDA timing includes allocation and transfers.")
    if not use_cuda:
        print("CUDA unavailable; running CPU comparisons only.")
    print("Batch | C++ + Python RHS | NumPy loop | CUDA total | max difference")
    for count in args.counts:
        initial = np.column_stack((np.linspace(0.0, 0.1, count), np.full(count, 2.0)))
        if use_cuda:
            solve_hopf_batch_cuda(initial, alpha, beta, t0, t1, steps)
        cpp, cpp_ms = timed(lambda: cpp_batch(initial), args.repeats)
        numpy, numpy_ms = timed(lambda: numpy_batch(initial), args.repeats)
        np.testing.assert_allclose(cpp, numpy, rtol=1e-8, atol=1e-8)
        difference = np.max(np.abs(cpp - numpy))
        gpu_label = "skipped"
        if use_cuda:
            gpu, gpu_ms = timed(
                lambda: solve_hopf_batch_cuda(initial, alpha, beta, t0, t1, steps),
                args.repeats,
            )
            np.testing.assert_allclose(gpu, cpp, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(gpu, numpy, rtol=1e-8, atol=1e-8)
            difference = max(difference, np.max(np.abs(gpu - cpp)))
            gpu_label = f"{gpu_ms:.3f} ms"
        print(f"{count:5d} | {cpp_ms:16.3f} ms | {numpy_ms:10.3f} ms | "
              f"{gpu_label:>12} | {difference:.3e}")
        if count == 1:
            print("Final C++ state:", cpp[0, -1])
            if use_cuda:
                print("Final CUDA state:", gpu[0, -1])


if __name__ == "__main__":
    main()
