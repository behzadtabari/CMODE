"""Numerical ODE solvers implemented in C++."""

import numpy as np
from collections import namedtuple

from . import _core
from ._core import Method, MultiStepMethod, cuda_available

Result = namedtuple("Result", ["t", "y"])


def solve(f, y0, t0, t1, steps, method, tolerance=1e-11, max_iterations=30):
    """Solve a scalar IVP, returning time and state arrays."""
    return Result(*_core.solve(f, y0, t0, t1, steps, method, tolerance, max_iterations))


def solve_linear(a, b, y0, t0, t1, steps, method):
    """Solve y' = a*y + b entirely in C++."""
    return Result(*_core.solve_linear(a, b, y0, t0, t1, steps, method))


def solve_system_euler(f, y0, t0, t1, steps):
    """Forward Euler for a vector RHS, returning NumPy time and state arrays.

    The C++ loop calls ``f(t, y)`` with a Python list and expects one derivative
    per state component. The returned state array has shape (steps + 1, len(y0)).
    """
    t, y = _core.solve_system_euler(f, y0, t0, t1, steps)
    return np.asarray(t), np.asarray(y)


def solve_linear_batch(a, b, y0, t0, t1, steps, method):
    """Return a NumPy array with one trajectory per row."""
    values = _core.solve_linear_batch(a, b, y0, t0, t1, steps, method)
    return np.asarray(values).reshape(len(a), steps + 1)


def solve_linear_batch_cuda(a, b, y0, t0, t1, steps, method):
    """Return GPU trajectories, or raise if CUDA was not compiled in."""
    if not hasattr(_core, "solve_linear_batch_cuda"):
        raise RuntimeError("CUDA backend was not enabled when building odelab")
    values = _core.solve_linear_batch_cuda(a, b, y0, t0, t1, steps, method)
    return np.asarray(values).reshape(len(a), steps + 1)

__all__ = ["Method", "Result", "cuda_available", "solve", "solve_linear",
           "solve_linear_batch", "solve_linear_batch_cuda", "solve_system_euler"]


def solve_hopf_batch_cuda(initial, alpha, beta, t0, t1, steps):
    '''Solve a batch of Hopf-system IVPs on CUDA.

    initial has shape (batch, 2); result has shape (batch, steps + 1, 2).
    '''
    initial = np.asarray(initial, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 2:
        raise ValueError("initial must have shape (batch, 2)")
    result = _core.solve_hopf_batch_cuda(
        initial.ravel().tolist(), alpha, beta, t0, t1, steps
    )
    return np.asarray(result).reshape(len(initial), steps + 1, 2)


__all__.append("solve_hopf_batch_cuda")


def solve_tridiagonal_cn(left_lower, left_diagonal, left_upper,
                         right_lower, right_diagonal, right_upper,
                         initial, steps, *, backend="cpu"):
    """Advance L*y_next=R*y using constant tridiagonal matrices.

    For M*y'=K*y, construct L=M-dt*K/2 and R=M+dt*K/2. Off-diagonals
    have n-1 entries. L needs a positive, diagonally dominant diagonal.
    Returns (final_state, timings_ms). Setup/solve are native wall times;
    total includes Python conversion, transfers, and native cleanup.
    """
    from time import perf_counter
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")
    solver = _core.solve_tridiagonal_cn if backend == "cpu" else _core.solve_tridiagonal_cn_cuda
    start = perf_counter()
    y, timing = solver(left_lower, left_diagonal, left_upper,
                       right_lower, right_diagonal, right_upper, initial, steps)
    timing["total_ms"] = 1000 * (perf_counter() - start)
    return y, timing


__all__.append("solve_tridiagonal_cn")


def solve_linear_multi_step(f, y0, t0, t1, steps, method=MultiStepMethod.BDF,
                            order=2, newton_atol=1e-12, newton_rtol=1e-10,
                            max_iterations=30, df_dy=None, startup_values=()):
    """Fixed-step scalar BDF1–6; return (t, y) arrays with named fields.

    BDF3–6 need order-1 accurate startup values. Newton tolerances control
    the nonlinear solve, not integration error.
    """
    return Result(*_core.solve_linear_multi_step(
        f, y0, t0, t1, steps, method, order, newton_atol, newton_rtol,
        max_iterations, df_dy, startup_values))


__all__.extend(["MultiStepMethod", "solve_linear_multi_step"])
