"""Numerical ODE solvers implemented in C++."""

import numpy as np
from collections import namedtuple

from . import _core
from ._core import Method, cuda_available

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
