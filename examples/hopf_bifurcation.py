"""Compare C++ and NumPy forward Euler near a Hopf bifurcation.

Run from the repository root:
    .venv/bin/python examples/hopf_bifurcation.py
    .venv/bin/python examples/hopf_bifurcation.py --no-show --save build/hopf.png

NumPy supplies array operations; forward Euler is implemented below.
The C++ integrator uses the same Python RHS callback.
"""

import argparse
from pathlib import Path

import numpy as np

from odelab import solve_system_euler


def f(t, y, alpha, beta):
    y1, y2 = y
    denominator = 1.0 + y1**2
    return np.array([
        alpha - y1 - 4.0 * y1 * y2 / denominator,
        beta * y1 * (1.0 - y2 / denominator),
    ])


def forward_euler(f, t0, y0, h, n_steps, alpha, beta):
    t = t0 + h * np.arange(n_steps + 1)
    y = np.empty((n_steps + 1, len(y0)), dtype=float)
    y[0] = y0
    for n in range(n_steps):
        y[n + 1] = y[n] + h * f(t[n], y[n], alpha, beta)
    return t, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Save the comparison figure")
    parser.add_argument("--no-show", action="store_true", help="Run without a GUI")
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha, beta = 10.0, 3.6
    t0, y0, h, n_steps = 0.0, [0.0, 2.0], 0.01, 2000
    t_numpy, y_numpy = forward_euler(f, t0, y0, h, n_steps, alpha, beta)
    t_cpp, y_cpp = solve_system_euler(
        lambda t, y: f(t, y, alpha, beta), y0, t0, t0 + h * n_steps, n_steps
    )
    np.testing.assert_allclose(t_cpp, t_numpy, rtol=0, atol=1e-13)
    np.testing.assert_allclose(y_cpp, y_numpy, rtol=1e-10, atol=1e-11)
    x = alpha / 5.0
    equilibrium = np.array([x, 1.0 + x**2])
    beta_hopf = (3.0 * x**2 - 5.0) / x
    print(f"Final time: {t_cpp[-1]:.2f}")
    print(f"C++ final state [y1, y2]: {y_cpp[-1]}")
    print(f"NumPy final state [y1, y2]: {y_numpy[-1]}")
    print(f"Maximum absolute difference: {np.max(np.abs(y_cpp - y_numpy)):.3e}")
    print(f"Equilibrium: {equilibrium}; Hopf threshold beta = {beta_hopf:g}")
    print("beta = 3.6 is on the locally stable side of the threshold.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for j, color in enumerate(("tab:blue", "tab:orange")):
        axes[0].plot(t_cpp, y_cpp[:, j], color=color, label=fr"C++ $y_{j + 1}$")
        axes[0].plot(t_numpy[::50], y_numpy[::50, j], "o", color=color,
                     fillstyle="none", markersize=4, label=fr"NumPy $y_{j + 1}$")
    axes[0].set(xlabel="Time t", ylabel="Solution", title="Forward Euler: C++ and NumPy")
    axes[1].plot(y_cpp[:, 0], y_cpp[:, 1], label="C++")
    axes[1].plot(y_numpy[::50, 0], y_numpy[::50, 1], "o", fillstyle="none",
                 markersize=4, label="NumPy")
    axes[1].plot(*equilibrium, "k*", markersize=12, label="Equilibrium")
    axes[1].set(xlabel=r"$y_1$", ylabel=r"$y_2$", title="Phase portrait")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(r"Near Hopf: $\alpha=10$, $\beta=3.6$ ($\beta_H=3.5$); $h=0.01$")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"Saved figure: {args.save}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
