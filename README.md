# Numerical ODEs and DAEs

A hobby project to implement and visualize numerical methods for ordinary differential equations (ODEs) and differential-algebraic equations (DAEs).

My main reference is *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations* by Uri M. Ascher and Linda R. Petzold. I’m using it to guide my learning while writing my own code, explanations and solve each chapter's exercises in different programming languages.

---

## What’s inside

* Experiments with explicit and implicit methods
* Visual comparisons of accuracy, stability, and step size
* Examples involving initial value problems, boundary value problems, and DAEs as the project grows

This repository is still under development.

## Run in Colab

Open [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb), select
a GPU runtime, and run the cells in order. First publish the updated
`colab_version` branch to GitHub so Colab can clone these files.
The original local `Colab_version.ipynb` is preserved as a reference; its CUDA
kernel and build fixes now live in the package instead of notebook patches.

The essential commands, from the repository root in Colab, are:

```sh
python -m pip install '.[test,examples]' \
  -Ccmake.define.ODELAB_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
python examples/chapter3_exercises/hopfbi_gpu_cpu_benchmarking.py --require-cuda
```

The build uses CMake 3.24 or newer to detect the runtime GPU architecture,
so no T4-specific architecture number is needed. CUDA device symbols are
resolved in the static CUDA library before linking the Python extension.
See CMake's [architecture](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
and [device linking](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html) documentation.
CUDA remains optional for local CPU builds. The Hopf CUDA API accepts an
initial array of shape `(batch, 2)` and returns `(batch, steps + 1, 2)`.

For a quick CPU-only check, install `.[test,examples]` without the CUDA settings
and run:

```sh
python examples/chapter3_exercises/hopfbi_gpu_cpu_benchmarking.py --counts 1 4 --steps 100 --repeats 1
```

The benchmark checks numerical agreement. GPU timing includes allocations,
transfers, and completed execution after warm-up. The C++ CPU solver calls a
Python RHS, while CUDA evaluates the RHS entirely on the GPU; timings compare
these complete implementations. The notebook uses fresh Python subprocesses
so rerunning a build does not reuse an already-loaded extension.

The quickstart streams build logs and runtime tracebacks directly into the
notebook. Its benchmark cell displays a table and saves CSV/JSON results for
each run; the next cell plots CPU/GPU timings and reference errors. Each backend
has its own median/minimum time in milliseconds, speedup relative to C++, maximum
absolute error and RMSE against a numerical reference, and difference from C++.
The reference uses [SciPy DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
with `rtol=1e-11`, `atol=1e-13`, evaluated at the Euler time samples. These are
reference errors, not exact-solution errors. Reference generation and warm-up are
outside the timed calls. To save the same table from a terminal:

```sh
python examples/chapter3_exercises/hopfbi_gpu_cpu_benchmarking.py --require-cuda \
  --json build/benchmark.json --csv build/benchmark.csv
```

## Use the C++ solvers from Python

The CPU extension exposes forward Euler, backward Euler, and trapezoidal
methods through pybind11. With Python, a C++17 compiler, and CMake installed,
install the package into a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Alternatively, if pybind11 is already installed, build directly into the source
package using the same Python interpreter that will load the extension:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_INSTALL_PREFIX="$PWD/src"
cmake --build build -j 2
cmake --install build
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from odelab import Method, solve_linear

# y' = -y, y(0) = 1
result = solve_linear(a=-1, b=0, y0=1, t0=0, t1=1,
                      steps=100, method=Method.trapezoidal)
print(result.t[-1], result.y[-1])
```

`solve` accepts a Python callback `f(t, y)`. `solve_linear` executes entirely
in C++ and `solve_linear_batch` returns a NumPy array with one trajectory per
row. Results expose `t` and `y` as Python lists and support `t, y = result`.

Run the Python integration tests after installation or with `PYTHONPATH` set:

```sh
python3 -m unittest discover -s tests/python -v
```

CUDA is optional and disabled by default. Building it requires a CUDA toolkit
and compatible NVIDIA hardware; the CPU extension does not require CUDA.

## First example: near a Hopf bifurcation

```sh
source .venv/bin/activate
python -m pip install '.[test,examples]'
python examples/chapter3_exercises/hopf_bifurcation.py
# Save a figure without opening a window:
python examples/chapter3_exercises/hopf_bifurcation.py --no-show --save build/hopf.png
```

The example compares the C++ `solve_system_euler` integrator with an explicit
forward Euler loop using NumPy, using the same Python RHS callback in both.
NumPy provides array operations rather than a built-in ODE solver. Both use
`y0 = [0, 2]`, `h = 0.01`, 2000 steps, `alpha = 10`, and `beta = 3.6`.
Agreement checks the implementations against each other, not against an exact
solution; both have forward Euler's discretization error. No speedup is assumed
because the C++ loop still calls Python at every step.

For positive parameters, write `x = alpha / 5`. The equilibrium is
`(x, 1 + x²)`. Its Jacobian has trace `(3x² - 5 - beta*x)/(1 + x²)`
and determinant `5*beta*x/(1 + x²)`. Thus at `alpha = 10` the Hopf threshold
is `beta = 3.5`, and the supplied `beta = 3.6` lies on the locally stable side.
This example shows a trajectory near that threshold, rather than a parameter
sweep establishing the bifurcation. Euler's finite step size also affects the
observed stability near the threshold.

---

## Random-grid method of lines: heat equation

Run the **Random-grid method of lines** cells in `notebooks/colab_quickstart.ipynb`
after repository/package setup, or run locally:

```sh
python examples/chapter3_exercises/method_of_line_1D_cpu_gpu.py --cpu-only
```

The example assumes `u_t = u_xx` on the unit rod (`alpha=1`), zero Dirichlet
conditions, and `u(x,0)=sin(pi*x)`. The exact solution
`exp(-pi²*t)*sin(pi*x)` supplies the accuracy reference. Crank–Nicolson advances
the conservative three-point spatial discretization in its mass-matrix form:
`(M - dt*K/2) u_next = (M + dt*K/2) u`, with control-volume lengths on `M`'s
diagonal. No uniform-grid formula is applied to random spacings.

Every run draws independent uniform interior coordinates, sorts them, and adds
the endpoints 0 and 1. There is no spacing regularization. CPU and GPU use the
same grid and double precision. The saved seed and actual grid allow replay
with `--seed NUMBER`; the default is fresh randomness. Different grid sizes
use independent meshes, so the plot is not a controlled convergence study.
Very small gaps can make the system poorly conditioned, and Crank–Nicolson
does not strongly damp the stiffest modes.

The spatial method of lines produces a coupled **ODE system** `M*u'=K*u`;
Crank–Nicolson is the trapezoidal rule applied to that system. The benchmark
compares four implementations, using identical matrices and double precision:

- NumPy/Python: Thomas factorization and substitution using arrays and Python loops.
- SciPy CPU: sparse LU factorization reused by the Python time loop.
- Your C++ CPU: native Thomas factorization and native time stepping.
- Your C++ CUDA: native parallel cyclic reduction, with coefficients precomputed
  once on the GPU and reused at every step. No CuPy is used.

`odelab.solve_tridiagonal_cn` accepts the three diagonals of each constant
matrix `L` and `R` for `L*y_next=R*y`, the initial vector, and step count.
For this ODE, `L=M-dt*K/2` and `R=M+dt*K/2`. This interface also supports other
constant tridiagonal linear ODE systems. The left matrix must have positive
diagonal and be diagonally dominant; the native algorithms do not pivot.
The result is `(final_state, timing_dict)`, with `backend="cpu"` or `"cuda"`.

Both native backends factor/precompute once per solve. Setup and time stepping
are measured separately. GPU solve timing waits for device completion; total
time includes transfers and Python conversions. Warm-up, common matrix assembly,
error calculation, and plotting are excluded. Speedups use the native C++ CPU
solver as baseline. GPU speedups are not assumed for these small single systems.
The normal CUDA package build in Colab supplies everything required.

```sh
python examples/chapter3_exercises/method_of_line_1D_cpu_gpu.py \
  --require-cuda --nodes 64 256 1024 --steps 200 --final-time 0.1 --repeats 3
```

Results in `build/method_of_line_1D/` include `results.csv`, `results.json`,
`grids_and_solutions.npz`, and `comparison.png`. The table reports final-time
maximum absolute and volume-weighted L2 errors against the exact solution,
CPU/GPU differences, timings, and solve/total speedups. Colab uses a separate
output directory for each run. CUDA is optional locally.

## Motivation

I learn numerical methods best by implementing them and seeing how they behave. This repository is my playground for doing that and sharing what I discover.

---

## Reference and attribution

Ascher, U. M., and Petzold, L. R. (1998). *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611971392

The book is a learning reference for this independent project. The repository is not affiliated with its authors or publisher.

---

## 📬 Contact

Questions or suggestions? Reach me at [behzadtabari.bt@gmail.com](mailto:behzadtabari.bt@gmail.com).

---

## ⚖️ License

My original code and documentation are available under the MIT License (see `LICENSE`). This license does not apply to material from the referenced book.
