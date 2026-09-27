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
python examples/hopfbi_gpu_cpu_benchmarking.py --require-cuda
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
python examples/hopfbi_gpu_cpu_benchmarking.py --counts 1 4 --steps 100 --repeats 1
```

The benchmark checks numerical agreement. GPU timing includes allocations,
transfers, and completed execution after warm-up. The C++ CPU solver calls a
Python RHS, while CUDA evaluates the RHS entirely on the GPU; timings compare
these complete implementations. The notebook uses fresh Python subprocesses
so rerunning a build does not reuse an already-loaded extension.

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
python examples/hopf_bifurcation.py
# Save a figure without opening a window:
python examples/hopf_bifurcation.py --no-show --save build/hopf.png
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
