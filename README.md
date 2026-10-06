# Numerical ODEs and DAEs

A hobby project to implement and visualize numerical methods for ordinary differential equations (ODEs) and differential-algebraic equations (DAEs).

My main reference is *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations* by Uri M. Ascher and Linda R. Petzold. I’m using it to guide my learning while writing my own code, explanations and solve each chapter's exercises using my own solver and conventional Python solvers like NumPy

---

## What’s inside

* Experiments with explicit and implicit methods
* Visual comparisons of accuracy, stability, and step size
* Examples involving initial value problems, boundary value problems, and DAEs as the project grows

This repository is still under development.

For a fixed-step implicit ODE example, compare the C++ BDF solver with SciPy:

```sh
python -m pip install '.[examples]'
python examples/chapter3_exercises/bdf_nonlinear_benchmark.py
```

The example solves `y'=-5*t*y²+5/t-1/t²` on `[1,2]` with `y(1)=1`.
The exact solution `y=1/t` provides an accuracy check. Results and a plot
are saved under `build/bdf_nonlinear/`.


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
