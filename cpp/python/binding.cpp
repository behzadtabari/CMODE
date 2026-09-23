#include "../include/odelab/ivp/first_order_solver.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
py::array_t<double> to_array(const std::vector<double>& values) {
  py::array_t<double> array(values.size());
  std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(double));
  return array;
}

py::tuple to_python(const odelab::Result& result) {
  return py::make_tuple(to_array(result.t), to_array(result.y));
}

py::array_t<double> to_matrix(const std::vector<double>& data,
                              std::size_t rows, std::size_t cols) {
  py::array_t<double> out({static_cast<py::ssize_t>(rows),
                           static_cast<py::ssize_t>(cols)});
  if (!data.empty()) std::memcpy(out.mutable_data(), data.data(), data.size() * sizeof(double));
  return out;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Compiled scalar ODE solvers";
  m.def("solve_system_euler", [](const odelab::SystemRhs& f, const std::vector<double>& y0,
                                 double t0, double t1, std::size_t steps) {
    const auto result = odelab::solve_system_euler(f, y0, t0, t1, steps);
    return py::make_tuple(result.t, result.y);
  }, py::arg("f"), py::arg("y0"), py::arg("t0"), py::arg("t1"), py::arg("steps"));
  py::enum_<odelab::Method>(m, "Method")
      .value("forward_euler", odelab::Method::forward_euler)
      .value("backward_euler", odelab::Method::backward_euler)
      .value("trapezoidal", odelab::Method::trapezoidal)
      .value("FORWARD_EULER", odelab::Method::forward_euler)
      .value("BACKWARD_EULER", odelab::Method::backward_euler)
      .value("TRAPEZOIDAL", odelab::Method::trapezoidal);

  m.def("solve", [](py::function f, double y0, double t0, double t1,
                    std::size_t steps, odelab::Method method,
                    double tolerance, int max_iterations) {
    // The callback is Python, so keep the GIL. For benchmarks use solve_linear.
    return to_python(odelab::solve([&f](double t, double y) {
      return f(t, y).cast<double>();
    }, y0, t0, t1, steps, method, tolerance, max_iterations));
  }, py::arg("f"), py::arg("y0"), py::arg("t0"), py::arg("t1"),
     py::arg("steps"), py::arg("method"), py::arg("tolerance") = 1e-11,
     py::arg("max_iterations") = 30);

  m.def("solve_linear", [](double a, double b, double y0, double t0,
                           double t1, std::size_t steps, odelab::Method method) {
    odelab::Result result;
    { py::gil_scoped_release release;
      result = odelab::solve_linear(a, b, y0, t0, t1, steps, method); }
    return to_python(result);
  }, py::arg("a"), py::arg("b"), py::arg("y0"), py::arg("t0"),
     py::arg("t1"), py::arg("steps"), py::arg("method"));

  m.def("solve_linear_batch", [](const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& y0,
                                 double t0, double t1, std::size_t steps,
                                 odelab::Method method) {
    std::vector<double> data;
    { py::gil_scoped_release release;
      data = odelab::solve_linear_batch(a, b, y0, t0, t1, steps, method); }
    return to_matrix(data, a.size(), steps + 1);
  }, py::arg("a"), py::arg("b"), py::arg("y0"), py::arg("t0"),
     py::arg("t1"), py::arg("steps"), py::arg("method"));

#ifdef ODELAB_HAS_CUDA
  m.def("cuda_available", &odelab::cuda_available);
  m.def("solve_linear_batch_cuda", [](const std::vector<double>& a,
                                      const std::vector<double>& b,
                                      const std::vector<double>& y0,
                                      double t0, double t1, std::size_t steps,
                                      odelab::Method method) {
    std::vector<double> data;
    { py::gil_scoped_release release;
      data = odelab::solve_linear_batch_cuda(a, b, y0, t0, t1, steps, method); }
    return to_matrix(data, a.size(), steps + 1);
  }, py::arg("a"), py::arg("b"), py::arg("y0"), py::arg("t0"),
     py::arg("t1"), py::arg("steps"), py::arg("method"));
#else
  m.def("cuda_available", [] { return false; });
  m.def("solve_linear_batch_cuda", [](py::args, py::kwargs) {
    throw std::runtime_error("CUDA backend was not built; set ODELAB_ENABLE_CUDA=ON");
  });
#endif
}
