#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace odelab {

enum class Method { forward_euler, backward_euler, trapezoidal };

struct Result {
  std::vector<double> t;
  std::vector<double> y;
};

using Rhs = std::function<double(double, double)>;

struct SystemResult {
  std::vector<double> t;
  std::vector<std::vector<double>> y;
};

using SystemRhs = std::function<std::vector<double>(double, const std::vector<double>&)>;

// Forward Euler for a coupled system; all components use the same old state.
SystemResult solve_system_euler(const SystemRhs& f, const std::vector<double>& y0,
                                double t0, double t1, std::size_t steps);

// Scalar IVP y' = f(t,y). Implicit steps use Newton iteration and a numerical
// derivative. Exceptions signal invalid input or failure to converge.
Result solve(const Rhs& f, double y0, double t0, double t1,
             std::size_t steps, Method method, double tolerance = 1e-11,
             int max_iterations = 30);

// Fast path y' = a*y + b, entirely in compiled code.
Result solve_linear(double a, double b, double y0, double t0, double t1,
                    std::size_t steps, Method method);

// Row-major output: trajectory i is y[i*(steps+1) : (i+1)*(steps+1)].
std::vector<double> solve_linear_batch(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& y0, double t0, double t1, std::size_t steps,
    Method method);

#ifdef ODELAB_HAS_CUDA
std::vector<double> solve_linear_batch_cuda(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& y0, double t0, double t1, std::size_t steps,
    Method method);
bool cuda_available();
#endif

}  // namespace odelab
