#include "../include/odelab/ivp/first_order_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace odelab {
namespace {

void check_inputs(double y0, double t0, double t1, std::size_t steps) {
  if (!std::isfinite(y0) || !std::isfinite(t0) || !std::isfinite(t1) ||
      !(t1 > t0) || steps == 0 || steps == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("require finite y0, finite t0 < t1, and steps > 0");
}

double finite(double x) {
  if (!std::isfinite(x)) throw std::runtime_error("non-finite solution or RHS");
  return x;
}

double newton(const std::function<double(double)>& residual, double guess,
              double tolerance, int max_iterations) {
  double x = finite(guess);
  for (int k = 0; k < max_iterations; ++k) {
    const double r = finite(residual(x));
    if (std::abs(r) <= tolerance * (1.0 + std::abs(x))) return x;
    const double delta = std::sqrt(std::numeric_limits<double>::epsilon()) *
                         (1.0 + std::abs(x));
    const double derivative = finite((residual(x + delta) - residual(x - delta)) /
                                     (2.0 * delta));
    if (std::abs(derivative) <= std::numeric_limits<double>::epsilon())
      throw std::runtime_error("Newton derivative is zero");
    x = finite(x - r / derivative);
  }
  throw std::runtime_error("Newton iteration did not converge");
}

double linear_step(double y, double a, double b, double h, Method method) {
  if (method == Method::forward_euler) return finite(y + h * (a * y + b));
  const double weight = method == Method::backward_euler ? 1.0 : 0.5;
  const double denom = 1.0 - weight * h * a;
  if (!std::isfinite(denom) || std::abs(denom) < 1e-14)
    throw std::runtime_error("implicit step has singular denominator");
  return finite(((1.0 + (1.0 - weight) * h * a) * y + h * b) / denom);
}

}  // namespace

SystemResult solve_system_euler(const SystemRhs& f, const std::vector<double>& y0,
                                double t0, double t1, std::size_t steps) {
  check_inputs(0.0, t0, t1, steps);
  if (!f || y0.empty()) throw std::invalid_argument("require RHS and nonempty state");
  for (double value : y0)
    if (!std::isfinite(value)) throw std::invalid_argument("initial state must be finite");
  const double h = (t1 - t0) / static_cast<double>(steps);
  if (!std::isfinite(h) || h <= 0.0)
    throw std::invalid_argument("step size must be positive and finite");
  SystemResult result;
  result.t.resize(steps + 1);
  result.y.resize(steps + 1);
  result.t[0] = t0;
  result.y[0] = y0;
  for (std::size_t i = 0; i < steps; ++i) {
    const auto derivative = f(result.t[i], result.y[i]);
    if (derivative.size() != y0.size())
      throw std::invalid_argument("RHS dimension must match initial state");
    result.y[i + 1].resize(y0.size());
    for (std::size_t j = 0; j < y0.size(); ++j)
      result.y[i + 1][j] = finite(result.y[i][j] + h * finite(derivative[j]));
    result.t[i + 1] = i + 1 == steps ? t1 : t0 + static_cast<double>(i + 1) * h;
  }
  return result;
}

Result solve(const Rhs& f, double y0, double t0, double t1,
             std::size_t steps, Method method, double tolerance, int max_iterations) {
  check_inputs(y0, t0, t1, steps);
  if (!f || !(tolerance > 0.0) || !std::isfinite(tolerance) || max_iterations < 1)
    throw std::invalid_argument("invalid RHS or Newton options");
  const double h = (t1 - t0) / static_cast<double>(steps);
  Result result;
  result.t.resize(steps + 1);
  result.y.resize(steps + 1);
  result.t[0] = t0;
  result.y[0] = y0;
  for (std::size_t i = 0; i < steps; ++i) {
    const double ti = t0 + static_cast<double>(i) * h;
    const double next_t = i + 1 == steps ? t1 : t0 + static_cast<double>(i + 1) * h;
    const double y = result.y[i];
    const double f0 = finite(f(ti, y));
    double next;
    if (method == Method::forward_euler) {
      next = finite(y + h * f0);
    } else if (method == Method::backward_euler) {
      next = newton([&](double x) { return x - y - h * finite(f(next_t, x)); },
                    y + h * f0, tolerance, max_iterations);
    } else {
      next = newton([&](double x) {
        return x - y - (h / 2.0) * (f0 + finite(f(next_t, x)));
      }, y + h * f0, tolerance, max_iterations);
    }
    result.t[i + 1] = next_t;
    result.y[i + 1] = next;
  }
  return result;
}

Result solve_linear(double a, double b, double y0, double t0, double t1,
                    std::size_t steps, Method method) {
  check_inputs(y0, t0, t1, steps);
  if (!std::isfinite(a) || !std::isfinite(b))
    throw std::invalid_argument("a and b must be finite");
  const double h = (t1 - t0) / static_cast<double>(steps);
  Result result;
  result.t.resize(steps + 1);
  result.y.resize(steps + 1);
  result.t[0] = t0;
  result.y[0] = y0;
  for (std::size_t i = 0; i < steps; ++i) {
    result.t[i + 1] = i + 1 == steps ? t1 : t0 + static_cast<double>(i + 1) * h;
    result.y[i + 1] = linear_step(result.y[i], a, b, h, method);
  }
  return result;
}

std::vector<double> solve_linear_batch(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& y0, double t0, double t1, std::size_t steps,
    Method method) {
  if (a.size() != b.size() || a.size() != y0.size())
    throw std::invalid_argument("a, b, and y0 must have equal lengths");
  check_inputs(0.0, t0, t1, steps);
  if (a.size() > std::numeric_limits<std::size_t>::max() / (steps + 1))
    throw std::invalid_argument("output dimensions overflow");
  const double h = (t1 - t0) / static_cast<double>(steps);
  std::vector<double> out(a.size() * (steps + 1));
  for (std::size_t j = 0; j < a.size(); ++j) {
    if (!std::isfinite(a[j]) || !std::isfinite(b[j]))
      throw std::invalid_argument("a and b must be finite");
    if (!std::isfinite(y0[j])) throw std::invalid_argument("y0 must be finite");
    const std::size_t offset = j * (steps + 1);
    out[offset] = y0[j];
    for (std::size_t i = 0; i < steps; ++i)
      out[offset + i + 1] = linear_step(out[offset + i], a[j], b[j], h, method);
  }
  return out;
}

}  // namespace odelab
