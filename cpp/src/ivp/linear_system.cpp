#include "odelab/ivp/linear_system_detail.hpp"

namespace odelab {
LinearSystemResult solve_tridiagonal_cn(
    const std::vector<double>& ll, const std::vector<double>& ld,
    const std::vector<double>& lu, const std::vector<double>& rl,
    const std::vector<double>& rd, const std::vector<double>& ru,
    const std::vector<double>& initial, std::size_t steps) {
  using namespace cn_detail;
  const auto start = Clock::now();
  validate(ll, ld, lu, rl, rd, ru, initial, steps);
  const auto n = initial.size();
  auto multipliers = ll;
  auto pivots = ld;
  auto y = initial;
  std::vector<double> rhs(n);
  for (std::size_t i = 1; i < n; ++i) {
    multipliers[i-1] /= pivots[i-1];
    pivots[i] -= multipliers[i-1] * lu[i-1];
    if (!std::isfinite(pivots[i]) || pivots[i] <= 0)
      throw std::runtime_error("invalid tridiagonal pivot; matrix may be singular or ill-conditioned");
  }
  const auto ready = Clock::now();
  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t i = 0; i < n; ++i) {
      rhs[i] = rd[i] * y[i];
      if (i) rhs[i] += rl[i-1] * y[i-1];
      if (i+1 < n) rhs[i] += ru[i] * y[i+1];
    }
    for (std::size_t i = 1; i < n; ++i) rhs[i] -= multipliers[i-1] * rhs[i-1];
    y[n-1] = rhs[n-1] / pivots[n-1];
    for (std::size_t i = n-1; i > 0; --i)
      y[i-1] = (rhs[i-1] - lu[i-1] * y[i]) / pivots[i-1];
  }
  const auto done = Clock::now();
  for (double value : y)
    if (!std::isfinite(value)) throw std::runtime_error("non-finite linear system solution");
  return {std::move(y), milliseconds(start, ready), milliseconds(ready, done)};
}
}  // namespace odelab
