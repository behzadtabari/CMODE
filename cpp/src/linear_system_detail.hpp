#pragma once
#include "odelab/ivp/linear_system.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace odelab::cn_detail {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
inline void validate(const std::vector<double>& ll, const std::vector<double>& ld,
                     const std::vector<double>& lu, const std::vector<double>& rl,
                     const std::vector<double>& rd, const std::vector<double>& ru,
                     const std::vector<double>& initial, std::size_t steps) {
  const auto n = initial.size();
  if (n == 0 || steps == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      ld.size() != n || rd.size() != n || ll.size() != n-1 || lu.size() != n-1 ||
      rl.size() != n-1 || ru.size() != n-1)
    throw std::invalid_argument("require nonempty matching tridiagonals, initial state and steps > 0");
  for (const auto* values : {&ll, &ld, &lu, &rl, &rd, &ru, &initial})
    for (double value : *values)
      if (!std::isfinite(value)) throw std::invalid_argument("all system values must be finite");
  for (std::size_t i = 0; i < n; ++i) {
    const long double off = (i ? std::abs(static_cast<long double>(ll[i-1])) : 0.0L) +
                            (i+1 < n ? std::abs(static_cast<long double>(lu[i])) : 0.0L);
    if (!(ld[i] > 0.0) || off > static_cast<long double>(ld[i]) * (1.0L + 1e-14L))
      throw std::invalid_argument("left matrix must have positive diagonal and be diagonally dominant");
  }
}
}  // namespace odelab::cn_detail
