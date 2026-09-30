#pragma once
#include <cstddef>
#include <vector>

namespace odelab {
struct LinearSystemResult {
  std::vector<double> y;
  double setup_ms;
  double solve_ms;
};
}