#pragma once
#include <cstddef>
#include <vector>

namespace odelab {
struct LinearSystemResult {
  std::vector<double> y;
  double setup_ms;
  double solve_ms;
};

// Integrate L*y_next = R*y for a constant tridiagonal linear ODE system.
// Crank-Nicolson uses L=M-dt*K/2, R=M+dt*K/2 for M*y'=K*y.
// Off-diagonals have n-1 entries. L must have positive diagonal and be
// diagonally dominant; no pivoting is used. Returns the final state.
LinearSystemResult solve_tridiagonal_cn(
    const std::vector<double>& ll, const std::vector<double>& ld,
    const std::vector<double>& lu, const std::vector<double>& rl,
    const std::vector<double>& rd, const std::vector<double>& ru,
    const std::vector<double>& initial, std::size_t steps);

#ifdef ODELAB_HAS_CUDA
LinearSystemResult solve_tridiagonal_cn_cuda(
    const std::vector<double>& ll, const std::vector<double>& ld,
    const std::vector<double>& lu, const std::vector<double>& rl,
    const std::vector<double>& rd, const std::vector<double>& ru,
    const std::vector<double>& initial, std::size_t steps);
#endif
}  // namespace odelab
