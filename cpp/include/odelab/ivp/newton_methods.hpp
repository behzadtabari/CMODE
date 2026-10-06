#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace odelab {

// Solve g(z)=0. These tolerances control Newton corrections only;
// they do not estimate time discretization error or select a time step.
inline double newton_scalar(
    const std::function<double(double)>& g,
    const std::function<double(double)>& gprime,
    double z,
    double newton_atol = 1e-12,
    double newton_rtol = 1e-10,
    int max_iterations = 30)
{
    if (!g || !gprime || !std::isfinite(z) ||
        !std::isfinite(newton_atol) || !std::isfinite(newton_rtol) ||
        newton_atol <= 0.0 || newton_rtol < 0.0 || max_iterations <= 0)
        throw std::invalid_argument("Invalid Newton inputs");

    for (int rho = 0; rho < max_iterations; ++rho) {
        const double residual = g(z);
        if (!std::isfinite(residual))
            throw std::runtime_error("Nonfinite Newton residual");
        if (residual == 0.0)
            return z;

        const double derivative = gprime(z);
        if (!std::isfinite(derivative) || derivative == 0.0)
            throw std::runtime_error("Invalid Newton derivative");

        const double delta = -residual / derivative;
        const double next = z + delta;
        if (!std::isfinite(delta) || !std::isfinite(next))
            throw std::runtime_error("Nonfinite Newton update");

        const double scale = std::max(std::abs(z), std::abs(next));
        const double threshold = newton_atol + newton_rtol * scale;
        if (!std::isfinite(threshold))
            throw std::runtime_error("Nonfinite Newton tolerance threshold");
        if (std::abs(delta) <= threshold)
            return next;

        z = next;
    }

    throw std::runtime_error("Newton did not converge");
}

}  // namespace odelab
