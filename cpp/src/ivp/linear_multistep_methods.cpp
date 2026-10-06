#include "odelab/ivp/linear_multistep_methods.hpp"
#include "odelab/ivp/newton_methods.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace odelab {
namespace {

// Constant-step BDF coefficients, ordered newest to oldest:
// alpha[0]*y_n + sum(j=1..k, alpha[j]*y_{n-j}) = h*f(t_n,y_n).
// Row k-1 contains BDF(k). The RHS coefficient is always 1.
constexpr std::array<std::array<double, 7>, 6> bdf_coefficients{{
    {{1.0,        -1.0, 0.0,        0.0,       0.0,      0.0,      0.0}},
    {{3.0/2.0,    -2.0, 1.0/2.0,    0.0,       0.0,      0.0,      0.0}},
    {{11.0/6.0,   -3.0, 3.0/2.0,   -1.0/3.0,   0.0,      0.0,      0.0}},
    {{25.0/12.0,  -4.0, 3.0,       -4.0/3.0,   1.0/4.0,  0.0,      0.0}},
    {{137.0/60.0, -5.0, 5.0,      -10.0/3.0,   5.0/4.0, -1.0/5.0,  0.0}},
    {{49.0/20.0,  -6.0, 15.0/2.0, -20.0/3.0,  15.0/4.0,-6.0/5.0,  1.0/6.0}}
}};

// Approximate partial f / partial y, keeping t fixed.
// This perturbation is unrelated to the time step h or Newton tolerances.
double numerical_df_dy(const Rhs& f, double t, double y)
{
    const double perturbation =
        std::cbrt(std::numeric_limits<double>::epsilon())
        * std::max(1.0, std::abs(y));
    const double upper = y + perturbation;
    const double lower = y - perturbation;
    return (f(t, upper) - f(t, lower)) / (upper - lower);
}

}  // namespace

Result solve_linear_multi_step(
    const Rhs& f, double y0, double t0, double t1,
    std::size_t steps, MultiStepMethod method, std::size_t order,
    double newton_atol, double newton_rtol, int max_iterations,
    const Rhs& df_dy, const std::vector<double>& startup_values)
{
    if (method != MultiStepMethod::bdf)
        throw std::invalid_argument("Only BDF is implemented");
    if (!f || !std::isfinite(y0) || !std::isfinite(t0) ||
        !std::isfinite(t1) || t0 == t1)
        throw std::invalid_argument("Invalid IVP inputs");
    if (order < 1 || order > 6)
        throw std::invalid_argument("BDF order must be in [1,6]");
    if (steps < order || steps == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("steps must be at least the BDF order");
    if (!std::isfinite(newton_atol) || !std::isfinite(newton_rtol) ||
        newton_atol <= 0.0 || newton_rtol < 0.0 || max_iterations <= 0)
        throw std::invalid_argument("Invalid Newton options");
    if (!startup_values.empty() && startup_values.size() != order - 1)
        throw std::invalid_argument("Supply exactly order-1 startup values");
    if (order > 2 && startup_values.empty())
        throw std::invalid_argument("BDF3..BDF6 require accurate startup values");
    for (double value : startup_values) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Startup values must be finite");
    }

    const double h = (t1 - t0) / static_cast<double>(steps);
    if (!std::isfinite(h) || h == 0.0)
        throw std::invalid_argument("Invalid time step");

    Result result;
    result.t.resize(steps + 1);
    result.y.resize(steps + 1);
    result.t[0] = t0;
    result.y[0] = y0;
    for (std::size_t n = 1; n <= steps; ++n) {
        result.t[n] = (n == steps) ? t1 : t0 + static_cast<double>(n)*h;
        if (!std::isfinite(result.t[n]) ||
            (h > 0.0 ? result.t[n] <= result.t[n-1]
                     : result.t[n] >= result.t[n-1]))
            throw std::invalid_argument("Time grid is not representable");
    }

    // With supplied history, the first new unknown is y_order.
    // Otherwise, start at y_1 (backward Euler when order==2).
    std::size_t first_step = 1;
    if (!startup_values.empty()) {
        for (std::size_t j = 0; j < startup_values.size(); ++j)
            result.y[j + 1] = startup_values[j];
        first_step = order;
    }

    for (std::size_t n = first_step; n <= steps; ++n) {
        const std::size_t active_order = (n == 1) ? 1 : order;
        const auto& alpha = bdf_coefficients[active_order - 1];
        const double t_n = result.t[n];
        double past = 0.0;
        for (std::size_t j = 1; j <= active_order; ++j)
            past += alpha[j] * result.y[n-j];
        const auto residual = [&](double value) {
            return alpha[0]*value + past - h*f(t_n, value);
        };
        const auto derivative = [&](double value) {
            return alpha[0] - h*(df_dy ? df_dy(t_n, value)
                                         : numerical_df_dy(f, t_n, value));
        };
        double guess = result.y[n-1];
        if (n > 1)
            guess += result.y[n-1] - result.y[n-2];
        else
            guess += h*f(result.t[0], result.y[0]);
        result.y[n] = newton_scalar(residual, derivative, guess,
                                   newton_atol, newton_rtol, max_iterations);
    }
    return result;
}

}  // namespace odelab
