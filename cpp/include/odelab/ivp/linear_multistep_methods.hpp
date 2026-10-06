#pragma once

#include <cstddef>
#include <vector>
#include "odelab/ivp/first_order_solver.hpp"

namespace odelab {

enum class MultiStepMethod { bdf };

// Scalar IVP y' = f(t,y), on a uniform grid with h = (t1-t0)/steps.
// Returns steps+1 time points and solution values, including (t0,y0).
// Supports fixed BDF orders 1..6; requires steps >= order.
//
// newton_atol/newton_rtol control Newton convergence only.
// There is no adaptive time stepping or time integration error estimator.
// df_dy is optional: an empty callback selects a central finite difference.
//
// startup_values, if supplied, contains y_1,...,y_{order-1} at the
// solver's uniform grid times. Its size must be exactly order-1.
// - BDF1 needs no startup values.
// - BDF2 uses one backward Euler step when startup_values is empty.
// - BDF3..BDF6 require supplied startup values with O(h^order) accuracy
//   or better, to preserve the requested convergence order.
//
// Invalid inputs or Newton failures throw exceptions.
Result solve_linear_multi_step(
    const Rhs& f,
    double y0,
    double t0,
    double t1,
    std::size_t steps,
    MultiStepMethod method = MultiStepMethod::bdf,
    std::size_t order = 2,
    double newton_atol = 1e-12,
    double newton_rtol = 1e-10,
    int max_iterations = 30,
    const Rhs& df_dy = Rhs{},
    const std::vector<double>& startup_values = {}
);

}  // namespace odelab
