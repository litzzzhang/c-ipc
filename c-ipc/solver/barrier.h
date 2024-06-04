#pragma once

#include <c-ipc/backend/stl_port.h>

namespace cipc {
static double barrier(const double d, const double dhat) {
    if (d <= 0.0) { return std::numeric_limits<double>::infinity(); }
    if (d >= dhat) { return 0.0; }
    const double d_minus_dhat = d - dhat;
    return -d_minus_dhat * d_minus_dhat * std::log(d / dhat);
}

static double barrier_first_derivative(const double d, const double dhat) {
    if (d <= 0.0 || d >= dhat) { return 0.0; }
    return (dhat - d) * (2 * std::log(d / dhat) - dhat / d + 1);
}

static double barrier_second_derivative(const double d, const double dhat) {
    if (d <= 0.0 || d >= dhat) { return 0.0; }
    const double dhat_over_d = dhat / d;
    return (dhat_over_d + 2) * dhat_over_d - 2 * std::log(d / dhat) - 3;
}

static double cipc_barrier(const double dist_squared, const double dhat, const double dmin) {
    return barrier(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

static double
cipc_barrier_first_derivative(const double dist_squared, const double dhat, const double dmin) {
    return barrier_first_derivative(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

static double cipc_barrier_second_derivative(const double dist_squared, const double dhat, const double dmin){
    return barrier_second_derivative(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}
} // namespace cipc
