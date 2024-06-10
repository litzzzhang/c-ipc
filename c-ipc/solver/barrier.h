#pragma once

#include <c-ipc/backend/stl_port.h>
#include <c-ipc/solver/eigen.h>

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

static double
cipc_barrier_second_derivative(const double dist_squared, const double dhat, const double dmin) {
    return barrier_second_derivative(dist_squared - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

static double init_barrier_stiffness(
    const double dhat, const double dmin, double &stiffness_max, const Matrix3Xr &elastic_grad,
    const Matrix3Xr &barrier_grad, const double m_average) {
    // const double min_barrier_stiffness_scale =  1e12;
    const double min_barrier_stiffness_scale = 0.75 * 1e13;
    double dhat2 = dhat * dhat;
    double dmin2 = dmin * dmin;
    // assume cloth is in a 2 x 2 x 2 box, diagonal is 2\sqrt(3)
    double d0 = 1e-8 * 2 * std::sqrt(3) + dmin;
    double d0_2 = d0 * d0;
    if (d0_2 - dmin2 >= 2 * dmin * dhat + dhat2) { d0_2 = dmin * dhat + 0.5 * dhat2; }

    double stiffness_min = 4 * d0_2 * cipc_barrier_second_derivative(d0_2, dhat, dmin);
    stiffness_min = min_barrier_stiffness_scale * m_average / stiffness_min;
    cipc_assert(std::isfinite(stiffness_min), "stiffness min is not finite");

    stiffness_max = 100 * stiffness_min;
    double kappa = 1.0;
    if (barrier_grad.reshaped().squaredNorm() > 0) {
        kappa = -std::abs(barrier_grad.reshaped().dot(elastic_grad.reshaped()))
                / barrier_grad.reshaped().squaredNorm();
        cipc_assert(std::isfinite(kappa), "kappa is not finite");
    }
    return std::min(stiffness_max, std::max(kappa, stiffness_min));
}

static double update_barrier_stiffness(
    const double prev_min_distance, const double curr_min_distance, const double stiffness_max,
    const double stiffness, const double dmin) {
    double dhat_eps_scale = 1e-8;
    double dhat_eps = dhat_eps_scale * (2 * std::sqrt(3) + dmin);
    // dhat_eps *= dhat_eps;
    // printf("prev dist:%.14f, curr dist:%.14f, dhat eps:%.14f\n", prev_min_distance, curr_min_distance, dhat_eps);
    if (prev_min_distance < dhat_eps && curr_min_distance < dhat_eps
        && curr_min_distance < prev_min_distance) {
        // printf("****************************************************************kappa
        // updated!\n");
        return std::min(stiffness_max, 2 * stiffness);
    }
    return stiffness;
}

} // namespace cipc
