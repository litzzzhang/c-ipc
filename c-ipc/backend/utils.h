#pragma once

#include "c-ipc/solver/eigen.h"
#include "c-ipc/backend/stl_port.h"

namespace cipc {

template <typename TestModel, typename RandomEngine>
bool gradient_checker(TestModel model, const Matrix3Xr &x, RandomEngine &rng) {
    integer vertex_num = static_cast<integer>(x.cols());
    integer dim = 3 * vertex_num;
    // current energy and gradient
    real energy = model.energy(x);
    Matrix3Xr gradient = model.gradient(x);
    // generate a random direction
    VectorXr test_dir = VectorXr::Zero(dim);
    static std::uniform_real_distribution<real> unit_random(0.0, 1.0);
    oneapi::tbb::parallel_for(0, dim, [&](integer i) {
        test_dir(i) = unit_random(rng);
    });

    while (test_dir.norm() < 1e-6) {
        oneapi::tbb::parallel_for(0, dim, [&](integer i) {
            test_dir(i) = unit_random(rng);
        });
    }

    for (integer i = 4; i < 13; i++) {
        real eps = std::pow(10, (real)(-i));

        Matrix3Xr x_plus = x + eps * test_dir.reshaped(3, vertex_num);
        real energy_plus = model.energy(x_plus);
        real numeric_diff_plus = (energy_plus - energy) / eps;
        real analytic_diff_plus = test_dir.dot(gradient.reshaped());
        real absolute_error_plus = std::abs(analytic_diff_plus - numeric_diff_plus);

        // printf("analytic energy: %.9f, numeric energy: %.9f ", energy, energy_plus);
        // printf("absolute_error: %.9f\n", absolute_error_plus);
        if (absolute_error_plus < 1e-3) { return true; }

        Matrix3Xr x_minus = x - eps * test_dir.reshaped(3, vertex_num);
        real energy_minus = model.energy(x_minus);
        real numeric_diff_minus = (energy - energy_minus) / eps;
        real analytic_diff_minus = test_dir.dot(gradient.reshaped());
        real absolute_error_minus = std::abs(analytic_diff_minus - numeric_diff_minus);

        // printf("[plus]: abs_err: %f, rel_err: %f\n", absolute_error_plus, relative_error_plus);
        if (absolute_error_minus < 1e-3) { return true; }
    }
    return false;
};

template <typename TestModel, typename RandomEngine>
bool hessian_checker(TestModel model, const Matrix3Xr &x, RandomEngine &rng) {
    integer vertex_num = static_cast<integer>(x.cols());
    integer dim = 3 * vertex_num;
    // current hessian and gradient
    Matrix3Xr grad = model.gradient(x);
    SparseMatrixXr hessian = model.hessian(x);
    // generate a random direction
    VectorXr test_dir = VectorXr::Zero(dim);
    static std::uniform_real_distribution<real> unit_random(0.0, 1.0);
    oneapi::tbb::parallel_for(0, dim, [&](integer i) {
        test_dir(i) = unit_random(rng);
    });

    while (test_dir.norm() < 1e-6) {
        oneapi::tbb::parallel_for(0, dim, [&](integer i) {
            test_dir(i) = unit_random(rng);
        });
    }
        printf("grad norm %.9f\n", grad.reshaped().norm());
    for (integer i = 4; i < 13; i++) {
        real eps = std::pow(10, (real)(-i));

        Matrix3Xr x_plus = x + eps * test_dir.reshaped(3, vertex_num);
        Matrix3Xr grad_plus = model.gradient(x_plus);

        VectorXr numeric_diff_plus = (grad_plus - grad).reshaped() / eps;
        VectorXr analytic_diff_plus = hessian * test_dir;
        real absolute_error_plus = (analytic_diff_plus - numeric_diff_plus).norm();

        printf("analytic: %.9f, numeric: %.9f ", analytic_diff_plus.norm(), numeric_diff_plus.norm());
        printf("absolute_error: %.9f\n", absolute_error_plus);
        if (absolute_error_plus < 1e-2) { return true; }

        Matrix3Xr x_minus = x - eps * test_dir.reshaped(3, vertex_num);
        Matrix3Xr grad_minus = model.gradient(x_minus);
        VectorXr numeric_diff_minus = (grad - grad_minus).reshaped() / eps;
        VectorXr analytic_diff_minus = hessian * test_dir;
        real absolute_error_minus = (analytic_diff_minus - numeric_diff_minus).norm();

        // // printf("[plus]: abs_err: %f, rel_err: %f\n", absolute_error_plus,relative_error_plus); 
        if (absolute_error_minus < 1e-2) { return true; }
    }
    return false;
};
} // namespace cipc
