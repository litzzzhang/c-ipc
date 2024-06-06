#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/backend/stl_port.h>

namespace cipc {

static bool floatnum_nearly_equal(real a, real b, real epsilon) {
    real abs_a = std::abs(a);
    real abs_b = std::abs(b);
    real diff = std::abs(a - b);
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || diff < std::numeric_limits<real>::min()) {
        return diff < (epsilon * std::numeric_limits<real>::min());
    } else {
        return diff / (abs_a + abs_b) < epsilon;
    }
}

struct gradient_diff_result {
    double numeric_diff;
    double analytic_diff;
};

template <typename RandomEngine>
static gradient_diff_result finite_gradient(
    const Matrix3Xr &x, RandomEngine &rng, std::function<double(const Matrix3Xr &)> f,
    const Matrix3Xr &grad, double eps = 1e-11) {
    integer vertex_num = static_cast<integer>(x.cols());
    integer dim = 3 * vertex_num;
    double energy = f(x);
    VectorXr test_dir = VectorXr::Zero(dim);
    static std::uniform_real_distribution<real> unit_random(0.0, 1.0);
    oneapi::tbb::parallel_for(0, dim, [&](integer i) {
        test_dir(i) = unit_random(rng);
    });

    while (test_dir.squaredNorm() < 1e-8) {
        oneapi::tbb::parallel_for(0, dim, [&](integer i) {
            test_dir(i) = unit_random(rng);
        });
    }

    gradient_diff_result result;
    Matrix3Xr x_plus = x + eps * test_dir.reshaped(3, vertex_num);
    Matrix3Xr x_minus = x - eps * test_dir.reshaped(3, vertex_num);
    double energy_plus = f(x_plus);
    double energy_minus = f(x_minus);
    result.numeric_diff = (energy_plus - energy_minus) / (2 * eps);
    result.analytic_diff = test_dir.dot(grad.reshaped());

    // result.numeric_diff_minus = (energy - energy_minus) / eps;
    // result.analytic_diff_minus = test_dir.dot(grad.reshaped());
    return result;
}

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

    while (test_dir.squaredNorm() < 1e-6) {
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
        if (floatnum_nearly_equal(numeric_diff_plus, analytic_diff_plus, 1e-3)) {
            printf("%d\n", i);
            return true;
        }

        Matrix3Xr x_minus = x - eps * test_dir.reshaped(3, vertex_num);
        real energy_minus = model.energy(x_minus);
        real numeric_diff_minus = (energy - energy_minus) / eps;
        real analytic_diff_minus = test_dir.dot(gradient.reshaped());

        if (floatnum_nearly_equal(numeric_diff_minus, analytic_diff_minus, 1e-3)) {
            printf("%d\n", i);
            return true;
        }
        printf("[plus]: analy: %f, numeric: %f\n", analytic_diff_plus, numeric_diff_plus);
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
    // printf("grad norm %.9f\n", grad.reshaped().norm());
    for (integer i = 4; i < 13; i++) {
        real eps = std::pow(10, (real)(-i));

        Matrix3Xr x_plus = x + eps * test_dir.reshaped(3, vertex_num);
        Matrix3Xr grad_plus = model.gradient(x_plus);

        VectorXr numeric_diff_plus = (grad_plus - grad).reshaped() / eps;
        VectorXr analytic_diff_plus = hessian * test_dir;
        real absolute_error_plus = (analytic_diff_plus - numeric_diff_plus).norm();

        // printf(
        // "analytic: %.9f, numeric: %.9f ", analytic_diff_plus.norm(), numeric_diff_plus.norm());
        // printf("absolute_error: %.9f\n", absolute_error_plus);
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

inline Matrix9r project_to_spd(const Matrix9r &m) {
    Eigen::SelfAdjointEigenSolver<Matrix9r> eigen_solver(m);
    const VectorXr &la = eigen_solver.eigenvalues();
    const MatrixXr &V = eigen_solver.eigenvectors();
    return V * la.cwiseMax(Eigen::Matrix<double, 9, 1>::Zero()).asDiagonal() * V.transpose();
}

inline Matrix12r project_to_spd(const Matrix12r &m) {
    Eigen::SelfAdjointEigenSolver<Matrix12r> eigen_solver(m);
    const VectorXr &la = eigen_solver.eigenvalues();
    const MatrixXr &V = eigen_solver.eigenvectors();
    return V * la.cwiseMax(Eigen::Matrix<double, 12, 1>::Zero()).asDiagonal() * V.transpose();
}
} // namespace cipc
