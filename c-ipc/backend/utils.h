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
    const Matrix3Xr &grad, double eps = 1e-8) {
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

    return result;
}

struct hessian_diff_result {
    VectorXr numeric_diff;
    VectorXr analytic_diff;
};

template <typename RandomEngine>
static hessian_diff_result finite_hessian(
    const Matrix3Xr &x, RandomEngine &rng, std::function<Matrix3Xr(const Matrix3Xr &)> g,
    const SparseMatrixXr &hess, double eps = 1e-8) {
    integer vertex_num = static_cast<integer>(x.cols());
    integer dim = 3 * vertex_num;
    Matrix3Xr gradient = g(x);

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

    hessian_diff_result result;
    Matrix3Xr x_plus = x + eps * test_dir.reshaped(3, vertex_num);
    Matrix3Xr x_minus = x - eps * test_dir.reshaped(3, vertex_num);
    Matrix3Xr gradient_plus = g(x_plus);
    Matrix3Xr gradient_minus = g(x_minus);
    result.numeric_diff = (gradient_plus - gradient_minus).reshaped() / (2 * eps);
    result.analytic_diff = hess * test_dir.reshaped();

    return result;
}

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
