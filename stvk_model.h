#pragma once

#include <c-ipc/geometry/mesh.h>
#include <c-ipc/solver/eigen.h>
namespace cipc {

static Matrix2r
first_fundamental_form(const Matrix3Xr &curr_pos, const Matrix3Xi &elements, integer element_idx) {
    const Vector3r v0 = curr_pos.col(elements(0, element_idx));
    const Vector3r v1 = curr_pos.col(elements(1, element_idx));
    const Vector3r v2 = curr_pos.col(elements(2, element_idx));
    Matrix2r F_t;
    F_t << (v1 - v0).dot(v1 - v0), (v1 - v0).dot(v2 - v0), (v2 - v0).dot(v1 - v0),
        (v2 - v0).dot(v2 - v0);
    return F_t;
}

static Matrix4x9r first_fundamental_form_grad(
    const Matrix3Xr &curr_pos, const Matrix3Xi &elements, integer element_idx) {
    const Vector3r v0 = curr_pos.col(elements(0, element_idx));
    const Vector3r v1 = curr_pos.col(elements(1, element_idx));
    const Vector3r v2 = curr_pos.col(elements(2, element_idx));
    Matrix4x9r grad = Matrix4x9r::Zero();
    grad.block<1, 3>(0, 3) += 2.0 * (v1 - v0).transpose();
    grad.block<1, 3>(0, 0) -= 2.0 * (v1 - v0).transpose();
    grad.block<1, 3>(1, 6) += (v1 - v0).transpose();
    grad.block<1, 3>(1, 3) += (v2 - v0).transpose();
    grad.block<1, 3>(1, 0) += -(v1 - v0).transpose() - (v2 - v0).transpose();
    grad.block<1, 3>(2, 6) += (v1 - v0).transpose();
    grad.block<1, 3>(2, 3) += (v2 - v0).transpose();
    grad.block<1, 3>(2, 0) += -(v1 - v0).transpose() - (v2 - v0).transpose();
    grad.block<1, 3>(3, 6) += 2.0 * (v2 - v0).transpose();
    grad.block<1, 3>(3, 0) -= 2.0 * (v2 - v0).transpose();
    return grad;
}

static std::array<Matrix9r, 4> first_fundamental_form_hess() {
    const Matrix3r I = Matrix3r::Identity();
    std::array<Matrix9r, 4> Ft_hess;
    for (integer i = 0; i < 4; i++) { Ft_hess[i] = Matrix9r::Zero(); }

    Ft_hess[0].block<3, 3>(0, 0) += 2.0 * I;
    Ft_hess[0].block<3, 3>(3, 3) += 2.0 * I;
    Ft_hess[0].block<3, 3>(0, 3) -= 2.0 * I;
    Ft_hess[0].block<3, 3>(3, 0) -= 2.0 * I;

    Ft_hess[1].block<3, 3>(3, 6) += I;
    Ft_hess[1].block<3, 3>(6, 3) += I;
    Ft_hess[1].block<3, 3>(0, 3) -= I;
    Ft_hess[1].block<3, 3>(0, 6) -= I;
    Ft_hess[1].block<3, 3>(3, 0) -= I;
    Ft_hess[1].block<3, 3>(6, 0) -= I;
    Ft_hess[1].block<3, 3>(0, 0) += 2.0 * I;

    Ft_hess[2].block<3, 3>(3, 6) += I;
    Ft_hess[2].block<3, 3>(6, 3) += I;
    Ft_hess[2].block<3, 3>(0, 3) -= I;
    Ft_hess[2].block<3, 3>(0, 6) -= I;
    Ft_hess[2].block<3, 3>(3, 0) -= I;
    Ft_hess[2].block<3, 3>(6, 0) -= I;
    Ft_hess[2].block<3, 3>(0, 0) += 2.0 * I;

    Ft_hess[3].block<3, 3>(0, 0) += 2.0 * I;
    Ft_hess[3].block<3, 3>(6, 6) += 2.0 * I;
    Ft_hess[3].block<3, 3>(0, 6) -= 2.0 * I;
    Ft_hess[3].block<3, 3>(6, 0) -= 2.0 * I;
    return Ft_hess;
}

class StvK {
  public:
    double lambda = 3;
    double mu = 2;
    StvK() = default;
    StvK(const double lambda_, const double mu_) : lambda(lambda_), mu(mu_) {}

    const real ComputeEnergyDensity(const Matrix2r &F_t, const Matrix2r &F_t_bar) const {
        const Matrix2r M = F_t_bar.inverse() * F_t - Matrix2r::Identity();
        return 0.25 * (0.5 * lambda * M.trace() * M.trace() + mu * (M * M).trace());
    }

    const real ComputeStretchingEnergy(const Mesh &curr_mesh, const double thickness) const {
        const Matrix3Xi &elements = curr_mesh.indices;
        const Matrix3Xr &curr_pos = curr_mesh.vertices;
        const Matrix3Xr &rest_pos = curr_mesh.rest_vertices;
        double result = 0.0;
        for (integer e = 0; e < elements.cols(); e++) {

            const Vector3r v0 = curr_pos.col(elements(0, e));
            const Vector3r v1 = curr_pos.col(elements(1, e));
            const Vector3r v2 = curr_pos.col(elements(2, e));
            real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
            const Matrix2r F_t = first_fundamental_form(curr_pos, elements, e);
            const Matrix2r F_t_bar = first_fundamental_form(rest_pos, elements, e);
            result += element_area * thickness * ComputeEnergyDensity(F_t, F_t_bar);
        }
        return result;
    }

    const Matrix3Xr ComputeStretchingForce(
        const Mesh &curr_mesh, const double thickness,
        const std::vector<std::vector<std::array<integer, 2>>> &gradient_map) const {
        const Matrix3Xi &elements = curr_mesh.indices;
        const Matrix3Xr &curr_pos = curr_mesh.vertices;
        const Matrix3Xr &rest_pos = curr_mesh.rest_vertices;
        integer element_num = static_cast<integer>(elements.cols());
        std::vector<Matrix3r> gradient_per_element(element_num);
        for (integer e = 0; e < elements.cols(); e++) {
            const Vector3r v0 = curr_pos.col(elements(0, e));
            const Vector3r v1 = curr_pos.col(elements(1, e));
            const Vector3r v2 = curr_pos.col(elements(2, e));
            real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
            const Matrix2r F_t = first_fundamental_form(curr_pos, elements, e);
            const Matrix2r F_t_bar = first_fundamental_form(rest_pos, elements, e);
            const Matrix2r F_t_bar_inv = F_t_bar.inverse();
            const Matrix2r M = F_t_bar_inv * F_t - Matrix2r::Identity();
            const Matrix2r temp = lambda * M.trace() * F_t_bar_inv + 2 * mu * M * F_t_bar_inv;
            const Matrix4x9r F_t_grad = first_fundamental_form_grad(curr_pos, elements, e);
            gradient_per_element[e] = thickness * 0.25 * element_area
                                      * (F_t_grad.transpose() * temp.reshaped()).reshaped(3, 3);
        }
        integer vertex_num = static_cast<integer>(curr_pos.cols());
        Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);
        for (integer v = 0; v < vertex_num; v++) {
            for (const auto &tuple : gradient_map[v]) {
                const integer e = tuple[0];
                const integer i = tuple[1];
                gradient.col(v) += gradient_per_element[e].col(i);
            }
        }
        return -gradient;
    }

    const SparseMatrixXr ComputeStretchingHessian(
        const Mesh &curr_mesh, const double thickness,
        const std::vector<std::vector<std::array<integer, 3>>> &hessian_map,
        const SparseMatrixXr &hessian_prev) const {
        const Matrix3Xi &elements = curr_mesh.indices;
        const Matrix3Xr &curr_pos = curr_mesh.vertices;
        const Matrix3Xr &rest_pos = curr_mesh.rest_vertices;
        integer element_num = static_cast<integer>(elements.cols());
        std::vector<Matrix9r> hessian_per_element(element_num);
        for (integer e = 0; e < element_num; e++) {
            const Vector3r v0 = curr_pos.col(elements(0, e));
            const Vector3r v1 = curr_pos.col(elements(1, e));
            const Vector3r v2 = curr_pos.col(elements(2, e));
            real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
            const Matrix2r F_t = first_fundamental_form(curr_pos, elements, e);
            const Matrix2r F_t_bar = first_fundamental_form(rest_pos, elements, e);
            const Matrix2r F_t_bar_inv = F_t_bar.inverse();
            const Matrix2r M = F_t_bar_inv * F_t - Matrix2r::Identity();
            const Matrix2r temp = lambda * M.trace() * F_t_bar_inv + 2 * mu * M * F_t_bar_inv;
            const Matrix4x9r F_t_grad = first_fundamental_form_grad(curr_pos, elements, e);
            const Vector9r inner = F_t_grad.transpose() * F_t_bar_inv.reshaped();
            hessian_per_element[e] = lambda * inner * inner.transpose();
            Matrix2r M_dot_Finv = M * F_t_bar_inv;
            std::array<Matrix9r, 4> F_t_hess = first_fundamental_form_hess();
            for (integer i = 0; i < 4; i++) {
                hessian_per_element[e] +=
                    (lambda * M.trace() * F_t_bar_inv(i) + 2 * mu * M_dot_Finv(i)) * F_t_hess[i];
            }
            Eigen::Matrix<double, 1, 9> inner00 =
                F_t_bar_inv(0, 0) * F_t_grad.row(0) + F_t_bar_inv(0, 1) * F_t_grad.row(2);
            Eigen::Matrix<double, 1, 9> inner01 =
                F_t_bar_inv(0, 0) * F_t_grad.row(1) + F_t_bar_inv(0, 1) * F_t_grad.row(3);
            Eigen::Matrix<double, 1, 9> inner10 =
                F_t_bar_inv(1, 0) * F_t_grad.row(0) + F_t_bar_inv(1, 1) * F_t_grad.row(2);
            Eigen::Matrix<double, 1, 9> inner11 =
                F_t_bar_inv(1, 0) * F_t_grad.row(1) + F_t_bar_inv(1, 1) * F_t_grad.row(3);
            hessian_per_element[e] += 2 * mu * inner00.transpose() * inner00;
            hessian_per_element[e] +=
                2 * mu * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
            hessian_per_element[e] += 2 * mu * inner11.transpose() * inner11;

            hessian_per_element[e] *= thickness * 0.25 * element_area;
        }
        integer vertex_num = static_cast<integer>(curr_pos.cols());
        SparseMatrixXr ret(hessian_prev);
        integer hessian_nonzero_num = static_cast<integer>(hessian_map.size());
        for (integer v = 0; v < hessian_nonzero_num; v++) {
            real val = 0;
            for (const auto &arr : hessian_map[v]) {
                const integer e = arr[0];
                const integer i = arr[1];
                const integer j = arr[2];
                val += hessian_per_element[e](i, j);
            }
            ret.valuePtr()[v] = val;
        }
        return ret;
    }
};

} // namespace cipc
