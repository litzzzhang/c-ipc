#pragma once

#include "c-ipc/solver/eigen.h"
#include "c-ipc/solver/dihedral_bending.h"

namespace cipc {

template <typename MaterialType>
class Simulator {
  public:
    Simulator(
        const Mesh &mesh, const real thickness, const real bending_stiffness, const real density);
    const Matrix3Xr get_position() const { return current_mesh_.vertices; }
    const Matrix3Xi get_indice() const { return current_mesh_.indices; }
    void set_position(const Matrix3Xr &new_pos) { current_mesh_.vertices = new_pos; }
    void set_velocity(const Matrix3Xr &new_vel) { velocity_ = new_vel; }
    void set_external_acceleration(const Matrix3Xr &new_acc) { external_acceleration_ = new_acc; }

    void Forward(const real timestep);
    // Stretching and shearing
    const real ComputeStretchingAndShearingEnergy(const Matrix3Xr &position) const {
        return material_.ComputeStretchingEnergy(position, current_mesh_.indices, D_inv_);
    };
    const Matrix3Xr ComputeStretchingAndShearingForce(const Matrix3Xr &position) const {
        return material_.ComputeStretchingForce(
            position, current_mesh_.indices, D_inv_, stretching_and_shearing_gradient_map_);
    };
    const SparseMatrixXr ComputeStretchingAndShearingHessian(const Matrix3Xr &position) const {
        return material_.ComputeStretchingHessian(
            position, current_mesh_.indices, D_inv_, stretching_and_shearing_hessian_nonzero_map_,
            stretching_and_shearing_hessian_);
    };

    // bending
    const real ComputeBendingEnergy(const Matrix3Xr &position) const {
        return bending_model_.ComputeBendingEnergy(current_mesh_, triangle_edge_info_, rest_area_);
    };
    const Matrix3Xr ComputeBendingForce(const Matrix3Xr &position) const {
        return bending_model_.ComputeBendingForce(current_mesh_, triangle_edge_info_, rest_area_);
    };
    const SparseMatrixXr ComputeBendingHessian(const Matrix3Xr &position) const {
        return bending_model_.ComputeBendingHessian(current_mesh_, triangle_edge_info_, rest_area_);
    };

    MaterialType material_;
    DihedralBending bending_model_;
    // x
    Mesh current_mesh_;
    // v
    Matrix3Xr velocity_;
    // a_ext
    Matrix3Xr external_acceleration_;
    // M
    SparseMatrixXr int_matrix_;
    // area per triangle
    VectorXr rest_area_;
    // Basis function derivatives
    std::vector<Matrix3x2r> D_inv_;
    const real bending_stiffness_;
    const real thickness_;
    const real density_;
    // Edge data structure for computing bending.
    // triangle_edge_info_[e][i] is the i-th edge of triangle elements_.col(e), which is the
    // vertices in index (elements_(i, e), elements_((i+1)%3, e)).
    std::vector<std::array<TriangleEdgeInfo, 3>> triangle_edge_info_;

    // Accelerating the matrix assembly.
    std::vector<std::vector<std::array<integer, 2>>> stretching_and_shearing_gradient_map_;
    SparseMatrixXr stretching_and_shearing_hessian_;
    integer stretching_and_shearing_hessian_nonzero_num_;
    std::vector<std::vector<std::array<integer, 3>>> stretching_and_shearing_hessian_nonzero_map_;
};

template <typename MaterialType>
inline Simulator<MaterialType>::Simulator(
    const Mesh &mesh, const real thickness, const real bending_stiffness, const real density)
    : thickness_(thickness), bending_stiffness_(bending_stiffness), density_(density) {
    // TO DO: change the degreee of freedom with constrain
    current_mesh_ = mesh;
    const Matrix3Xr &vertices = mesh.vertices;
    const Matrix3Xi &indices = mesh.indices;

    const integer dof_num = static_cast<integer>(vertices.cols());
    // init velocity and acceleration
    velocity_ = Matrix3Xr::Zero(3, dof_num);
    external_acceleration_ = Matrix3Xr::Zero(3, dof_num);

    bending_model_.bending_stiffness_ = bending_stiffness;
    // precompute mass matrix
    std::vector<Eigen::Triplet<real>> int_mat_nonzeros;
    const integer element_num = static_cast<integer>(indices.cols());
    rest_area_.setZero(element_num);
    for (integer e = 0; e < element_num; e++) {
        auto v0 = vertices.col(indices.col(e)(0));
        auto v1 = vertices.col(indices.col(e)(1));
        auto v2 = vertices.col(indices.col(e)(2));
        rest_area_(e) = 0.5f * ((v1 - v0).cross(v2 - v0)).norm();
        cipc_assert(rest_area_(e) > 0.0f, "Area of elements must be positive!");
        for (integer i = 0; i < 3; i++) {
            for (integer j = 0; j < 3; j++) {
                int_mat_nonzeros.emplace_back(
                    indices(i, e), indices(j, e), rest_area_(e) / (i == j ? 6 : 12));
            }
        }
    }
    int_matrix_ = FromTriplet(dof_num, dof_num, int_mat_nonzeros);

    for (integer e = 0; e < element_num; e++) {
        Matrix3r Ds = vertices(Eigen::all, indices.col(e));
        Ds.row(2) = Vector3r::Ones();
        D_inv_.push_back(Ds.inverse().leftCols(2));
    }

    // assemble gradient map
    {
        stretching_and_shearing_gradient_map_.clear();
        stretching_and_shearing_gradient_map_.resize(dof_num);
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = current_mesh_.indices.col(e);
            for (integer i = 0; i < 3; ++i) {
                stretching_and_shearing_gradient_map_[dof_map(i)].push_back({e, i});
            }
        }
    }

    // Assemble the nonzero structures in stretching and shearing energy Hessian.
    {
        const integer dim = 3;
        std::vector<Eigen::Triplet<real>> stretching_and_shearing_hess_nonzeros;
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = current_mesh_.indices.col(e);
            for (integer i = 0; i < 3; ++i)
                for (integer j = 0; j < 3; ++j)
                    for (integer di = 0; di < dim; ++di)
                        for (integer dj = 0; dj < dim; ++dj) {
                            const integer row_idx = dof_map(i) * dim + di;
                            const integer col_idx = dof_map(j) * dim + dj;
                            stretching_and_shearing_hess_nonzeros.emplace_back(
                                row_idx, col_idx, static_cast<real>(1));
                        }
        }
        stretching_and_shearing_hessian_ =
            FromTriplet(3 * dof_num, 3 * dof_num, stretching_and_shearing_hess_nonzeros);

        stretching_and_shearing_hessian_nonzero_num_ =
            static_cast<integer>(stretching_and_shearing_hessian_.nonZeros());
        stretching_and_shearing_hessian_nonzero_map_.clear();
        stretching_and_shearing_hessian_nonzero_map_.resize(
            stretching_and_shearing_hessian_nonzero_num_);
        for (integer e = 0; e < element_num; ++e) {
            const VectorXi dof_map = current_mesh_.indices.col(e);
            for (integer i = 0; i < 3; ++i)
                for (integer j = 0; j < 3; ++j)
                    for (integer di = 0; di < dim; ++di)
                        for (integer dj = 0; dj < dim; ++dj) {
                            const integer row_idx = dof_map(i) * dim + di;
                            const integer col_idx = dof_map(j) * dim + dj;
                            const integer k = static_cast<integer>(
                                &stretching_and_shearing_hessian_.coeffRef(row_idx, col_idx)
                                - stretching_and_shearing_hessian_.valuePtr());
                            stretching_and_shearing_hessian_nonzero_map_[k].push_back(
                                {e, i * dim + di, j * dim + dj});
                        }
        }
    }

    // Construct triangle_edge_info_.
    triangle_edge_info_.clear();
    triangle_edge_info_.resize(element_num);
    for (integer e = 0; e < element_num; ++e) {
        const Eigen::Matrix<real, 2, 3> &vertices_2d =
            vertices(Eigen::all, current_mesh_.indices.col(e)).block(0, 0, 2, 3);
        for (integer i = 0; i < 3; ++i) {
            auto &info = triangle_edge_info_[e][i];
            info.edge_length = (vertices_2d.col(i) - vertices_2d.col((i + 1) % 3)).norm();
            info.other_triangle = -1;
        }
    }
    std::map<std::pair<integer, integer>, std::pair<integer, integer>> edge_map;
    for (integer e = 0; e < element_num; ++e) {
        for (integer i = 0; i < 3; ++i) {
            const integer idx0 = indices(i, e);
            const integer idx1 = indices((i + 1) % 3, e);
            const std::pair<integer, integer> key =
                std::make_pair(idx0 < idx1 ? idx0 : idx1, idx0 < idx1 ? idx1 : idx0);
            if (edge_map.find(key) == edge_map.end()) {
                // We haven't see this edge before.
                edge_map[key] = std::make_pair(e, i);
            } else {
                // We have seen this edge before.
                const integer other = edge_map[key].first;
                const integer other_edge = edge_map[key].second;
                triangle_edge_info_[e][i].other_triangle = other;
                edge_map.erase(key);
            }
        }
    }
}

template <typename MaterialType>
inline void Simulator<MaterialType>::Forward(const real timestep) {
    const real h = timestep;
    const real inv_h = 1.0f / h;
    const integer dof_num = static_cast<integer>(current_mesh_.vertices.cols());

    const Matrix3Xr &x0 = current_mesh_.vertices;
    const Matrix3Xr &v0 = velocity_;
    const Matrix3Xr &a = external_acceleration_;

    const Matrix3Xr y = x0 + v0 * h + a * h * h;
    const real half_rho_inv_h2 = density_ * inv_h * inv_h / 2;

    auto E = [&](const Matrix3Xr &x_next) -> real {
        real energy_kinetic = 0;

        for (integer d = 0; d < 3; ++d) {
            const VectorXr x_next_d(x_next.row(d));
            const VectorXr y_d(y.row(d));
            const VectorXr diff_d = x_next_d - y_d;
            energy_kinetic += diff_d.dot(int_matrix_ * diff_d);
        }
        energy_kinetic *= half_rho_inv_h2;
        const real energy_ss = ComputeStretchingAndShearingEnergy(x_next);
        const real energy_bending = ComputeBendingEnergy(x_next);
        return energy_kinetic + energy_ss + energy_bending;
        // return energy_kinetic;
        // return energy_kinetic + energy_ss;
    };

    // Its gradient.
    auto grad_E = [&](const Matrix3Xr &x_next) -> const VectorXr {
        Matrix3Xr gradient_kinetic = Matrix3Xr::Zero(3, x_next.cols());
        for (integer d = 0; d < 3; ++d) {
            const VectorXr x_next_d(x_next.row(d));
            const VectorXr y_d(y.row(d));
            const VectorXr diff_d = x_next_d - y_d;
            gradient_kinetic.row(d) += RowVectorXr(int_matrix_ * diff_d);
        }
        gradient_kinetic *= density_ * inv_h * inv_h;
        const Matrix3Xr gradient_ss = -ComputeStretchingAndShearingForce(x_next);
        const Matrix3Xr gradient_bending = -ComputeBendingForce(x_next);
        return (gradient_kinetic + gradient_ss + gradient_bending).reshaped();
        // return (gradient_kinetic).reshaped();
        // return (gradient_kinetic + gradient_ss).reshaped();
    };
    auto Hess_E = [&](const Matrix3Xr &x_next) -> const SparseMatrixXr {
        std::vector<Eigen::Triplet<real>> kinetic_nonzeros;
        std::vector<Eigen::Triplet<real>> int_nonzeros = ToTriplet(int_matrix_);
        const real scale = density_ * inv_h * inv_h;
        for (const auto &triplet : int_nonzeros)
            for (integer d = 0; d < 3; ++d) {
                kinetic_nonzeros.push_back(Eigen::Triplet<real>(
                    triplet.row() * 3 + d, triplet.col() * 3 + d, triplet.value() * scale));
            }
        const SparseMatrixXr H_kinetic = FromTriplet(3 * dof_num, 3 * dof_num, kinetic_nonzeros);
        const SparseMatrixXr H_ss = ComputeStretchingAndShearingHessian(x_next);
        const SparseMatrixXr H_bending = ComputeBendingHessian(x_next);
        // return H_kinetic;
        // return H_kinetic + H_ss;
        return H_kinetic + H_ss + H_bending;
    };

    Matrix3Xr xk = x0;
    real Ek = E(xk);
    VectorXr gk = grad_E(xk);
    integer newton_iter = 100;
    while (gk.cwiseAbs().maxCoeff() > 1e-5) {
        cipc_assert(newton_iter > 0, "Newton iteration failed");
        Eigen::SimplicialLDLT<SparseMatrixXr> direct_solver(Hess_E(xk));
        const VectorXr pk = direct_solver.solve(-gk);

        // line search
        // TO DO: add ACCD aware line search
        real ls_step = 1.0f;
        real E_updated = E(xk + pk.reshaped(3, dof_num));
        integer ls_iter = 50;
        while (E_updated > Ek + 0.01f * ls_step * gk.dot(pk)) {
            cipc_assert(
                ls_iter > 0, "Line search failed to find sufficient decrease");
            ls_step /= 2;
            E_updated = E(xk + ls_step * pk.reshaped(3, dof_num));
            ls_iter -= 1;
        }
        xk += ls_step * pk.reshaped(3, dof_num);
        // Exit if no progress could be made.
        if (ls_step * pk.cwiseAbs().maxCoeff() <= 1e-12) { break; }

        Ek = E_updated;
        gk = grad_E(xk);
        newton_iter -= 1;
    }

    // update pos and vel
    const Matrix3Xr next_position = xk;
    velocity_ = (next_position - current_mesh_.vertices) * inv_h;
    current_mesh_.vertices = next_position;

    // update current mesh
    current_mesh_.ComputeFaceNormals();
    // naive collision
    for (integer i = 0; i < static_cast<integer>(current_mesh_.vertices.cols()); ++i) {
        Vector3r point = current_mesh_.vertices.col(i);
        const real point_norm = point.squaredNorm();
        if (point_norm < 0.35) {
            const real projected_velocity = velocity_.col(i).dot(point);
            if (projected_velocity < 0) velocity_.col(i) -= projected_velocity * point /
            point_norm;
        }
    }
}

} // namespace cipc
