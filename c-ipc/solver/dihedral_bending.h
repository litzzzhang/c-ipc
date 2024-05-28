#pragma once

#include "c-ipc/solver/eigen.h"
#include "c-ipc/geometry/mesh.h"
#include "c-ipc/backend/parallel.h"

namespace cipc {

// Data structures for querying edges.
struct TriangleEdgeInfo {
  public:
    real edge_length;
    // Index of the other triangle in elements_, -1 if the edge is at boundary
    integer other_triangle; // other triangle which shares same the edge
};

class DihedralBending {
  public:
    real bending_stiffness_;
    DihedralBending() = default;
    DihedralBending(const real bending_stiffness) : bending_stiffness_(bending_stiffness) {}

    static const real ComputeBendingAngle(const Vector3r &n1, const Vector3r &n2) {
        const real sin_angle = n1.cross(n2).norm();
        const real cos_angle = n1.dot(n2);
        const real angle = std::atan2(sin_angle, cos_angle);
        return angle;
    }

    static const real
    ComputeTriangleHeight(const Vector3r &v0, const Vector3r &v1, const Vector3r &v2) {
        return ((v0 - v1).cross(v2 - v1)).norm() / (v2 - v1).norm();
    }

    static const Vector3r ComputeFaceNormal(const Matrix3r &vertices) {
        Vector3r v0 = vertices.col(0);
        Vector3r v1 = vertices.col(1);
        Vector3r v2 = vertices.col(2);
        return (v1 - v0).cross(v2 - v0);
    }

    static const Matrix3r CrossProductMatrix(const Vector3r &a) {
        Matrix3r A = Matrix3r::Zero();
        A(1, 0) = a.z();
        A(2, 0) = -a.y();
        A(0, 1) = -a.z();
        A(2, 1) = a.x();
        A(0, 2) = a.y();
        A(1, 2) = -a.x();
        return A;
    }

    static const std::pair<Vector3r, Vector3r>
    ComputeDihedralAngleGradient(const Vector3r &normal, const Vector3r &other_normal) {
        const Vector3r nn_cross = normal.cross(other_normal);
        const real sin_angle = nn_cross.norm();
        Vector3r unit_nn_cross = Vector3r::Zero();
        if (sin_angle > static_cast<real>(1e-6) * normal.norm() * other_normal.norm())
            unit_nn_cross = nn_cross / sin_angle; // Avoid divide-by-zero.
        const Vector3r sin_angle_u = other_normal.cross(unit_nn_cross);
        const Vector3r sin_angle_v = -normal.cross(unit_nn_cross);

        const real cos_angle = normal.dot(other_normal);
        const Vector3r cos_angle_u = other_normal;
        const Vector3r cos_angle_v = normal;
        const real s2_and_c2 = normal.squaredNorm() * other_normal.squaredNorm();
        const real dangle_dsin = cos_angle / s2_and_c2;
        const real dangle_dcos = -sin_angle / s2_and_c2;

        const Vector3r angle_u = dangle_dsin * sin_angle_u + dangle_dcos * cos_angle_u;
        const Vector3r angle_v = dangle_dsin * sin_angle_v + dangle_dcos * cos_angle_v;
        return std::make_pair(angle_u, angle_v);
    }

    // angle = ComputeDihedralAngleFromNonUnitNormal(u, v)
    // The following function computes [angle_uu, angle_uv, angle_vv].
    static const Eigen::Matrix<real, 3, 9>
    ComputeDihedralAngleHessian(const Vector3r &normal, const Vector3r &other_normal) {
        const Vector3r nn_cross = normal.cross(other_normal);
        const real sin_angle = nn_cross.norm();
        Vector3r unit_nn_cross = Vector3r::Zero();
        Matrix3r unit_nn_cross_u = Matrix3r::Zero();
        Matrix3r unit_nn_cross_v = Matrix3r::Zero();
        const Matrix3r I = Matrix3r::Identity();
        if (sin_angle > static_cast<real>(1e-6) * normal.norm() * other_normal.norm()) {
            unit_nn_cross = nn_cross / sin_angle; // Avoid divide-by-zero.
            const Matrix3r N = (I - unit_nn_cross * unit_nn_cross.transpose()) / sin_angle;
            unit_nn_cross_u = N * -CrossProductMatrix(other_normal);
            unit_nn_cross_v = N * CrossProductMatrix(normal);
        }
        const Vector3r sin_angle_u = other_normal.cross(unit_nn_cross);
        const Vector3r sin_angle_v = -normal.cross(unit_nn_cross);
        const Matrix3r sin_angle_uu = CrossProductMatrix(other_normal) * unit_nn_cross_u;
        const Matrix3r sin_angle_uv =
            CrossProductMatrix(other_normal) * unit_nn_cross_v - CrossProductMatrix(unit_nn_cross);
        const Matrix3r sin_angle_vv = -CrossProductMatrix(normal) * unit_nn_cross_v;

        const real cos_angle = normal.dot(other_normal);
        const Vector3r cos_angle_u = other_normal;
        const Vector3r cos_angle_v = normal;

        const real u_sqr = normal.squaredNorm();
        const real v_sqr = other_normal.squaredNorm();
        const real s2_and_c2 = u_sqr * v_sqr;
        const Vector3r s2_and_c2_u = v_sqr * 2 * normal;
        const Vector3r s2_and_c2_v = u_sqr * 2 * other_normal;
        const real dangle_dsin = cos_angle / s2_and_c2;
        const real dangle_dcos = -sin_angle / s2_and_c2;
        const Vector3r dangle_dsin_u = (cos_angle_u - dangle_dsin * s2_and_c2_u) / s2_and_c2;
        const Vector3r dangle_dsin_v = (cos_angle_v - dangle_dsin * s2_and_c2_v) / s2_and_c2;
        const Vector3r dangle_dcos_u = (-sin_angle_u - dangle_dcos * s2_and_c2_u) / s2_and_c2;
        const Vector3r dangle_dcos_v = (-sin_angle_v - dangle_dcos * s2_and_c2_v) / s2_and_c2;

        const Matrix3r angle_uu = dangle_dsin * sin_angle_uu
                                  + sin_angle_u * dangle_dsin_u.transpose()
                                  + cos_angle_u * dangle_dcos_u.transpose();
        const Matrix3r angle_uv = dangle_dsin * sin_angle_uv
                                  + sin_angle_u * dangle_dsin_v.transpose() + dangle_dcos * I
                                  + cos_angle_u * dangle_dcos_v.transpose();
        const Matrix3r angle_vv = dangle_dsin * sin_angle_vv
                                  + sin_angle_v * dangle_dsin_v.transpose()
                                  + cos_angle_v * dangle_dcos_v.transpose();

        Eigen::Matrix<real, 3, 9> hess;
        hess.leftCols(3) = angle_uu;
        hess.middleCols(3, 3) = angle_uv;
        hess.rightCols(3) = angle_vv;

        return hess;
    }

    static const Eigen::Matrix<real, 3, 9> ComputeNormalGradient(const Matrix3r &vertices) {
        const Vector3r normal =
            (vertices.col(1) - vertices.col(0)).cross(vertices.col(2) - vertices.col(1));

        Eigen::Matrix<real, 3, 9> grad_normal;
        for (integer i = 0; i < 3; ++i) {
            grad_normal.middleCols<3>(3 * i) =
                CrossProductMatrix(vertices.col((i + 2) % 3) - vertices.col((i + 1) % 3));
        }

        return grad_normal;
    }

    static const std::array<Eigen::Matrix<real, 3, 9>, 9> ComputeNormalHessian() {
        std::array<Eigen::Matrix<real, 3, 9>, 9> hess;
        for (integer i = 0; i < 9; ++i) hess[i].setZero();
        for (integer i = 0; i < 3; ++i) {
            for (integer j = 0; j < 3; ++j) {
                hess[((i + 2) % 3) * 3 + j].middleCols<3>(3 * i) +=
                    CrossProductMatrix(Vector3r::Unit(j));
                hess[((i + 1) % 3) * 3 + j].middleCols<3>(3 * i) +=
                    CrossProductMatrix(-Vector3r::Unit(j));
            }
        }

        return hess;
    }

    const real ComputeBendingEnergy(
        const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;

    const Matrix3Xr ComputeBendingForce(
        const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;

    const SparseMatrixXr ComputeBendingHessian(
        const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;
};

inline const real DihedralBending::ComputeBendingEnergy(
    const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer element_num = static_cast<integer>(current_mesh.indices.cols());
    real energy = oneapi::tbb::parallel_deterministic_reduce(
        oneapi::tbb::blocked_range<integer>(0, element_num), 0.0,
        [&](oneapi::tbb::blocked_range<integer> r, real local) {
            for (integer e = r.begin(); e < r.end(); e++) {
                const Vector3r normal = ComputeFaceNormal(
                    current_mesh.vertices(Eigen::all, current_mesh.indices.col(e)));
                for (integer i = 0; i < 3; i++) {
                    const TriangleEdgeInfo &info = edge_info[e][i];
                    if (info.other_triangle == -1) { continue; }
                    const Vector3r other_normal = ComputeFaceNormal(current_mesh.vertices(
                        Eigen::all, current_mesh.indices.col(info.other_triangle)));
                    const real angle = ComputeBendingAngle(normal, other_normal);
                    const real rest_edge_length = info.edge_length;
                    local +=
                        3.0 * angle * angle * rest_edge_length * rest_edge_length / rest_area(e);
                }
            }
            return local;
        },
        [](real x, real y) {
            return x + y;
        });
    return bending_stiffness_ * energy;
}

inline const Matrix3Xr DihedralBending::ComputeBendingForce(
    const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer vertex_num = static_cast<integer>(current_mesh.vertices.cols());
    const integer element_num = static_cast<integer>(current_mesh.indices.cols());
    Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);

    std::vector<Matrix3r> gradient_per_element;
    gradient_per_element.assign(element_num, Matrix3r::Zero());
    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        const Matrix3r &this_vertices =
            current_mesh.vertices(Eigen::all, current_mesh.indices.col(e));
        const Vector3r normal = ComputeFaceNormal(this_vertices);
        // iterate each edge
        for (integer i = 0; i < 3; i++) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) { continue; }
            const Matrix3r &other_vertices =
                current_mesh.vertices(Eigen::all, current_mesh.indices.col(info.other_triangle));
            const Vector3r other_normal = ComputeFaceNormal(other_vertices);
            const real angle = ComputeBendingAngle(normal, other_normal);
            const real rest_edge_length = info.edge_length;

            auto [dtheta_dn, other_dthetadn] = ComputeDihedralAngleGradient(normal, other_normal);
            const real coeff =
                6 * rest_edge_length * rest_edge_length / rest_area(e) * std::fabs(angle);

            auto d_normal_dx = ComputeNormalGradient(this_vertices);
            gradient_per_element[e] += coeff * (d_normal_dx.transpose() * dtheta_dn).reshaped(3, 3);
        }
        cipc_assert(!gradient_per_element[e].hasNaN(), "element {} gradient has nan", e);
    });

    for (integer e = 0; e < element_num; e++) {
        integer idx1 = current_mesh.indices(0, e);
        integer idx2 = current_mesh.indices(1, e);
        integer idx3 = current_mesh.indices(2, e);
        gradient.col(idx1) += gradient_per_element[e].col(0);
        gradient.col(idx2) += gradient_per_element[e].col(1);
        gradient.col(idx3) += gradient_per_element[e].col(2);
    }
    return -bending_stiffness_ * gradient;
}

inline const SparseMatrixXr DihedralBending::ComputeBendingHessian(
    const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer hess_size = static_cast<integer>(current_mesh.vertices.cols()) * 3;
    SparseMatrixXr Hess(hess_size, hess_size);
    Hess.setZero();
    const integer element_num = static_cast<integer>(current_mesh.indices.cols());
    const Matrix3Xi &elements_ = current_mesh.indices;
    const Matrix3Xr &position = current_mesh.vertices;
    std::vector<Matrix9r> hess_per_element;
    hess_per_element.assign(element_num, Matrix9r::Zero());
    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        const Matrix3r &this_vertices =
            current_mesh.vertices(Eigen::all, current_mesh.indices.col(e));
        const Vector3r normal = ComputeFaceNormal(this_vertices);
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) continue;
            const Matrix3r &other_vertices =
                current_mesh.vertices(Eigen::all, current_mesh.indices.col(info.other_triangle));
            const Vector3r other_normal = ComputeFaceNormal(other_vertices);
            const real angle = ComputeBendingAngle(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real coeff1 = 6 * rest_shape_edge_length * rest_shape_edge_length / rest_area(e);
            const real coeff2 = coeff1 * std::fabs(angle);
            // compute grad theta
            auto [dtheta_dn, other_dthetadn] = ComputeDihedralAngleGradient(normal, other_normal);
            auto d_normal_dx = ComputeNormalGradient(this_vertices);
            auto d_othernormal_dx = ComputeNormalGradient(other_vertices);
            Vector9r dtheta_dx = d_normal_dx.transpose() * dtheta_dn;

            hess_per_element[e] += coeff1 * dtheta_dx * dtheta_dx.transpose();
            // compute hess theta
            auto normal_hessian_list = ComputeNormalHessian();
            for (integer idx = 0; idx < 9; idx++) {
                hess_per_element[e].col(idx) +=
                    coeff2 * normal_hessian_list[idx].transpose() * (dtheta_dn + other_dthetadn);
            }

            Eigen::Matrix<real, 3, 9> ddtheta_d2n =
                ComputeDihedralAngleHessian(normal, other_normal);

            Matrix9r diff_expand = Matrix9r::Zero();
            diff_expand.block(0, 0, 3, 9) = d_normal_dx;
            diff_expand.block(3, 0, 3, 9) = d_normal_dx + d_othernormal_dx;
            diff_expand.block(6, 0, 3, 9) = d_othernormal_dx;
            hess_per_element[e] += coeff2 * d_normal_dx.transpose() * ddtheta_d2n * diff_expand;
        }
    });
    for (int e = 0; e < element_num; e++) {
        const Matrix9r &hess = hess_per_element[e];
        for (int v1 = 0; v1 < 3; v1++) {
            for (int v2 = 0; v2 < 3; v2++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        integer idx1 = 3 * elements_(v1, e) + i;
                        integer idx2 = 3 * elements_(v2, e) + j;
                        cipc_assert(3 * v1 + i < 9 && 3 * v2 + j < 9, "out of range");
                        Hess.coeffRef(idx1, idx2) += hess(3 * v1 + i, 3 * v2 + j);
                    }
                }
            }
        }
    }

    Hess.makeCompressed();
    return bending_stiffness_ * Hess;
}
} // namespace cipc
