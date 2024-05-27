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
        return std::acos(n1.dot(n2));
    }

    static const real
    ComputeTriangleHeight(const Vector3r &v0, const Vector3r &v1, const Vector3r &v2) {
        return ((v0 - v1).cross(v2 - v1)).norm() / (v2 - v1).norm();
    }

    static const Vector4i FindVertexIndex(const Vector3i &element0, const Vector3i &element1) {
        integer v0, v1, v2, v3;
        for (integer i = 0; i < 3; i++) {
            bool is_in_element0 = false;
            for (integer j = 0; j < 3; j++) {
                if (element0(i) == element1(j)) { is_in_element0 = true; }
            }
            if (!is_in_element0) { v0 = element1(i); }
        }

        for (int i = 0; i < 3; i++) {
            if (element1(i) != v0) { v1 = element1(i); }
        }
        for (int i = 0; i < 3; i++) {
            if (element1(i) != v0 && element1(i) != v1) { v2 = element1(i); }
        }
        for (int i = 0; i < 3; i++) {
            if (element0(i) != v1 && element0(i) != v2) { v3 = element0(i); }
        }
        return Vector4i(v0, v1, v2, v3);
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
    cipc_assert(
        current_mesh.face_normal_valid,
        "face normal invalid, compute face normal before bending calculation!");

    const integer element_num = static_cast<integer>(current_mesh.indices.cols());
    real energy = 0.0f;
    // TO DO: use parallel reduce to accelerate
    for (integer e = 0; e < element_num; e++) {
        const Vector3r normal = current_mesh.face_normals.col(e);

        // iterate three edges
        for (integer i = 0; i < 3; i++) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) { continue; }
            const Vector3r other_normal = current_mesh.face_normals.col(info.other_triangle);
            const real angle = ComputeBendingAngle(normal, other_normal);
            const real rest_edge_length = info.edge_length;
            energy += 3.0f * angle * angle * rest_edge_length * rest_edge_length / rest_area(e);
        }
    }
    return bending_stiffness_ * energy;
}

inline const Matrix3Xr DihedralBending::ComputeBendingForce(
    const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer vertex_num = static_cast<integer>(current_mesh.vertices.cols());
    const integer element_num = static_cast<integer>(current_mesh.indices.cols());
    Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);

    // TO DO: parallel
    for (integer e = 0; e < element_num; e++) {
        const Vector3r normal = current_mesh.face_normals.col(e);
        // iterate each edge
        for (integer i = 0; i < 3; i++) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) { continue; }
            const Vector3r other_normal = current_mesh.face_normals.col(info.other_triangle);
            const real angle = ComputeBendingAngle(normal, other_normal);
            const real rest_edge_length = info.edge_length;
            Vector4i hinge_vectex_idx = FindVertexIndex(
                current_mesh.indices.col(e), current_mesh.indices.col(info.other_triangle));
            const Vector3r v0 = current_mesh.vertices.col(hinge_vectex_idx(0));
            const Vector3r v1 = current_mesh.vertices.col(hinge_vectex_idx(1));
            const Vector3r v2 = current_mesh.vertices.col(hinge_vectex_idx(2));
            const Vector3r v3 = current_mesh.vertices.col(hinge_vectex_idx(3));
            // face normal
            Vector3r n1 = ((v2 - v1).cross(v0 - v1)).normalized();
            Vector3r n2 = ((v3 - v1).cross(v2 - v1)).normalized();
            // height
            real h[3][2];
            h[0][0] = ComputeTriangleHeight(v0, v1, v2);
            h[1][0] = ComputeTriangleHeight(v1, v0, v2);
            h[2][0] = ComputeTriangleHeight(v2, v0, v1);
            h[0][1] = ComputeTriangleHeight(v3, v1, v2);
            h[1][1] = ComputeTriangleHeight(v1, v3, v2);
            h[2][1] = ComputeTriangleHeight(v2, v1, v3);
            // edge
            Vector3r edge[3][2];
            edge[0][0] = (v2 - v1);
            edge[0][1] = (v2 - v1);
            edge[1][0] = (v0 - v2);
            edge[2][0] = (v0 - v1);
            edge[1][1] = (v3 - v2);
            edge[2][1] = (v3 - v1);
            // cos alpha
            real cos[2][2];
            cos[0][0] = ((edge[0][0]).normalized()).dot((edge[2][0]).normalized());
            cos[0][1] = ((edge[0][0]).normalized()).dot((edge[2][1]).normalized());
            cos[1][0] = -((edge[0][0]).normalized()).dot((edge[1][0]).normalized());
            cos[1][1] = -((edge[0][0]).normalized()).dot((edge[1][1]).normalized());
            real sign = 1;
            if ((n1.cross(n2)).dot(v2 - v1) < 0) sign = -1;
            const real coefficient_signed =
                6 * rest_edge_length * rest_edge_length / rest_area(e) * abs(angle) * sign;
            // compute grad theta
            for (int row = 0; row < 3; row++) {
                gradient(row, hinge_vectex_idx(0)) += -coefficient_signed / h[0][0] * n1(row);
                gradient(row, hinge_vectex_idx(1)) +=
                    coefficient_signed
                    * (cos[1][0] / h[1][0] * n1(row) + cos[1][1] / h[1][1] * n2(row));
                gradient(row, hinge_vectex_idx(2)) +=
                    coefficient_signed
                    * (cos[0][0] / h[2][0] * n1(row) + cos[0][1] / h[2][1] * n2(row));
                gradient(row, hinge_vectex_idx(3)) += -coefficient_signed / h[0][1] * n2(row);
            }
        }
    }
    return -bending_stiffness_ * gradient;
}

// TO DO: parallel
inline const SparseMatrixXr DihedralBending::ComputeBendingHessian(
    const Mesh &current_mesh, const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer hess_size = static_cast<integer>(current_mesh.vertices.cols()) * 3;
    SparseMatrixXr Hess(hess_size, hess_size);
    Hess.setZero();
    const integer element_num = static_cast<integer>(current_mesh.indices.cols());

    for (integer e = 0; e < element_num; e++) {
        for (integer i = 0; i < 3; i++) {
            const Vector3r normal = current_mesh.face_normals.col(e);
            // iterate each edge
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) { continue; }
            const Vector3r other_normal = current_mesh.face_normals.col(info.other_triangle);
            const real angle = ComputeBendingAngle(normal, other_normal);
            const real rest_edge_length = info.edge_length;
            Vector4i hinge_vectex_idx = FindVertexIndex(
                current_mesh.indices.col(e), current_mesh.indices.col(info.other_triangle));
            const Vector3r v0 = current_mesh.vertices.col(hinge_vectex_idx(0));
            const Vector3r v1 = current_mesh.vertices.col(hinge_vectex_idx(1));
            const Vector3r v2 = current_mesh.vertices.col(hinge_vectex_idx(2));
            const Vector3r v3 = current_mesh.vertices.col(hinge_vectex_idx(3));
            // face normal
            Vector3r n1 = ((v2 - v1).cross(v0 - v1)).normalized();
            Vector3r n2 = ((v3 - v1).cross(v2 - v1)).normalized();
            // height
            real h[3][2];
            h[0][0] = ComputeTriangleHeight(v0, v1, v2);
            h[1][0] = ComputeTriangleHeight(v1, v0, v2);
            h[2][0] = ComputeTriangleHeight(v2, v0, v1);
            h[0][1] = ComputeTriangleHeight(v3, v1, v2);
            h[1][1] = ComputeTriangleHeight(v1, v3, v2);
            h[2][1] = ComputeTriangleHeight(v2, v1, v3);
            // edge
            Vector3r edge[3][2];
            edge[0][0] = (v2 - v1);
            edge[0][1] = (v2 - v1);
            edge[1][0] = (v0 - v2);
            edge[2][0] = (v0 - v1);
            edge[1][1] = (v3 - v2);
            edge[2][1] = (v3 - v1);
            // cos alpha
            real cos[2][2];
            cos[0][0] = ((edge[0][0]).normalized()).dot((edge[2][0]).normalized());
            cos[0][1] = ((edge[0][0]).normalized()).dot((edge[2][1]).normalized());
            cos[1][0] = -((edge[0][0]).normalized()).dot((edge[1][0]).normalized());
            cos[1][1] = -((edge[0][0]).normalized()).dot((edge[1][1]).normalized());

            // edge normal
            Vector3r m[3][2];
            m[0][0] = edge[0][0].cross(n1).normalized();
            m[1][0] = edge[1][0].cross(n1).normalized();
            m[2][0] = n1.cross(edge[2][0]).normalized();
            m[0][1] = n2.cross(edge[0][0]).normalized();
            m[1][1] = n2.cross(edge[1][1]).normalized();
            m[2][1] = edge[2][1].cross(n2).normalized();
            real sign = 1;
            if ((n1.cross(n2)).dot(v2 - v1) < 0) sign = -1;
            const real coefficient = 6 * rest_edge_length * rest_edge_length / rest_area(e);
            const real coefficient_signed =
                6 * rest_edge_length * rest_edge_length / rest_area(e) * abs(angle) * sign;
            // compute grad theta
            Matrix3Xr gradient_theta = Matrix3Xr::Zero(3, 4);
            for (int row = 0; row < 3; row++) {
                gradient_theta(row, 0) += -1 / h[0][0] * n1(row);
                gradient_theta(row, 1) +=
                    cos[1][0] / h[1][0] * n1(row) + cos[1][1] / h[1][1] * n2(row);
                gradient_theta(row, 2) +=
                    cos[0][0] / h[2][0] * n1(row) + cos[0][1] / h[2][1] * n2(row);
                gradient_theta(row, 3) += -1 / h[0][1] * n2(row);
            }
            Matrix12r H = Matrix12r::Zero();
            H.block<3, 3>(0 * 3, 0 * 3) =
                -1 / h[0][0] / h[0][0] * (m[0][0] * n1.transpose() + n1 * m[0][0].transpose());
            H.block<3, 3>(3 * 3, 3 * 3) =
                -1 / h[0][1] / h[0][1] * (m[0][1] * n2.transpose() + n2 * m[0][1].transpose());
            H.block<3, 3>(1 * 3, 1 * 3) =
                cos[1][0] / (h[1][0] * h[1][0])
                    * (m[1][0] * n1.transpose() + n1 * m[1][0].transpose())
                - n1 * m[0][0].transpose() / edge[0][0].squaredNorm()
                + cos[1][1] / (h[1][1] * h[1][1])
                      * (m[1][1] * n2.transpose() + n2 * m[1][1].transpose())
                - n2 * m[0][1].transpose() / edge[0][0].squaredNorm();
            H.block<3, 3>(2 * 3, 2 * 3) =
                cos[0][0] / (h[2][0] * h[2][0])
                    * (m[2][0] * n1.transpose() + n1 * m[2][0].transpose())
                - n1 * m[0][0].transpose() / edge[0][0].squaredNorm()
                + cos[0][1] / (h[2][1] * h[2][1])
                      * (m[2][1] * n2.transpose() + n2 * m[2][1].transpose())
                - n2 * m[0][1].transpose() / edge[0][0].squaredNorm();
            H.block<3, 3>(1 * 3, 0 * 3) =
                (-1 / (h[0][0] * h[1][0])
                 * (m[1][0] * n1.transpose() - cos[1][0] * n1 * m[0][0].transpose()))
                    .transpose();
            H.block<3, 3>(0 * 3, 1 * 3) =
                -1 / (h[0][0] * h[1][0])
                * (m[1][0] * n1.transpose() - cos[1][0] * n1 * m[0][0].transpose());
            H.block<3, 3>(2 * 3, 0 * 3) =
                (-1 / (h[0][0] * h[2][0])
                 * (m[2][0] * n1.transpose() - cos[0][0] * n1 * m[0][0].transpose()))
                    .transpose();
            H.block<3, 3>(0 * 3, 2 * 3) =
                -1 / (h[0][0] * h[2][0])
                * (m[2][0] * n1.transpose() - cos[0][0] * n1 * m[0][0].transpose());
            H.block<3, 3>(1 * 3, 3 * 3) =
                (-1 / (h[0][1] * h[1][1])
                 * (m[1][1] * n2.transpose() - cos[1][1] * n2 * m[0][1].transpose()))
                    .transpose();
            H.block<3, 3>(3 * 3, 1 * 3) =
                -1 / (h[0][1] * h[1][1])
                * (m[1][1] * n2.transpose() - cos[1][1] * n2 * m[0][1].transpose());
            H.block<3, 3>(2 * 3, 3 * 3) =
                (-1 / (h[0][1] * h[2][1])
                 * (m[2][1] * n2.transpose() - cos[0][1] * n2 * m[0][1].transpose()))
                    .transpose();
            H.block<3, 3>(3 * 3, 2 * 3) =
                -1 / (h[0][1] * h[2][1])
                * (m[2][1] * n2.transpose() - cos[0][1] * n2 * m[0][1].transpose());
            H.block<3, 3>(1 * 3, 2 * 3) =
                1 / (h[1][0] * h[2][0])
                    * (cos[1][0] * m[2][0] * n1.transpose() + cos[0][0] * n1 * m[1][0].transpose())
                + n1 * m[0][0].transpose() / edge[0][0].squaredNorm()
                + 1 / (h[1][1] * h[2][1])
                      * (cos[1][1] * m[2][1] * n2.transpose()
                         + cos[0][1] * n2 * m[1][1].transpose())
                + n2 * m[0][1].transpose() / edge[0][0].squaredNorm();
            H.block<3, 3>(2 * 3, 1 * 3) =
                (1 / (h[1][0] * h[2][0])
                     * (cos[1][0] * m[2][0] * n1.transpose() + cos[0][0] * n1 * m[1][0].transpose())
                 + n1 * m[0][0].transpose() / edge[0][0].squaredNorm()
                 + 1 / (h[1][1] * h[2][1])
                       * (cos[1][1] * m[2][1] * n2.transpose()
                          + cos[0][1] * n2 * m[1][1].transpose())
                 + n2 * m[0][1].transpose() / edge[0][0].squaredNorm())
                    .transpose();
            H.block<3, 3>(0 * 3, 3 * 3) = Matrix3r::Zero();
            H.block<3, 3>(3 * 3, 0 * 3) = Matrix3r::Zero();

            H = coefficient
                * (gradient_theta.reshaped() * gradient_theta.reshaped().transpose()
                   + abs(angle) * sign * H);
            // compute Hess
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int idx1 = 0; idx1 < 4; idx1++)
                        for (int idx2 = 0; idx2 < 4; idx2++) {
                            Hess.coeffRef(
                                3 * hinge_vectex_idx(idx1) + i, 3 * hinge_vectex_idx(idx2) + j) +=
                                H(3 * idx1 + i, 3 * idx2 + j);
                        }
        }
    }

    Hess.makeCompressed();
    return bending_stiffness_ * Hess;
}
} // namespace cipc
