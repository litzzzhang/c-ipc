#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/geometry/mesh.h>
#include <c-ipc/backend/utils.h>
#include <c-ipc/backend/parallel.h>
#include <c-ipc/backend/stl_port.h>

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

    static const real
    ComputeDihedralAngleFromNonUnitNormal(const Vector3r &normal, const Vector3r &other_normal) {
        const real sin_angle = normal.cross(other_normal).norm();
        const real cos_angle = normal.dot(other_normal);
        const real angle = std::atan2(sin_angle, cos_angle);
        return angle;
    }

    static const real ComputeHeight(const Vector3r &v0, const Vector3r &v1, const Vector3r &v2) {
        return ((v0 - v1).cross(v2 - v1)).norm() / (v2 - v1).norm();
    }

    static const Vector3r ComputeNormal(const Matrix3r &vertices) {
        // This is the normal direction vector that is not normalized.
        // You may assume that in this homework the area of a triangle does not shrink below 1e-5,
        // therefore the normal direction (a x b)/||a x b|| does not suffer from numerical issues.
        return (vertices.col(1) - vertices.col(0)).cross(vertices.col(2) - vertices.col(1));
    }
    static const Vector4i FindVertexIdx(const Vector3i &mesh0, const Vector3i &mesh1) {
        integer v0, v1, v2, v3;
        for (int i = 0; i < 3; i++) {
            bool isInMesh0 = false;
            for (int j = 0; j < 3; j++) {
                if (mesh1(i) == mesh0(j)) isInMesh0 = true;
            }
            if (!isInMesh0) v0 = mesh1(i);
        }
        for (int i = 0; i < 3; i++) {
            if (mesh1(i) != v0) v1 = mesh1(i);
        }
        for (int i = 0; i < 3; i++) {
            if (mesh1(i) != v0 && mesh1(i) != v1) v2 = mesh1(i);
        }
        for (int i = 0; i < 3; i++) {
            if (mesh0(i) != v1 && mesh0(i) != v2) v3 = mesh0(i);
        }
        return Vector4i(v0, v1, v2, v3);
    }

    const real ComputeBendingEnergy(
        const Matrix3Xr &position, const Matrix3Xi &indices,
        const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;

    const Matrix3Xr ComputeBendingForce(
        const Matrix3Xr &position, const Matrix3Xi &indices,
        const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;

    const SparseMatrixXr ComputeBendingHessian(
        const Matrix3Xr &position, const Matrix3Xi &indices,
        const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
        const VectorXr &rest_area) const;
};

inline const real DihedralBending::ComputeBendingEnergy(
    const Matrix3Xr &position, const Matrix3Xi &indices,
    const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer element_num = static_cast<integer>(indices.cols());
    real energy = oneapi::tbb::parallel_deterministic_reduce(
        oneapi::tbb::blocked_range<integer>(0, element_num), 0.0,
        [&](oneapi::tbb::blocked_range<integer> r, real local) {
            for (integer e = r.begin(); e < r.end(); e++) {
                const Vector3r normal = ComputeNormal(position(Eigen::all, indices.col(e)));
                for (integer i = 0; i < 3; i++) {
                    const TriangleEdgeInfo &info = edge_info[e][i];
                    if (info.other_triangle == -1) { continue; }
                    const Vector3r other_normal =
                        ComputeNormal(position(Eigen::all, indices.col(info.other_triangle)));
                    const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
                    const real rest_edge_length = info.edge_length;
                    const real A = rest_area(e) + rest_area(info.other_triangle);
                    local += 3.0 * angle * angle * rest_edge_length * rest_edge_length / A;
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
    const Matrix3Xr &position, const Matrix3Xi &indices,
    const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer vertex_num = static_cast<integer>(position.cols());
    const integer element_num = static_cast<integer>(indices.cols());
    Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);

    std::vector<Matrix3x4r> gradient_per_hinge;
    gradient_per_hinge.assign(3 * element_num, Matrix3x4r::Zero());
    std::vector<Vector4i> index_per_hinge;
    index_per_hinge.assign(3 * element_num, Vector4i::Zero());

    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        const Matrix3r &this_vertices = position(Eigen::all, indices.col(e));
        const Vector3r normal = ComputeNormal(this_vertices);
        // iterate each edge
        for (integer i = 0; i < 3; i++) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) { continue; }
            const Matrix3r &other_vertices = position(Eigen::all, indices.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vertices);
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_edge_length = info.edge_length;
            const real diamond_area = (rest_area(e) + rest_area(info.other_triangle)) / 3;

            Vector4i vertexIdx = FindVertexIdx(indices.col(e), indices.col(info.other_triangle));
            const Vector3r v0 = position.col(vertexIdx(0));
            const Vector3r v1 = position.col(vertexIdx(1));
            const Vector3r v2 = position.col(vertexIdx(2));
            const Vector3r v3 = position.col(vertexIdx(3));
            // face normal
            Vector3r n1 = ((v2 - v1).cross(v0 - v1)).normalized();
            Vector3r n2 = ((v3 - v1).cross(v2 - v1)).normalized();
            // height
            real h[3][2];
            h[0][0] = ComputeHeight(v0, v1, v2);
            h[1][0] = ComputeHeight(v1, v0, v2);
            h[2][0] = ComputeHeight(v2, v0, v1);
            h[0][1] = ComputeHeight(v3, v1, v2);
            h[1][1] = ComputeHeight(v1, v3, v2);
            h[2][1] = ComputeHeight(v2, v1, v3);
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
                2 * rest_edge_length * rest_edge_length / diamond_area * abs(angle) * sign;

            index_per_hinge[3 * e + i] = vertexIdx;
            Matrix3x4r grad = Matrix3x4r::Zero();
            grad.col(0) = -coefficient_signed / h[0][0] * n1;
            grad.col(1) =
                coefficient_signed * (cos[1][0] / h[1][0] * n1 + cos[1][1] / h[1][1] * n2);
            grad.col(2) =
                coefficient_signed * (cos[0][0] / h[2][0] * n1 + cos[0][1] / h[2][1] * n2);
            grad.col(3) = -coefficient_signed / h[0][1] * n2;
            gradient_per_hinge[3 * e + i] = grad;

            // // compute grad theta
            // gradient.col(vertexIdx(0)) += -coefficient_signed / h[0][0] * n1;
            // gradient.col(vertexIdx(1)) +=
            //     coefficient_signed * (cos[1][0] / h[1][0] * n1 + cos[1][1] / h[1][1] * n2);
            // gradient.col(vertexIdx(2)) +=
            //     coefficient_signed * (cos[0][0] / h[2][0] * n1 + cos[0][1] / h[2][1] * n2);
            // gradient.col(vertexIdx(3)) += -coefficient_signed / h[0][1] * n2;
        }

        // cipc_assert(!gradient_per_element[e].hasNaN(), "element {} gradient has nan", e);
    });

    for (integer h = 0; h < 3 * element_num; h++) {
        Vector4i vertexIdx = index_per_hinge[h];
        Matrix3x4r grad = gradient_per_hinge[h];
        gradient.col(vertexIdx(0)) += grad.col(0);
        gradient.col(vertexIdx(1)) += grad.col(1);
        gradient.col(vertexIdx(2)) += grad.col(2);
        gradient.col(vertexIdx(3)) += grad.col(3);
    }
    return -bending_stiffness_ * gradient;
}

inline const SparseMatrixXr DihedralBending::ComputeBendingHessian(
    const Matrix3Xr &position, const Matrix3Xi &indices,
    const std::vector<std::array<TriangleEdgeInfo, 3>> &edge_info,
    const VectorXr &rest_area) const {

    const integer hess_size = static_cast<integer>(position.cols()) * 3;
    SparseMatrixXr Hess(hess_size, hess_size);
    Hess.setZero();
    const integer element_num = static_cast<integer>(indices.cols());
    const Matrix3Xi &elements_ = indices;
    // std::vector<Matrix9r> hess_per_element;
    // hess_per_element.assign(element_num, Matrix9r::Zero());
    std::vector<Matrix12r> hessian_per_hinge;
    hessian_per_hinge.assign(3 * element_num, Matrix12r::Zero());
    std::vector<Vector4i> index_per_hinge;
    index_per_hinge.assign(3 * element_num, Vector4i::Zero());
    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        // for (integer e = 0; e < element_num; e++) {
        const Matrix3r &this_vertices = position(Eigen::all, indices.col(e));
        const Vector3r normal = ComputeNormal(this_vertices);
        for (integer i = 0; i < 3; ++i) {
            const TriangleEdgeInfo &info = edge_info[e][i];
            if (info.other_triangle == -1) continue;
            const Matrix3r &other_vertices = position(Eigen::all, indices.col(info.other_triangle));
            const Vector3r other_normal = ComputeNormal(other_vertices);
            const real angle = ComputeDihedralAngleFromNonUnitNormal(normal, other_normal);
            const real rest_shape_edge_length = info.edge_length;
            const real diamond_area = (rest_area(e) + rest_area(info.other_triangle)) / 3;

            Vector4i vertexIdx =
                FindVertexIdx(elements_.col(e), elements_.col(info.other_triangle));
            const Vector3r v0 = position.col(vertexIdx(0));
            const Vector3r v1 = position.col(vertexIdx(1));
            const Vector3r v2 = position.col(vertexIdx(2));
            const Vector3r v3 = position.col(vertexIdx(3));
            // face normal
            Vector3r n1 = ((v2 - v1).cross(v0 - v1)).normalized();
            Vector3r n2 = ((v3 - v1).cross(v2 - v1)).normalized();
            // height
            real h[3][2];
            h[0][0] = ComputeHeight(v0, v1, v2);
            h[1][0] = ComputeHeight(v1, v0, v2);
            h[2][0] = ComputeHeight(v2, v0, v1);
            h[0][1] = ComputeHeight(v3, v1, v2);
            h[1][1] = ComputeHeight(v1, v3, v2);
            h[2][1] = ComputeHeight(v2, v1, v3);
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
            cos[0][0] = (edge[0][0].normalized()).dot(edge[2][0].normalized());
            cos[1][0] = -(edge[0][0].normalized()).dot(edge[1][0].normalized());
            cos[0][1] = (edge[0][0].normalized()).dot(edge[2][1].normalized());
            cos[1][1] = -(edge[0][0].normalized()).dot(edge[1][1].normalized());
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
            const real coefficient =
                2 * rest_shape_edge_length * rest_shape_edge_length / diamond_area;
            const real coefficient_signed = 2 * rest_shape_edge_length * rest_shape_edge_length
                                            / diamond_area * abs(angle) * sign;
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
            H = project_to_spd(H);

            hessian_per_hinge[3 * e + i] = H;
            index_per_hinge[3 * e + i] = vertexIdx;
            // // compute Hess
            // for (int i = 0; i < 3; i++)
            //     for (int j = 0; j < 3; j++)
            //         for (int idx1 = 0; idx1 < 4; idx1++)
            //             for (int idx2 = 0; idx2 < 4; idx2++) {
            //                 Hess.coeffRef(3 * vertexIdx(idx1) + i, 3 * vertexIdx(idx2) + j) +=
            //                     H(3 * idx1 + i, 3 * idx2 + j);
            //             }
        }
    });
    for (int h = 0; h < 3 * element_num; h++) {
        Vector4i vertexIdx = index_per_hinge[h];
        Matrix12r H = hessian_per_hinge[h];
        // compute Hess
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int idx1 = 0; idx1 < 4; idx1++)
                    for (int idx2 = 0; idx2 < 4; idx2++) {
                        Hess.coeffRef(3 * vertexIdx(idx1) + i, 3 * vertexIdx(idx2) + j) +=
                            H(3 * idx1 + i, 3 * idx2 + j);
                    }
    }
    // for (int e = 0; e < element_num; e++) {
    //     const Matrix9r &hess = hess_per_element[e];
    //     for (int v1 = 0; v1 < 3; v1++) {
    //         for (int v2 = 0; v2 < 3; v2++) {
    //             for (int i = 0; i < 3; i++) {
    //                 for (int j = 0; j < 3; j++) {
    //                     integer idx1 = 3 * elements_(v1, e) + i;
    //                     integer idx2 = 3 * elements_(v2, e) + j;
    //                     cipc_assert(3 * v1 + i < 9 && 3 * v2 + j < 9, "out of range");
    //                     Hess.coeffRef(idx1, idx2) += hess(3 * v1 + i, 3 * v2 + j);
    //                 }
    //             }
    //         }
    //     }
    // }

    Hess.makeCompressed();
    return bending_stiffness_ * Hess;
}
} // namespace cipc
