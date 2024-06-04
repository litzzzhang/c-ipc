#pragma once

#include <c-ipc/geometry/primative_collision_base.h>
#include <c-ipc/geometry/distance.h>
#include <c-ipc/geometry/accd.h>

namespace cipc {
class VertexFaceCollision : public PrimativeCollision {
  public:
    integer vertex_idx, face_idx;
    VertexFaceCollision(integer face_id, integer vertex_id)
        : vertex_idx(vertex_id), face_idx(face_id) {}

    Vector4i vertices_idx(const Matrix2Xi &edges, const Matrix3Xi &faces) const override {
        return Vector4i(vertex_idx, faces(0, face_idx), faces(1, face_idx), faces(2, face_idx));
    }

    real distance(const Matrix3x4r &position) const override {
        return point_triangle_distance(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }
    Matrix3x4r distance_grad(const Matrix3x4r &position) const override {
        return point_triangle_distance_gradient(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }
    Matrix12r distance_hess(const Matrix3x4r &position) const override {
        return point_triangle_distance_hessian(
            position.col(0), position.col(1), position.col(2), position.col(3),
            PointTriangleDistType::AUTO);
    }

    bool is_mollified() const override{ return false; }
    double mollifier(const Matrix3x4r &position, double eps) const override { return 1.0; }

    Matrix3x4r mollifier_grad(const Matrix3x4r &position, double eps) const override {
        return Matrix3x4r::Zero();
    }

    Matrix12r mollifier_hess(const Matrix3x4r &position, double eps) const override {
        return Matrix12r::Zero();
    }

    double compute_accd_timestep(
        const Matrix3x4r &pos0, const Matrix3x4r &pos1, const double thickness,
        const double t_ccd_fullstep) const override {
        double t_ccd_addictive = 0.0;
        if (!vertex_face_accd(pos0, pos1, thickness, t_ccd_fullstep, t_ccd_addictive, 0.9)) {
            return t_ccd_fullstep;
        }
        return t_ccd_addictive;
    }

    // TO DO: sorting???
};
} // namespace cipc