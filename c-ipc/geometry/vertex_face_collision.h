#pragma once

#include <c-ipc/geometry/primative_collision_base.h>
#include <c-ipc/geometry/distance.h>

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
    // Matrix3x4r distance_grad(const Matrix3x4r &position) const override;
    // Matrix12r distance_hess(const Matrix3x4r &position) const override;

    // TO DO: sorting???
};
} // namespace cipc
