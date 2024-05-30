#pragma once

#include <c-ipc/geometry/primative_collision_base.h>
#include <c-ipc/geometry/distance.h>
namespace cipc {
class EdgeEdgeCollision : public PrimativeCollision {
  public:
    integer edge0_idx, edge1_idx;

    EdgeEdgeCollision(integer e0_idx, integer e1_idx) : edge0_idx(e0_idx), edge1_idx(e1_idx) {
    }


    Vector4i vertices_idx(const Matrix2Xi &edges, const Matrix3Xi &faces) const {
        return Vector4i(
            edges(0, edge0_idx), edges(1, edge0_idx), edges(0, edge1_idx), edges(1, edge1_idx));
    }

    real distance(const Matrix3x4r &position) const override {
        return edge_edge_distance(
            position.col(0), position.col(1), position.col(2), position.col(3), EdgeEdgeDistType::AUTO);
    }
};
} // namespace cipc
