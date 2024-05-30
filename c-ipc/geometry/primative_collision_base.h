#pragma once

#include <c-ipc/solver/eigen.h>

namespace cipc {
class PrimativeCollision {
  public:
    virtual ~PrimativeCollision() = default;

    // In 3D, face-vertex and edge-edge are both 4-vertices collision stencil
    integer num_vertices() const { return 4; }

    virtual Vector4i vertices_idx(const Matrix2Xi &edges, const Matrix3Xi &faces) const = 0;

    Matrix3x4r
    vertices(const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces) const {
        Matrix3x4r vertices_;
        Vector4i vertices_idx = this->vertices_idx(edges, faces);
        for (integer i = 0; i < 4; i++) { vertices_.col(i) = vertices.col(vertices_idx(i)); }
        return vertices_;
    }
    // TO DO: dof function?? in narrow phase

    // utils for computing barrier energy/grad/hessian
    virtual real distance(const Matrix3x4r &position) const = 0;
    // virtual Matrix3x4r distance_grad(const Matrix3x4r &position) const = 0;
    // virtual Matrix12r distance_hess(const Matrix3x4r &position) const = 0;
};
} // namespace cipc
