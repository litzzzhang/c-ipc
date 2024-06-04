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

    // utils for computing barrier energy/grad/hessian
    virtual real distance(const Matrix3x4r &position) const = 0;
    virtual Matrix3x4r distance_grad(const Matrix3x4r &position) const = 0;
    virtual Matrix12r distance_hess(const Matrix3x4r &position) const = 0;

    // mollifier to enhance continuity
    virtual bool is_mollified() const = 0;
    virtual double mollifier(const Matrix3x4r& position, double eps) const = 0;
    virtual Matrix3x4r mollifier_grad(const Matrix3x4r& position, double eps) const = 0;
    virtual Matrix12r mollifier_hess(const Matrix3x4r& position, double eps) const = 0;

    virtual double compute_accd_timestep(
        const Matrix3x4r &pos0, const Matrix3x4r &pos1, const double thickness,
        const double t_ccd_fullstep) const = 0;
};
} // namespace cipc
