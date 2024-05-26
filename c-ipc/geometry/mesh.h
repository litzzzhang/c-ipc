#pragma once

#include "c-ipc/solver/eigen.h"

namespace cipc {

class Mesh {
  public:
    Matrix3Xr vertices, vert_normals, face_normals;
    Matrix3Xi indices;

    integer vertices_num, elem_num;

    Mesh() = default;

    inline void clear() {
        vertices.setZero();
        vert_normals.setZero();
        face_normals.setZero();
        indices.setZero();
    }

    void ComputeFaceNormals(bool normalized = true) {
        face_normals.setZero(elem_num, 3);
        // parallel for
        oneapi::tbb::parallel_for(0, elem_num, [&](integer i) {
            face_normals.row(i) = compute_face_normal(i, vertices, indices);
        });
    }

  private:
    Vector3r compute_face_normal(
        integer face_idx, Matrix3Xr &vertices, Matrix3Xi &indices, bool normalized = true) const {
        cipc_assert(face_idx >= 0 && face_idx < indices.cols(), "Invalid face index");

        Vector3r v0 = vertices.row(indices.row(face_idx)(0));
        Vector3r v1 = vertices.row(indices.row(face_idx)(1));
        Vector3r v2 = vertices.row(indices.row(face_idx)(2));
        Vector3r n = (v1 - v0).cross(v2 - v0);
        return normalized ? n.normalized() : n;
    }
};
} // namespace cipc
