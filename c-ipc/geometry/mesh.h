#pragma once

#include "c-ipc/solver/eigen.h"

namespace cipc {

class Mesh {
  public:
    Matrix3Xr vertices, vert_normals, face_normals;
    Matrix3Xi indices;
    Matrix2Xi edges;
    bool face_normal_valid = false, vertex_normal_valid = false;

    Mesh() = default;

    Mesh(Matrix3Xr vertices_, Matrix3Xi indices_) : vertices(vertices_), indices(indices_) {
        vert_normals = vertices;
        face_normals = vertices;
        vert_normals.setZero();
        face_normals.setZero();
        ComputeVertexNormals(true);
    }

    inline void clear() {
        vertices.setZero();
        vert_normals.setZero();
        face_normals.setZero();
        indices.setZero();
    }

    void ComputeFaceNormals(bool normalized = true) {
        integer elem_num = static_cast<integer>(indices.cols());
        face_normals.setZero(3, elem_num);
        // parallel for
        oneapi::tbb::parallel_for(0, elem_num, [&](integer i) {
            face_normals.col(i) = compute_face_normal(i, vertices, indices, normalized);
        });
        face_normal_valid = true;
    }

    void ComputeVertexNormals(bool normalized = true) {
        integer vertex_num = static_cast<integer>(vertices.cols());
        integer elem_num = static_cast<integer>(indices.cols());
        vert_normals.setZero(3, vertex_num);
        if (face_normal_valid) {
            oneapi::tbb::parallel_for(0, elem_num, [&](integer i) {
                Vector3r n = face_normals.col(i);
                vert_normals.col(indices(0, i)) += n;
                vert_normals.col(indices(1, i)) += n;
                vert_normals.col(indices(2, i)) += n;
            });
        } else {
            oneapi::tbb::parallel_for(0, elem_num, [&](integer i) {
                Vector3r n = compute_face_normal(i, vertices, indices, normalized);
                vert_normals.col(indices(0, i)) += n;
                vert_normals.col(indices(1, i)) += n;
                vert_normals.col(indices(2, i)) += n;
            });
        }

        if (normalized) {
            integer vertex_num = static_cast<integer>(vertices.cols());
            oneapi::tbb::parallel_for(0, vertex_num, [&](integer i) {
                // eigen has checked zero vec
                vert_normals.col(i) = vert_normals.col(i).normalized();
            });
        }
        vertex_normal_valid = true;
    }

  private:
    Vector3r compute_face_normal(
        integer face_idx, Matrix3Xr &vertices, Matrix3Xi &indices, bool normalized = true) const {
        cipc_assert(face_idx >= 0 && face_idx < indices.cols(), "Invalid face index");

        Vector3r v0 = vertices.col(indices(0, face_idx));
        Vector3r v1 = vertices.col(indices(1, face_idx));
        Vector3r v2 = vertices.col(indices(2, face_idx));
        Vector3r n = (v1 - v0).cross(v2 - v0);
        return normalized ? n.normalized() : n;
    }
};
} // namespace cipc
