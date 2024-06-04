#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/backend/stl_port.h>

namespace cipc {
struct edge_pair {
    integer e0, e1;
    edge_pair(integer e0_, integer e1_) : e0(e0_), e1(e1_) {}

    bool operator==(const edge_pair &other) const { return e0 == other.e0 && e1 == other.e1; }
    size_t operator()(const edge_pair &pair) const {
        return std::hash<int>()(pair.e0) + std::hash<int>()(pair.e1);
    }
};

// Custom hash function for edge_pair
struct edge_pair_hash {
    std::size_t operator()(const edge_pair& ep) const {
        // Use std::hash to combine the hashes of a and b
        return std::hash<int>()(ep.e0) ^ (std::hash<int>()(ep.e1) << 1);
    }
};
class Mesh {
  public:
    Matrix3Xr vertices, rest_vertices, vert_normals, face_normals;
    Matrix3Xi indices;
    Matrix2Xi edges;
    bool face_normal_valid = false, vertex_normal_valid = false;

    Mesh() = default;

    Mesh(Matrix3Xr vertices_, Matrix3Xi indices_) : vertices(vertices_), indices(indices_) {
        rest_vertices = vertices;
        vert_normals = vertices;
        face_normals = vertices;
        vert_normals.setZero();
        face_normals.setZero();
        ComputeVertexNormals(true);
    }

    inline void clear() {
        vertices.setZero();
        rest_vertices.setZero();
        vert_normals.setZero();
        face_normals.setZero();
        indices.setZero();
    }

    void ComputeEdgeIndex() {
        integer elem_num = static_cast<integer>(indices.cols());

        std::unordered_map<edge_pair, integer, edge_pair_hash> edge_map;
        for (integer i = 0; i < elem_num; i++) {
            Vector3i local_index = indices.col(i);
            integer v0 = local_index(0), v1 = local_index(1), v2 = local_index(2);
            std::array<integer, 3> local_indices = {v0, v1, v2};
            std::sort(local_indices.begin(), local_indices.end());
            edge_pair e0(local_indices[0], local_indices[1]);
            edge_pair e1(local_indices[0], local_indices[2]);
            edge_pair e2(local_indices[1], local_indices[2]);

            if (edge_map.find(e0) == edge_map.end()) { edge_map.insert(std::make_pair(e0, 1)); }
            if (edge_map.find(e1) == edge_map.end()) { edge_map.insert(std::make_pair(e1, 1)); }
            if (edge_map.find(e2) == edge_map.end()) { edge_map.insert(std::make_pair(e2, 1)); }
        }

        edges = Matrix2Xi::Zero(2, edge_map.size());
        integer i = 0;
        for (auto it = edge_map.begin(); it != edge_map.end(); it++, i++) {
            Vector2i idx(it->first.e0, it->first.e1);
            edges.col(i) = idx;
        }
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
