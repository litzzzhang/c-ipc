#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/geometry/bvh.h>
#include <c-ipc/geometry/vertex_face_collision.h>
#include <c-ipc/geometry/edge_edge_collision.h>

namespace cipc {
class ConstrainSet {
  public:
    ConstrainSet() = default;
    void build(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        const double inflation_raduis);
    void build(
        const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double inflation_raduis);
    void clear();
    bool isempty() const;
    size_t size() const;
    PrimativeCollision &operator[](size_t i);
    const PrimativeCollision &operator[](size_t i) const;

    std::vector<VertexFaceCollision> vertex_face_set;
    std::vector<EdgeEdgeCollision> edge_edge_set;
};

inline void ConstrainSet::build(
    const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
    const double inflation_raduis) {
    clear();
    std::shared_ptr<BroadPhaseBVH> board_phase = std::make_shared<BroadPhaseBVH>();
    board_phase->build(vertices, edges, faces, inflation_raduis);
    board_phase->detect_edge_edge_collision(edge_edge_set);
    board_phase->detect_vertex_face_collision(vertex_face_set);
}

inline void ConstrainSet::build(
    const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
    const Matrix3Xi &faces, const double inflation_raduis) {
    clear();
    std::shared_ptr<BroadPhaseBVH> board_phase = std::make_shared<BroadPhaseBVH>();
    board_phase->build(vertices0, vertices1, edges, faces, inflation_raduis);
    board_phase->detect_edge_edge_collision(edge_edge_set);
    board_phase->detect_vertex_face_collision(vertex_face_set);
}

inline void ConstrainSet::clear() {
    vertex_face_set.clear();
    edge_edge_set.clear();
}

inline bool ConstrainSet::isempty() const {
    return vertex_face_set.empty() && edge_edge_set.empty();
}

inline size_t ConstrainSet::size() const {
    return vertex_face_set.size() + edge_edge_set.size();
}

inline PrimativeCollision &ConstrainSet::operator[](size_t i) {
    cipc_assert(i < size() && i >= 0, "out of range");
    if (i < vertex_face_set.size()){
        return vertex_face_set[i];
    }
    i -= vertex_face_set.size();
    cipc_assert(i < size() && i >= 0, "out of range");
    return edge_edge_set[i];
}

inline const PrimativeCollision &ConstrainSet::operator[](size_t i) const {

    cipc_assert(i < size() && i >= 0, "out of range");
    if (i < vertex_face_set.size()){
        return vertex_face_set[i];
    }
    i -= vertex_face_set.size();
    cipc_assert(i < size() && i >= 0, "out of range");
    return edge_edge_set[i];
}

} // namespace cipc
