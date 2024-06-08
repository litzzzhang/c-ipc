#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/geometry/bvh.h>
#include <c-ipc/geometry/vertex_face_collision.h>
#include <c-ipc/geometry/edge_edge_collision.h>

namespace cipc {

const integer max_iteration = 100000;
class ConstrainSet {
  public:
    ConstrainSet() = default;
    void build(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        const double inflation_raduis);
    void build(
        const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double inflation_raduis);

    double compute_accd_timestep(
        const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double min_distance);

    void remove_adjacent_edge_edge(const Matrix2Xi &edges);
    void remove_adjacent_vertex_face(const Matrix3Xi &faces);
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
    remove_adjacent_edge_edge(edges);
    remove_adjacent_vertex_face(faces);
}

inline void ConstrainSet::build(
    const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
    const Matrix3Xi &faces, const double inflation_raduis) {
    clear();
    std::shared_ptr<BroadPhaseBVH> board_phase = std::make_shared<BroadPhaseBVH>();
    board_phase->build(vertices0, vertices1, edges, faces, inflation_raduis);
    board_phase->detect_edge_edge_collision(edge_edge_set);
    board_phase->detect_vertex_face_collision(vertex_face_set);
    remove_adjacent_edge_edge(edges);
    remove_adjacent_vertex_face(faces);
}

inline double ConstrainSet::compute_accd_timestep(
    const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix2Xi &edges,
    const Matrix3Xi &faces, const double min_distance) {
    build(vertices0, vertices1, edges, faces, min_distance / 2);

    double time_of_impact = 1.0;
    // TO DO: parallel
    for (integer i = 0; i < size(); i++) {
        const PrimativeCollision &collision = (*this)[i];
        const Vector4i idx = collision.vertices_idx(edges, faces);
        const Matrix3x4r pos0 = vertices0(Eigen::all, idx);
        const Matrix3x4r pos1 = vertices1(Eigen::all, idx);
        time_of_impact = std::min(
            time_of_impact,
            collision.compute_accd_timestep(pos0, pos1, min_distance, time_of_impact, max_iteration));
    }
    return time_of_impact;
}

inline void ConstrainSet::remove_adjacent_edge_edge(const Matrix2Xi &edges) {
    std::vector<EdgeEdgeCollision> valid_edge_edge_set;
    for (integer i = 0; i < edge_edge_set.size(); i++) {
        EdgeEdgeCollision collision = edge_edge_set[i];
        Vector2i edge0_vertex_idx = edges.col(collision.edge0_idx);
        Vector2i edge1_vertex_idx = edges.col(collision.edge1_idx);
        integer i0 = edge0_vertex_idx(0), i1 = edge0_vertex_idx(1), i2 = edge1_vertex_idx(0),
                i3 = edge1_vertex_idx(1);
        if (i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3) {
            valid_edge_edge_set.push_back(collision);
        }
    }
    edge_edge_set = valid_edge_edge_set;
}

inline void ConstrainSet::remove_adjacent_vertex_face(const Matrix3Xi &faces) {
    std::vector<VertexFaceCollision> valid_vertex_face_set;
    for (integer i = 0; i < vertex_face_set.size(); i++) {
        VertexFaceCollision collision = vertex_face_set[i];
        integer v = collision.vertex_idx; 
        Vector3i face_idx = faces.col(collision.face_idx);
        integer t0 = face_idx(0), t1 = face_idx(1), t2 = face_idx(2);
        if (v != t0 && v != t1 && v != t2) {
            valid_vertex_face_set.push_back(collision);
        }
    }
    vertex_face_set = valid_vertex_face_set;
}

inline void ConstrainSet::clear() {
    vertex_face_set.clear();
    edge_edge_set.clear();
}

inline bool ConstrainSet::isempty() const {
    return vertex_face_set.empty() && edge_edge_set.empty();
}

inline size_t ConstrainSet::size() const { return vertex_face_set.size() + edge_edge_set.size(); }

inline PrimativeCollision &ConstrainSet::operator[](size_t i) {
    cipc_assert(i < size() && i >= 0, "out of range");
    if (i < vertex_face_set.size()) { return vertex_face_set[i]; }
    i -= vertex_face_set.size();
    cipc_assert(i < size() && i >= 0, "out of range");
    return edge_edge_set[i];
}

inline const PrimativeCollision &ConstrainSet::operator[](size_t i) const {

    cipc_assert(i < size() && i >= 0, "out of range");
    if (i < vertex_face_set.size()) { return vertex_face_set[i]; }
    i -= vertex_face_set.size();
    cipc_assert(i < size() && i >= 0, "out of range");
    return edge_edge_set[i];
}

} // namespace cipc
