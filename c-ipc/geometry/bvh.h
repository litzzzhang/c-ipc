#pragma once

#include <c-ipc/solver/eigen.h>
#include <c-ipc/backend/parallel.h>
#include <c-ipc/geometry/aabb.h>
#include <c-ipc/geometry/bvh_base.h>
#include <c-ipc/geometry/edge_edge_collision.h>
#include <c-ipc/geometry/vertex_face_collision.h>

namespace cipc {
class BroadPhaseBVH {
  public:
    BroadPhaseBVH() = default;

    void init_bvh(const std::vector<AABB> &boxes, SimpleBVH::BVH &bvh);

    // static version
    void build(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        const double inflation_radius = 0.0);
    // dynamic version
    void build(
        const Matrix3Xr &x0, const Matrix3Xr &x1, const Matrix2Xi &edges, const Matrix3Xi &faces,
        const double inflation_radius = 0.0);

    void detect_edge_edge_collision(std::vector<EdgeEdgeCollision> &collisions) const;

    void detect_vertex_face_collision(std::vector<VertexFaceCollision> &collisions) const;

    void clear() {
        vertex_boxes.clear();
        edge_boxes.clear();
        face_boxes.clear();
        vertex_bvh.clear();
        edge_bvh.clear();
        face_bvh.clear();
    }

    std::vector<AABB> vertex_boxes;
    std::vector<AABB> edge_boxes;
    std::vector<AABB> face_boxes;
    SimpleBVH::BVH vertex_bvh;
    SimpleBVH::BVH edge_bvh;
    SimpleBVH::BVH face_bvh;

  private:
    template <typename Primative, bool swap_order = false, bool triangular = false>
    static void detect_collsion(
        const std::vector<AABB> &boxes, const SimpleBVH::BVH &bvh,
        std::vector<Primative> &collision_primatives);
};

inline void BroadPhaseBVH::init_bvh(const std::vector<AABB> &boxes, SimpleBVH::BVH &bvh) {
    if (boxes.size() == 0) { return; }
    std::vector<std::array<Vector3r, 2>> corner_list(boxes.size());
    for (integer i = 0; i < boxes.size(); i++) { corner_list[i] = {boxes[i].min, boxes[i].max}; }
    bvh.init(corner_list);
}
inline void BroadPhaseBVH::build(
    const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
    const double inflation_radius) {
    AABB::build_vertex_boxes(vertices, vertex_boxes, inflation_radius);
    AABB::build_edge_boxes(vertex_boxes, edges, edge_boxes);
    AABB::build_face_boxes(vertex_boxes, faces, face_boxes);
    init_bvh(vertex_boxes, vertex_bvh);
    init_bvh(edge_boxes, edge_bvh);
    init_bvh(face_boxes, face_bvh);
}
inline void BroadPhaseBVH::build(
    const Matrix3Xr &x0, const Matrix3Xr &x1, const Matrix2Xi &edges, const Matrix3Xi &faces,
    const double inflation_radius) {
    AABB::build_vertex_boxes(x0, x1, vertex_boxes, inflation_radius);
    AABB::build_edge_boxes(vertex_boxes, edges, edge_boxes);
    AABB::build_face_boxes(vertex_boxes, faces, face_boxes);
    init_bvh(vertex_boxes, vertex_bvh);
    init_bvh(edge_boxes, edge_bvh);
    init_bvh(face_boxes, face_bvh);
}

inline void
BroadPhaseBVH::detect_edge_edge_collision(std::vector<EdgeEdgeCollision> &collisions) const {
    if (edge_boxes.size() == 0) { return; }
    detect_collsion<EdgeEdgeCollision, false, true>(edge_boxes, edge_bvh, collisions);
}

inline void
BroadPhaseBVH::detect_vertex_face_collision(std::vector<VertexFaceCollision> &collisions) const {
    if (vertex_boxes.size() == 0 || face_boxes.size() == 0) { return; }
    detect_collsion<VertexFaceCollision, true, false>(vertex_boxes, face_bvh, collisions);
}

template <typename Primative, bool swap_order, bool triangular>
inline void BroadPhaseBVH::detect_collsion(
    const std::vector<AABB> &boxes, const SimpleBVH::BVH &bvh,
    std::vector<Primative> &collision_primatives) {

    // oneapi::tbb::concurrent_vector<Primative> collision_primatives_concurrent;

    // oneapi::tbb::parallel_for(0, static_cast<integer>(boxes.size()), [&](integer i) {
    for (integer i = 0; i < static_cast<int>(boxes.size()); i++) {
        std::vector<unsigned int> list;
        bvh.intersect_3D_box(boxes[i].min, boxes[i].max, list);
        for (const unsigned int j : list) {
            int ai = i, bi = j;
            if constexpr (swap_order) { std::swap(ai, bi); }
            if constexpr (triangular) {
                if (ai >= bi) { continue; }
            }
            Primative collision(ai, bi);
            // collision_primatives_concurrent.push_back(collision);
            collision_primatives.push_back(collision);
        }
    }
    // });

    // collision_primatives.reserve(collision_primatives_concurrent.size());
    // for (auto it = collision_primatives_concurrent.begin();
    //      it != collision_primatives_concurrent.end(); ++it) {
    //     collision_primatives.push_back(*it);
    // }
}
} // namespace cipc
