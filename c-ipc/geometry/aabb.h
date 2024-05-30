#pragma once
#if _MSC_VER
#  pragma fenv_access(on)
#else
#  pragma STDC FENV_ACCESS ON
#endif

#include "c-ipc/solver/eigen.h"

namespace cipc {
class AABB {
  public:
    Vector3r min, max;
    std::array<integer, 3> vertex_ids;
    AABB() = default;
    AABB(const Vector3r &min_, const Vector3r &max_) : min(min_), max(max_) {}
    AABB(const AABB &aabb1, const AABB &aabb2) {
        this->min = aabb1.min.cwiseMin(aabb2.min);
        this->max = aabb1.max.cwiseMax(aabb2.max);
    }
    AABB(const AABB &aabb1, const AABB &aabb2, const AABB &aabb3) {
        this->min = aabb1.min.cwiseMin(aabb2.min).cwiseMin(aabb3.min);
        this->max = aabb1.max.cwiseMax(aabb2.max).cwiseMax(aabb3.max);
    }

    void clear() {
        min = Vector3r(1.0, 1.0, 1.0);
        max = Vector3r(-1.0, -1.0, -1.0);
    }

    bool isempty() const { return min(0) > max(0) || min(1) > max(1) || min(2) > max(2); }
    bool is_intersect(const AABB &other) const {
        bool condition1 = this->min.cwiseMin(other.max).cwiseEqual(this->min).all();
        bool condition2 = other.min.cwiseMin(this->max).cwiseEqual(other.min).all();
        return condition1 && condition2;
    };
    void conservative_inflation(const double inflation_radius) {
        const integer curr_round_mode = std::fegetround();
        std::fesetround(FE_DOWNWARD);
        cipc_assert(std::fegetround() == FE_DOWNWARD, "float round not set");
        min -= inflation_radius * Vector3r::Constant(1.0);
        std::fesetround(FE_UPWARD);
        cipc_assert(std::fegetround() == FE_UPWARD, "float round not set");
        max += inflation_radius * Vector3r::Constant(1.0);
        std::fesetround(curr_round_mode);
    }

    // static version
    static void build_vertex_boxes(
        const Matrix3Xr &vertices, std::vector<AABB> &vertex_boxes, const double inflation_radius) {
        integer vertex_num = static_cast<integer>(vertices.cols());
        vertex_boxes.resize(vertex_num);
        for (integer i = 0; i < vertex_num; i++) {
            AABB v_aabb(vertices.col(i), vertices.col(i));
            v_aabb.conservative_inflation(inflation_radius);
            v_aabb.vertex_ids = {i, -1, -1};
            vertex_boxes[i] = v_aabb;
        }
    }

    // dynamic version for ccd
    static void build_vertex_boxes(
        const Matrix3Xr &x0, const Matrix3Xr &x1, std::vector<AABB> &vertex_boxes,
        const double inflation_radius) {
        integer vertex_num = static_cast<integer>(x0.cols());
        vertex_boxes.resize(vertex_num);
        for (integer i = 0; i < vertex_num; i++) {
            AABB aabb0(x0.col(i), x0.col(i));
            AABB aabb1(x1.col(i), x1.col(i));
            aabb0.conservative_inflation(inflation_radius);
            aabb1.conservative_inflation(inflation_radius);
            vertex_boxes[i] = AABB(aabb0, aabb1);
            vertex_boxes[i].vertex_ids = {i, -1, -1};
        }
    }

    static void build_edge_boxes(
        const std::vector<AABB> &vertex_boxes, const Matrix2Xi &edges,
        std::vector<AABB> &edge_boxes) {
        integer edge_num = static_cast<integer>(edges.cols());
        edge_boxes.resize(edge_num);
        for (integer i = 0; i < edge_num; i++) {
            integer e0 = edges(0, i), e1 = edges(1,i); 
            AABB aabb(vertex_boxes[edges(0, i)], vertex_boxes[edges(1, i)]);
            aabb.vertex_ids = {edges(0, i), edges(1, i), -1};
            edge_boxes[i] = aabb;
        }
    }

    static void build_face_boxes(
        const std::vector<AABB> &vertex_boxes, const Matrix3Xi &faces,
        std::vector<AABB> &face_boxes) {
        integer face_num = static_cast<integer>(faces.cols());
        face_boxes.resize(face_num);
        for (integer i = 0; i < face_num; i++) {
            AABB aabb(
                vertex_boxes[faces(0, i)], vertex_boxes[faces(1, i)], vertex_boxes[faces(2, i)]);
            aabb.vertex_ids = {faces(0, i), faces(1, i), faces(2, i)};
            face_boxes[i] = aabb;
        }
    }
};

} // namespace cipc
