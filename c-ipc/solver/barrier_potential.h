#pragma once

#include <c-ipc/solver/barrier.h>
#include <c-ipc/geometry/constrain_set.h>

namespace cipc {

class BarrierPotential {
  public:
    BarrierPotential() = default;
    double dhat = 1e-3;
    double closest_distance = 1.0;
    bool is_built = false;
    std::vector<EdgeEdgeCollision> edge_edge_collisions;
    std::vector<VertexFaceCollision> vertex_face_collisions;

    void build(
        const Matrix3Xr &vertices, const Matrix3Xr &rest_vertices, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double dhat, const double dmin, const ConstrainSet &c) {
        clear();
        // exclude pairs whose distance are greater than offset squared in narrow phase
        const double offsetsqured = (dmin + dhat) * (dmin + dhat);
        auto is_colliding = [&](double dist_squared) {
            return dist_squared < offsetsqured;
        };

        for (integer i = 0; i < c.edge_edge_set.size(); i++) {
            EdgeEdgeCollision collision = c.edge_edge_set[i];
            Vector4i edge_idx = collision.vertices_idx(edges, faces);
            Matrix3x4r edge_pos = collision.vertices(vertices, edges, faces);
            Matrix3x4r edge_pos_rest = collision.vertices(rest_vertices, edges, faces);
            collision.eps = 1e-3 * (edge_pos_rest.col(0) - edge_pos_rest.col(1)).squaredNorm()
                            * (edge_pos_rest.col(2) - edge_pos_rest.col(3)).squaredNorm();
            double dist = collision.distance(edge_pos);
            cipc_assert(dist > 0, "ee collision distance {} should be positive", dist);
            closest_distance = std::min(dist, closest_distance);
            if (!is_colliding(dist)) { continue; }
            edge_edge_collisions.push_back(collision);
        }

        for (integer i = 0; i < c.vertex_face_set.size(); i++) {
            const VertexFaceCollision collision = c.vertex_face_set[i];
            Vector4i vertex_face_idx = collision.vertices_idx(edges, faces);
            Matrix3x4r vertex_face_pos = collision.vertices(vertices, edges, faces);

            double dist = collision.distance(vertex_face_pos);
            if (dist < 0.0) { int stop = 1; }
            cipc_assert(dist > 0, "vf collision distance should be positive");
            closest_distance = std::min(dist, closest_distance);
            if (!is_colliding(dist)) { continue; }
            vertex_face_collisions.push_back(collision);
        }
        is_built = true;
    }

    // static version
    void build(
        const Matrix3Xr &vertices, const Matrix3Xr &rest_vertices, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double dhat, const double dmin) {
        double inflation_radius = (dhat + dmin) / 2;
        ConstrainSet c;
        c.build(vertices, edges, faces, inflation_radius);
        build(vertices, rest_vertices, edges, faces, dhat, dmin, c);
    }

    // dynamic version
    void build(
        const Matrix3Xr &vertices0, const Matrix3Xr &vertices1, const Matrix3Xr &rest_vertices,
        const Matrix2Xi &edges, const Matrix3Xi &faces, const double dhat, const double dmin) {
        double inflation_radius = (dhat + dmin) / 2;
        ConstrainSet c;
        c.build(vertices0, vertices1, edges, faces, inflation_radius);
        build(vertices1, rest_vertices, edges, faces, dhat, dmin, c);
    }
    size_t size() const { return vertex_face_collisions.size() + edge_edge_collisions.size(); }

    void clear() {
        edge_edge_collisions.clear();
        vertex_face_collisions.clear();
    }

    PrimativeCollision &operator[](size_t i) {
        cipc_assert(i < size() && i >= 0, "out of range");
        if (i < vertex_face_collisions.size()) { return vertex_face_collisions[i]; }
        i -= vertex_face_collisions.size();
        cipc_assert(i < size() && i >= 0, "out of range");
        return vertex_face_collisions[i];
    }

    const PrimativeCollision &operator[](size_t i) const {
        cipc_assert(i < size() && i >= 0, "out of range");
        if (i < vertex_face_collisions.size()) { return vertex_face_collisions[i]; }
        i -= vertex_face_collisions.size();
        cipc_assert(i < size() && i >= 0, "out of range");
        return vertex_face_collisions[i];
    }

    const double ComputeBarrierPotential(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        double ee_energy = 0.0, vf_energy = 0.0;
        integer ee_num = static_cast<integer>(edge_edge_collisions.size());
        integer vf_num = static_cast<integer>(vertex_face_collisions.size());

        ee_energy = oneapi::tbb::parallel_deterministic_reduce(
            oneapi::tbb::blocked_range<integer>(0, ee_num), 0.0,
            [&](oneapi::tbb::blocked_range<integer> r, double local) {
                for (integer i = r.begin(); i < r.end(); i++) {
                    const EdgeEdgeCollision collision = edge_edge_collisions[i];
                    Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
                    double dist = collision.distance(curr_pos);
                    local += collision.mollifier(curr_pos, collision.eps)
                             * cipc_barrier(dist, dhat, dmin);
                }
                return local;
            },
            [](double x, double y) {
                return x + y;
            });
        if (ee_energy > 0.0) { printf("*********** edge edge collision ****************\n"); }
        vf_energy = oneapi::tbb::parallel_deterministic_reduce(
            oneapi::tbb::blocked_range<integer>(0, vf_num), 0.0,
            [&](oneapi::tbb::blocked_range<integer> r, double local) {
                for (integer i = r.begin(); i < r.end(); i++) {
                    const VertexFaceCollision collision = vertex_face_collisions[i];
                    Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
                    double dist = collision.distance(curr_pos);
                    local += cipc_barrier(dist, dhat, dmin);
                }
                return local;
            },
            [](double x, double y) {
                return x + y;
            });
        if (vf_energy > 0.0) { printf("*********** vertex face collision ****************\n"); }
        // for (integer i = 0; i < edge_edge_collisions.size(); i++) {
        //     EdgeEdgeCollision collision = edge_edge_collisions[i];
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     double dist = collision.distance(curr_pos);
        //     ee_energy +=
        //         collision.mollifier(curr_pos, collision.eps) * cipc_barrier(dist, dhat, dmin);
        // }
        // for (integer i = 0; i < vertex_face_collisions.size(); i++) {
        //     VertexFaceCollision collision = vertex_face_collisions[i];
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     double dist = collision.distance(curr_pos);
        //     vf_energy +=
        //         collision.mollifier(curr_pos, collision.eps) * cipc_barrier(dist, dhat, dmin);
        // }
        return ee_energy + vf_energy;
    }

    const Matrix3Xr ComputeBarrierGradient(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        integer vertex_num = static_cast<integer>(vertices.cols());
        Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);

        std::vector<Matrix3x4r> gradient_per_collision(size());
        std::vector<Vector4i> gradient_map(size());

        integer ee_num = static_cast<integer>(edge_edge_collisions.size());
        integer vf_num = static_cast<integer>(vertex_face_collisions.size());
        integer total_collision_num = static_cast<integer>(size());

        oneapi::tbb::parallel_for(0, ee_num, [&](integer i) {
            EdgeEdgeCollision collision = edge_edge_collisions[i];
            gradient_per_collision[i] = Matrix3x4r::Zero();
            gradient_map[i] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const double f = cipc_barrier(d, dhat, dmin);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            const double m = collision.mollifier(curr_pos, collision.eps);
            const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
            gradient_per_collision[i] = f * m_grad + m * f_grad * d_grad;
        });

        oneapi::tbb::parallel_for(0, vf_num, [&](integer i) {
            VertexFaceCollision collision = vertex_face_collisions[i];
            gradient_per_collision[i + ee_num] = Matrix3x4r::Zero();
            gradient_map[i + ee_num] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            gradient_per_collision[i + ee_num] = f_grad * d_grad;
        });

        // for (integer i = 0; i < edge_edge_collisions.size(); i++) {
        //     EdgeEdgeCollision collision = edge_edge_collisions[i];
        //     gradient_per_collision[i] = Matrix3x4r::Zero();
        //     gradient_map[i] = collision.vertices_idx(edges, faces);
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     const double d = collision.distance(curr_pos);
        //     const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
        //     const double f = cipc_barrier(d, dhat, dmin);
        //     const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
        //     const double m = collision.mollifier(curr_pos, collision.eps);
        //     const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
        //     gradient_per_collision[i] = f * m_grad + m * f_grad * d_grad;
        // }

        // integer ee_collision_size = static_cast<integer>(edge_edge_collisions.size());
        // for (integer i = 0; i < vertex_face_collisions.size(); i++) {
        //     VertexFaceCollision collision = vertex_face_collisions[i];
        //     gradient_per_collision[i + ee_collision_size] = Matrix3x4r::Zero();
        //     gradient_map[i + ee_collision_size] = collision.vertices_idx(edges, faces);
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     const double d = collision.distance(curr_pos);
        //     const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
        //     const double f = cipc_barrier(d, dhat, dmin);
        //     const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
        //     const double m = collision.mollifier(curr_pos, collision.eps);
        //     const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
        //     gradient_per_collision[i + ee_collision_size] = f * m_grad + m * f_grad * d_grad;
        // }
        for (integer i = 0; i < size(); i++) {
            Vector4i index = gradient_map[i];
            gradient(Eigen::all, index) += gradient_per_collision[i];
        }
        return gradient;
    }

    const SparseMatrixXr ComputeBarrierHessian(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        integer vertex_num = static_cast<integer>(vertices.cols());
        std::vector<Matrix12r> hessian_per_collision(size());
        std::vector<Vector4i> hessian_map(size());

        integer ee_collision_size = static_cast<integer>(edge_edge_collisions.size());

        integer ee_num = static_cast<integer>(edge_edge_collisions.size());
        integer vf_num = static_cast<integer>(vertex_face_collisions.size());
        integer total_collision_num = static_cast<integer>(size());
        oneapi::tbb::parallel_for(0, ee_num, [&](integer i) {
            EdgeEdgeCollision collision = edge_edge_collisions[i];
            hessian_per_collision[i] = Matrix12r::Zero();
            hessian_map[i] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const Matrix12r d_hess = collision.distance_hess(curr_pos);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            const double f_hess = cipc_barrier_second_derivative(d, dhat, dmin);

            if (!collision.is_mollified()) {
                hessian_per_collision[i] =
                    f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() + f_grad * d_hess;
            } else {
                const double f = cipc_barrier(d, dhat, dmin);
                const double m = collision.mollifier(curr_pos, collision.eps);
                const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
                const Matrix12r m_hess = collision.mollifier_hess(curr_pos, collision.eps);

                hessian_per_collision[i] =
                    f * m_hess
                    + f_grad
                          * (d_grad.reshaped() * m_grad.reshaped().transpose()
                             + m_grad.reshaped() * d_grad.reshaped().transpose())
                    + m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose()
                    + m * f_grad * d_hess;
            }

            hessian_per_collision[i] = project_to_spd(hessian_per_collision[i]);
        });

        oneapi::tbb::parallel_for(0, vf_num, [&](integer i) {
            VertexFaceCollision collision = vertex_face_collisions[i];
            hessian_per_collision[i + ee_num] = Matrix12r::Zero();
            hessian_map[i + ee_num] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const Matrix12r d_hess = collision.distance_hess(curr_pos);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            const double f_hess = cipc_barrier_second_derivative(d, dhat, dmin);

            if (!collision.is_mollified()) {
                hessian_per_collision[i + ee_num] =
                    f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() + f_grad * d_hess;
            } else {
                const double f = cipc_barrier(d, dhat, dmin);
                const double m = collision.mollifier(curr_pos, collision.eps);
                const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
                const Matrix12r m_hess = collision.mollifier_hess(curr_pos, collision.eps);
                const Matrix12r grad_f_grad_m =
                    f_grad * d_grad.reshaped() * m_grad.reshaped().transpose();

                hessian_per_collision[i + ee_num] =
                    f * m_hess + grad_f_grad_m + grad_f_grad_m.transpose()
                    + m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose()
                    + m * f_grad * d_hess;
            }

            hessian_per_collision[i + ee_num] = project_to_spd(hessian_per_collision[i + ee_num]);
        });
        // for (integer i = 0; i < ee_collision_size; i++) {
        //     EdgeEdgeCollision collision = edge_edge_collisions[i];
        //     hessian_per_collision[i] = Matrix12r::Zero();
        //     hessian_map[i] = collision.vertices_idx(edges, faces);
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     const double d = collision.distance(curr_pos);
        //     const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
        //     const Matrix12r d_hess = collision.distance_hess(curr_pos);
        //     const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
        //     const double f_hess = cipc_barrier_second_derivative(d, dhat, dmin);

        //     if (!collision.is_mollified()) {
        //         hessian_per_collision[i] =
        //             f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() + f_grad * d_hess;
        //     } else {
        //         const double f = cipc_barrier(d, dhat, dmin);
        //         const double m = collision.mollifier(curr_pos, collision.eps);
        //         const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
        //         const Matrix12r m_hess = collision.mollifier_hess(curr_pos, collision.eps);

        //         hessian_per_collision[i] =
        //             f * m_hess
        //             + f_grad
        //                   * (d_grad.reshaped() * m_grad.reshaped().transpose()
        //                      + m_grad.reshaped() * d_grad.reshaped().transpose())
        //             + m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose()
        //             + m * f_grad * d_hess;
        //     }

        //     hessian_per_collision[i] = project_to_spd(hessian_per_collision[i]);
        // }

        // for (integer i = 0; i < vertex_face_collisions.size(); i++) {
        //     VertexFaceCollision collision = vertex_face_collisions[i];
        //     hessian_per_collision[i + ee_collision_size] = Matrix12r::Zero();
        //     hessian_map[i + ee_collision_size] = collision.vertices_idx(edges, faces);
        //     Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
        //     const double d = collision.distance(curr_pos);
        //     const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
        //     const Matrix12r d_hess = collision.distance_hess(curr_pos);
        //     const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
        //     const double f_hess = cipc_barrier_second_derivative(d, dhat, dmin);

        //     if (!collision.is_mollified()) {
        //         hessian_per_collision[i + ee_collision_size] =
        //             f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() + f_grad * d_hess;
        //     } else {
        //         const double f = cipc_barrier(d, dhat, dmin);
        //         const double m = collision.mollifier(curr_pos, collision.eps);
        //         const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
        //         const Matrix12r m_hess = collision.mollifier_hess(curr_pos, collision.eps);

        //         hessian_per_collision[i + ee_collision_size] =
        //             f * m_hess
        //             + f_grad
        //                   * (d_grad.reshaped() * m_grad.reshaped().transpose()
        //                      + m_grad.reshaped() * d_grad.reshaped().transpose())
        //             + m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose()
        //             + m * f_grad * d_hess;
        //     }

        //     hessian_per_collision[i + ee_collision_size] = project_to_spd(hessian_per_collision[i
        //     + ee_collision_size]);
        // }

        const integer dim = 3;
        std::vector<Eigen::Triplet<real>> barrier_hessian_nonzeros;
        SparseMatrixXr hessian(3 * vertex_num, 3 * vertex_num);
        hessian.setZero();
        for (integer c = 0; c < size(); c++) {
            const Vector4i index = hessian_map[c];
            for (integer i = 0; i < 4; i++)
                for (integer j = 0; j < 4; j++)
                    for (integer di = 0; di < dim; di++)
                        for (integer dj = 0; dj < dim; dj++) {
                            const integer row_idx = index(i) * dim + di;
                            const integer col_idx = index(j) * dim + dj;
                            hessian.coeffRef(row_idx, col_idx) +=
                                hessian_per_collision[c](i * dim + di, j * dim + dj);
                        }
        }

        hessian.makeCompressed();
        return hessian;
    }

    double accd(
        const Matrix3Xr &rest_vertices, const Matrix3Xr &vertices0, const Matrix3Xr &vertices1,
        const Matrix2Xi &edges, const Matrix3Xi &faces, const double dmin, const double dhat) {

        double inflation_radius = 0.5 * (dmin + dhat);
        // if (!is_built) { build(vertices0, vertices1, rest_vertices, edges, faces, dhat, dmin); }
        // double time_of_impact = 1.0;
        // for (integer i = 0; i < size(); i++) {
        //     const PrimativeCollision &collision = (*this)[i];
        //     const Vector4i idx = collision.vertices_idx(edges, faces);
        //     const Matrix3x4r pos0 = vertices0(Eigen::all, idx);
        //     const Matrix3x4r pos1 = vertices1(Eigen::all, idx);
        //     time_of_impact = std::min(
        //         time_of_impact, collision.compute_accd_timestep(pos0, pos1, dmin,
        //         time_of_impact));
        // }
        ConstrainSet c;
        c.build(vertices0, vertices1, edges, faces, inflation_radius);
        return c.compute_accd_timestep(vertices0, vertices1, edges, faces, dmin);
    }
};
} // namespace cipc
