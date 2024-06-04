#pragma once

#include <c-ipc/solver/barrier.h>
#include <c-ipc/geometry/constrain_set.h>

namespace cipc {

class BarrierPotential {
  public:
    BarrierPotential() = default;
    double dhat = 1e-4;
    bool is_built = false;
    std::vector<EdgeEdgeCollision> edge_edge_collisions;
    std::vector<VertexFaceCollision> vertex_face_collisions;

    void build(
        const Matrix3Xr &vertices, const Matrix3Xr &rest_vertices, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double dhat, const double dmin, const ConstrainSet &c) {
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
            if (!is_colliding(collision.distance(edge_pos))) { continue; }
            edge_edge_collisions.push_back(collision);
        }

        for (integer i = 0; i < c.vertex_face_set.size(); i++) {
            const VertexFaceCollision collision = c.vertex_face_set[i];
            Vector4i vertex_face_idx = collision.vertices_idx(edges, faces);
            Matrix3x4r vertex_face_pos = collision.vertices(vertices, edges, faces);

            if (!is_colliding(collision.distance(vertex_face_pos))) { continue; }
            vertex_face_collisions.push_back(collision);
        }
        is_built = true;
    }

    void build(
        const Matrix3Xr &vertices, const Matrix3Xr &rest_vertices, const Matrix2Xi &edges,
        const Matrix3Xi &faces, const double dhat, const double dmin) {
        double inflation_radius = (dhat + dmin) / 2;
        ConstrainSet c;
        c.build(vertices, edges, faces, inflation_radius);
        this->build(vertices, rest_vertices, edges, faces, dhat, dmin, c);
    }
    size_t size() const { return vertex_face_collisions.size() + edge_edge_collisions.size(); }

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

    double ComputeBarrierPotential(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        double energy = 0.0;
        integer total_num = static_cast<integer>(size());

        for (integer i = 0; i < edge_edge_collisions.size(); i++) {
            EdgeEdgeCollision collision = edge_edge_collisions[i];
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            double dist = collision.distance(curr_pos);
            energy += collision.mollifier(curr_pos, collision.eps) * cipc_barrier(dist, dhat, dmin);
        }
        for (integer i = 0; i < vertex_face_collisions.size(); i++) {
            VertexFaceCollision collision = vertex_face_collisions[i];
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            double dist = collision.distance(curr_pos);
            energy += collision.mollifier(curr_pos, collision.eps) * cipc_barrier(dist, dhat, dmin);
        }
        return energy;
    }

    Matrix3Xr ComputeBarrierGradient(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        integer vertex_num = static_cast<integer>(vertices.cols());
        Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);

        std::vector<Matrix3x4r> gradient_per_collision(size());
        std::vector<Vector4i> gradient_map(size());

        for (integer i = 0; i < edge_edge_collisions.size(); i++) {
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
        }

        integer ee_collision_size = static_cast<integer>(edge_edge_collisions.size());
        for (integer i = 0; i < vertex_face_collisions.size(); i++) {
            VertexFaceCollision collision = vertex_face_collisions[i];
            gradient_per_collision[i + ee_collision_size] = Matrix3x4r::Zero();
            gradient_map[i + ee_collision_size] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const double f = cipc_barrier(d, dhat, dmin);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            const double m = collision.mollifier(curr_pos, collision.eps);
            const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
            gradient_per_collision[i + ee_collision_size] = f * m_grad + m * f_grad * d_grad;
        }
        for (integer i = 0; i < size(); i++) {
            Vector4i index = gradient_map[i];
            gradient(Eigen::all, index) = gradient_per_collision[i];
        }
        return gradient;
    }

    SparseMatrixXr ComputeBarrierHessian(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces,
        double dmin) const {
        cipc_assert(is_built, "not built yet for barrier potential");
        integer vertex_num = static_cast<integer>(vertices.cols());
        std::vector<Matrix12r> hessian_per_collision(size());
        std::vector<Vector4i> hessian_map(size());

        integer ee_collision_size = static_cast<integer>(edge_edge_collisions.size());
        for (integer i = 0; i < ee_collision_size; i++) {
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

            // TO DO: project to spd
        }

        for (integer i = 0; i < vertex_face_collisions.size(); i++) {
            VertexFaceCollision collision = vertex_face_collisions[i];
            hessian_per_collision[i + ee_collision_size] = Matrix12r::Zero();
            hessian_map[i + ee_collision_size] = collision.vertices_idx(edges, faces);
            Matrix3x4r curr_pos = collision.vertices(vertices, edges, faces);
            const double d = collision.distance(curr_pos);
            const Matrix3x4r d_grad = collision.distance_grad(curr_pos);
            const Matrix12r d_hess = collision.distance_hess(curr_pos);
            const double f_grad = cipc_barrier_first_derivative(d, dhat, dmin);
            const double f_hess = cipc_barrier_second_derivative(d, dhat, dmin);

            if (!collision.is_mollified()) {
                hessian_per_collision[i + ee_collision_size] =
                    f_hess * d_grad.reshaped() * d_grad.reshaped().transpose() + f_grad * d_hess;
            } else {
                const double f = cipc_barrier(d, dhat, dmin);
                const double m = collision.mollifier(curr_pos, collision.eps);
                const Matrix3x4r m_grad = collision.mollifier_grad(curr_pos, collision.eps);
                const Matrix12r m_hess = collision.mollifier_hess(curr_pos, collision.eps);

                hessian_per_collision[i + ee_collision_size] =
                    f * m_hess
                    + f_grad
                          * (d_grad.reshaped() * m_grad.reshaped().transpose()
                             + m_grad.reshaped() * d_grad.reshaped().transpose())
                    + m * f_hess * d_grad.reshaped() * d_grad.reshaped().transpose()
                    + m * f_grad * d_hess;
            }

            // TO DO: project to spd
        }

        const integer dim = 3;
        std::vector<Eigen::Triplet<real>> barrier_hessian_nonzeros;
        for (integer c = 0; c < size(); c++) {
            const Vector4i index = hessian_map[c];
            for (integer i = 0; i < 4; i++)
                for (integer j = 0; j < 4; j++)
                    for (integer di = 0; di < dim; di++)
                        for (integer dj = 0; dj < dim; dj++) {
                            const integer row_idx = index(i) * dim + di;
                            const integer col_idx = index(j) * dim + dj;
                            barrier_hessian_nonzeros.emplace_back(
                                row_idx, col_idx,
                                hessian_per_collision[c](i * dim + di, j * dim + dj));
                        }
        }

        SparseMatrixXr hessian =
            FromTriplet(3 * vertex_num, 3 * vertex_num, barrier_hessian_nonzeros);
        return hessian;
    }
};
} // namespace cipc
