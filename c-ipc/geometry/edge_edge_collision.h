#pragma once

#include <c-ipc/geometry/primative_collision_base.h>
#include <c-ipc/geometry/distance.h>
#include <c-ipc/geometry/accd.h>

namespace cipc {
class EdgeEdgeCollision : public PrimativeCollision {
  public:
    integer edge0_idx, edge1_idx;
    double eps = 0.0;

    EdgeEdgeCollision(integer e0_idx, integer e1_idx) : edge0_idx(e0_idx), edge1_idx(e1_idx) {}

    Vector4i vertices_idx(const Matrix2Xi &edges, const Matrix3Xi &faces) const {
        return Vector4i(
            edges(0, edge0_idx), edges(1, edge0_idx), edges(0, edge1_idx), edges(1, edge1_idx));
    }

    Matrix3x4r vertices(
        const Matrix3Xr &vertices, const Matrix2Xi &edges, const Matrix3Xi &faces) const override {
        return vertices(Eigen::all, vertices_idx(edges, faces));
    }

    real distance(const Matrix3x4r &position) const override {
        return edge_edge_distance(
            position.col(0), position.col(1), position.col(2), position.col(3),
            EdgeEdgeDistType::AUTO);
    }

    Matrix3x4r distance_grad(const Matrix3x4r &position) const override {
        return edge_edge_distance_gradient(
            position.col(0), position.col(1), position.col(2), position.col(3),
            EdgeEdgeDistType::AUTO);
    }

    Matrix12r distance_hess(const Matrix3x4r &position) const override {
        return edge_edge_distance_hessian(
            position.col(0), position.col(1), position.col(2), position.col(3),
            EdgeEdgeDistType::AUTO);
    }

    bool is_mollified() const override { return true; }

    double mollifier(const Matrix3x4r &position, double eps) const override {
        const Vector3r ea0 = position.col(0), ea1 = position.col(1), eb0 = position.col(2),
                       eb1 = position.col(3);
        const double edge_edge_cross_norm_squared = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
        return edge_edge_mollifier(edge_edge_cross_norm_squared, eps);
    }

    Matrix3x4r mollifier_grad(const Matrix3x4r &position, double eps) const override {
        const Vector3r ea0 = position.col(0), ea1 = position.col(1), eb0 = position.col(2),
                       eb1 = position.col(3);
        const double edge_edge_cross_norm_squared = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
        if (edge_edge_cross_norm_squared < eps) {
            return edge_edge_mollifier_gradient(edge_edge_cross_norm_squared, eps)
                   * edge_edge_cross_squarednorm_gradient(ea0, ea1, eb0, eb1);
        }
        return Matrix3x4r::Zero();
    }

    Matrix12r mollifier_hess(const Matrix3x4r &position, double eps) const override {
        const Vector3r ea0 = position.col(0), ea1 = position.col(1), eb0 = position.col(2),
                       eb1 = position.col(3);
        const double edge_edge_cross_norm_squared = edge_edge_cross_squarednorm(ea0, ea1, eb0, eb1);
        if (edge_edge_cross_norm_squared < eps) {
            const Vector12r grad =
                (edge_edge_cross_squarednorm_gradient(ea0, ea1, eb0, eb1)).reshaped();
            return (edge_edge_mollifier_gradient(edge_edge_cross_norm_squared, eps)
                    * edge_edge_cross_squarednorm_hessian(ea0, ea1, eb0, eb1))
                   + ((edge_edge_mollifier_hessian(edge_edge_cross_norm_squared, eps) * grad)
                      * grad.transpose());
        }
        return Matrix12r::Zero();
    }

    double compute_accd_timestep(
        const Matrix3x4r &pos0, const Matrix3x4r &pos1, const double thickness,
        const double t_ccd_fullstep, const integer max_iteration) const override {
        double t_ccd_addictive = 0.0;

        const Vector3r ea0 = pos0.col(0), ea1 = pos0.col(1), eb0 = pos0.col(2), eb1 = pos0.col(3);
        const double init_dist = edge_edge_distance(ea0, ea1, eb0, eb1, EdgeEdgeDistType::AUTO);
        if ((pos0 - pos1).squaredNorm() == 0.0) {
            if (init_dist > thickness) { return t_ccd_fullstep; }
            printf("initial distance is below dmin, toi = 0!\n");
            return 0.0;
        }

        if (!edge_edge_accd(
                pos0, pos1, thickness, t_ccd_fullstep, t_ccd_addictive, max_iteration, 0.9)) {
            return t_ccd_fullstep;
        }
        return t_ccd_addictive;
    }

  private:
    double edge_edge_mollifier(const double cross_norm_squared, const double eps) const {
        if (cross_norm_squared < eps) {
            const double x_div_eps_x = cross_norm_squared / eps;
            return (-x_div_eps_x + 2) * x_div_eps_x;
        } else {
            return 1.0;
        }
    }

    // gradient w.r.t. squared cross norm
    double edge_edge_mollifier_gradient(const double cross_norm_squared, const double eps) const {
        if (cross_norm_squared < eps) {
            const double one_div_eps = 1 / eps;
            return 2 * one_div_eps * (-one_div_eps * cross_norm_squared + 1);
        } else {
            return 0.0;
        }
    }

    // hessian w.r.t. squared cross norm
    double edge_edge_mollifier_hessian(const double cross_norm_squared, const double eps) const {
        if (cross_norm_squared < eps) {
            return -2 / (eps * eps);
        } else {
            return 0.0;
        }
    }

    double edge_edge_cross_squarednorm(
        const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) const {
        return (ea1 - ea0).cross(eb1 - eb0).squaredNorm();
    }

    Matrix3x4r edge_edge_cross_squarednorm_gradient(
        const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) const {
        Matrix3x4r grad = Matrix3x4r::Zero();
        double v01 = ea0(0), v02 = ea0(1), v03 = ea0(2), v11 = ea1(0), v12 = ea1(1), v13 = ea1(2),
               v21 = eb0(0), v22 = eb0(1), v23 = eb0(2), v31 = eb1(0), v32 = eb1(1), v33 = eb1(2);
        double t8, t9, t10, t11, t12, t13, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33;

        t8 = -v11 + v01;
        t9 = -v12 + v02;
        t10 = -v13 + v03;
        t11 = -v31 + v21;
        t12 = -v32 + v22;
        t13 = -v33 + v23;
        t23 = t8 * t12 + -(t9 * t11);
        t24 = t8 * t13 + -(t10 * t11);
        t25 = t9 * t13 + -(t10 * t12);
        t26 = t8 * t23 * 2.0;
        t27 = t9 * t23 * 2.0;
        t28 = t8 * t24 * 2.0;
        t29 = t10 * t24 * 2.0;
        t30 = t9 * t25 * 2.0;
        t31 = t10 * t25 * 2.0;
        t32 = t11 * t23 * 2.0;
        t33 = t12 * t23 * 2.0;
        t23 = t11 * t24 * 2.0;
        t10 = t13 * t24 * 2.0;
        t9 = t12 * t25 * 2.0;
        t8 = t13 * t25 * 2.0;
        grad(0, 0) = t33 + t10;
        grad(1, 0) = -t32 + t8;
        grad(2, 0) = -t23 - t9;
        grad(0, 1) = -t33 - t10;
        grad(1, 1) = t32 - t8;
        grad(2, 1) = t23 + t9;
        grad(0, 2) = -t27 - t29;
        grad(1, 2) = t26 - t31;
        grad(2, 2) = t28 + t30;
        grad(0, 3) = t27 + t29;
        grad(1, 3) = -t26 + t31;
        grad(2, 3) = -t28 - t30;
        return grad;
    }

    Matrix12r edge_edge_cross_squarednorm_hessian(
        const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) const {
        Matrix12r hess = Matrix12r::Zero();
        double v01 = ea0(0), v02 = ea0(1), v03 = ea0(2), v11 = ea1(0), v12 = ea1(1), v13 = ea1(2),
               v21 = eb0(0), v22 = eb0(1), v23 = eb0(2), v31 = eb1(0), v32 = eb1(1), v33 = eb1(2);
        double t8, t9, t10, t11, t12, t13, t32, t33, t34, t35, t48, t36, t49, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t50, t51, t52, t20, t23, t24, t25, t86, t87, t88,
            t74, t75, t76, t77, t78, t79, t89, t90, t91, t92, t93, t94, t95;

        t8 = -v11 + v01;
        t9 = -v12 + v02;
        t10 = -v13 + v03;
        t11 = -v31 + v21;
        t12 = -v32 + v22;
        t13 = -v33 + v23;
        t32 = t8 * t9 * 2.0;
        t33 = t8 * t10 * 2.0;
        t34 = t9 * t10 * 2.0;
        t35 = t8 * t11 * 2.0;
        t48 = t8 * t12;
        t36 = t48 * 2.0;
        t49 = t9 * t11;
        t37 = t49 * 2.0;
        t38 = t48 * 4.0;
        t48 = t8 * t13;
        t39 = t48 * 2.0;
        t40 = t49 * 4.0;
        t41 = t9 * t12 * 2.0;
        t49 = t10 * t11;
        t42 = t49 * 2.0;
        t43 = t48 * 4.0;
        t48 = t9 * t13;
        t44 = t48 * 2.0;
        t45 = t49 * 4.0;
        t49 = t10 * t12;
        t46 = t49 * 2.0;
        t47 = t48 * 4.0;
        t48 = t49 * 4.0;
        t49 = t10 * t13 * 2.0;
        t50 = t11 * t12 * 2.0;
        t51 = t11 * t13 * 2.0;
        t52 = t12 * t13 * 2.0;
        t20 = t8 * t8 * 2.0;
        t9 = t9 * t9 * 2.0;
        t8 = t10 * t10 * 2.0;
        t23 = t11 * t11 * 2.0;
        t24 = t12 * t12 * 2.0;
        t25 = t13 * t13 * 2.0;
        t86 = t35 + t41;
        t87 = t35 + t49;
        t88 = t41 + t49;
        t74 = t20 + t9;
        t75 = t20 + t8;
        t76 = t9 + t8;
        t77 = t23 + t24;
        t78 = t23 + t25;
        t79 = t24 + t25;
        t89 = t40 + -t36;
        t90 = t36 + -t40;
        t91 = t37 + -t38;
        t92 = t38 + -t37;
        t93 = t45 + -t39;
        t94 = t39 + -t45;
        t95 = t42 + -t43;
        t37 = t43 + -t42;
        t39 = t48 + -t44;
        t45 = t44 + -t48;
        t38 = t46 + -t47;
        t40 = t47 + -t46;
        t36 = -t35 + -t41;
        t13 = -t35 + -t49;
        t11 = -t41 + -t49;
        t12 = -t20 + -t9;
        t10 = -t20 + -t8;
        t8 = -t9 + -t8;
        t9 = -t23 + -t24;
        t49 = -t23 + -t25;
        t48 = -t24 + -t25;

        hess(0, 0) = t79;
        hess(1, 0) = -t50;
        hess(2, 0) = -t51;
        hess(3, 0) = t48;
        hess(4, 0) = t50;
        hess(5, 0) = t51;
        hess(6, 0) = t11;
        hess(7, 0) = t92;
        hess(8, 0) = t37;
        hess(9, 0) = t88;
        hess(10, 0) = t91;
        hess(11, 0) = t95;

        hess(0, 1) = -t50;
        hess(1, 1) = t78;
        hess(2, 1) = -t52;
        hess(3, 1) = t50;
        hess(4, 1) = t49;
        hess(5, 1) = t52;
        hess(6, 1) = t89;
        hess(7, 1) = t13;
        hess(8, 1) = t40;
        hess(9, 1) = t90;
        hess(10, 1) = t87;
        hess(11, 1) = t38;

        hess(0, 2) = -t51;
        hess(1, 2) = -t52;
        hess(2, 2) = t77;
        hess(3, 2) = t51;
        hess(4, 2) = t52;
        hess(5, 2) = t9;
        hess(6, 2) = t93;
        hess(7, 2) = t39;
        hess(8, 2) = t36;
        hess(9, 2) = t94;
        hess(10, 2) = t45;
        hess(11, 2) = t86;

        hess(0, 3) = t48;
        hess(1, 3) = t50;
        hess(2, 3) = t51;
        hess(3, 3) = t79;
        hess(4, 3) = -t50;
        hess(5, 3) = -t51;
        hess(6, 3) = t88;
        hess(7, 3) = t91;
        hess(8, 3) = t95;
        hess(9, 3) = t11;
        hess(10, 3) = t92;
        hess(11, 3) = t37;

        hess(0, 4) = t50;
        hess(1, 4) = t49;
        hess(2, 4) = t52;
        hess(3, 4) = -t50;
        hess(4, 4) = t78;
        hess(5, 4) = -t52;
        hess(6, 4) = t90;
        hess(7, 4) = t87;
        hess(8, 4) = t38;
        hess(9, 4) = t89;
        hess(10, 4) = t13;
        hess(11, 4) = t40;

        hess(0, 5) = t51;
        hess(1, 5) = t52;
        hess(2, 5) = t9;
        hess(3, 5) = -t51;
        hess(4, 5) = -t52;
        hess(5, 5) = t77;
        hess(6, 5) = t94;
        hess(7, 5) = t45;
        hess(8, 5) = t86;
        hess(9, 5) = t93;
        hess(10, 5) = t39;
        hess(11, 5) = t36;

        hess(0, 6) = t11;
        hess(1, 6) = t89;
        hess(2, 6) = t93;
        hess(3, 6) = t88;
        hess(4, 6) = t90;
        hess(5, 6) = t94;
        hess(6, 6) = t76;
        hess(7, 6) = -t32;
        hess(8, 6) = -t33;
        hess(9, 6) = t8;
        hess(10, 6) = t32;
        hess(11, 6) = t33;

        hess(0, 7) = t92;
        hess(1, 7) = t13;
        hess(2, 7) = t39;
        hess(3, 7) = t91;
        hess(4, 7) = t87;
        hess(5, 7) = t45;
        hess(6, 7) = -t32;
        hess(7, 7) = t75;
        hess(8, 7) = -t34;
        hess(9, 7) = t32;
        hess(10, 7) = t10;
        hess(11, 7) = t34;

        hess(0, 8) = t37;
        hess(1, 8) = t40;
        hess(2, 8) = t36;
        hess(3, 8) = t95;
        hess(4, 8) = t38;
        hess(5, 8) = t86;
        hess(6, 8) = -t33;
        hess(7, 8) = -t34;
        hess(8, 8) = t74;
        hess(9, 8) = t33;
        hess(10, 8) = t34;
        hess(11, 8) = t12;

        hess(0, 9) = t88;
        hess(1, 9) = t90;
        hess(2, 9) = t94;
        hess(3, 9) = t11;
        hess(4, 9) = t89;
        hess(5, 9) = t93;
        hess(6, 9) = t8;
        hess(7, 9) = t32;
        hess(8, 9) = t33;
        hess(9, 9) = t76;
        hess(10, 9) = -t32;
        hess(11, 9) = -t33;

        hess(0, 10) = t91;
        hess(1, 10) = t87;
        hess(2, 10) = t45;
        hess(3, 10) = t92;
        hess(4, 10) = t13;
        hess(5, 10) = t39;
        hess(6, 10) = t32;
        hess(7, 10) = t10;
        hess(8, 10) = t34;
        hess(9, 10) = -t32;
        hess(10, 10) = t75;
        hess(11, 10) = -t34;

        hess(0, 11) = t95;
        hess(1, 11) = t38;
        hess(2, 11) = t86;
        hess(3, 11) = t37;
        hess(4, 11) = t40;
        hess(5, 11) = t36;
        hess(6, 11) = t33;
        hess(7, 11) = t34;
        hess(8, 11) = t12;
        hess(9, 11) = -t33;
        hess(10, 11) = -t34;
        hess(11, 11) = t74;
        return hess;
    }

    Matrix3x4r edge_edge_cross_squarednorm_gradient(
        const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1,
        double scale) const {

        Matrix3x4r grad = Matrix3x4r::Zero();
        double ea0x = ea0(0), ea0y = ea0(1), ea0z = ea0(2), ea1x = ea1(0), ea1y = ea1(1),
               ea1z = ea1(2), eb0x = eb0(0), eb0y = eb0(1), eb0z = eb0(2), eb1x = eb1(0),
               eb1y = eb1(1), eb1z = eb1(2);
        const auto t0 = ea0x - ea1x;
        const auto t1 = eb0x - eb1x;
        const auto t2 = eb0y - eb1y;
        const auto t3 = eb0z - eb1z;
        const auto t4 = 2 * scale;
        const auto t5 = t4 * ((t1 * t1) + (t2 * t2) + (t3 * t3));
        const auto t6 = t0 * t5;
        const auto t7 = ea0y - ea1y;
        const auto t8 = t5 * t7;
        const auto t9 = ea0z - ea1z;
        const auto t10 = t5 * t9;
        const auto t11 = t4 * ((t0 * t0) + (t7 * t7) + (t9 * t9));
        const auto t12 = t1 * t11;
        const auto t13 = t11 * t2;
        const auto t14 = t11 * t3;

        grad(0, 0) = t6;
        grad(1, 0) = t8;
        grad(2, 0) = t10;
        grad(0, 1) = -t6;
        grad(1, 1) = -t8;
        grad(2, 1) = -t10;
        grad(0, 2) = t12;
        grad(1, 2) = t13;
        grad(2, 2) = t14;
        grad(0, 3) = -t12;
        grad(1, 3) = -t13;
        grad(2, 3) = -t14;
        return grad;
    }
};
} // namespace cipc
