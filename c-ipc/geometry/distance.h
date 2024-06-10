#pragma once

#include <c-ipc/solver/eigen.h>

namespace cipc {
enum class PointEdgeDistType { P_E0, P_E1, P_E, AUTO };
enum class PointTriangleDistType { P_T0, P_T1, P_T2, P_T0T1, P_T1T2, P_T2T0, P_T, AUTO };
enum class EdgeEdgeDistType {
    EA0_EB0,
    EA0_EB1,
    EA1_EB0,
    EA1_EB1,
    EA_EB0,
    EA_EB1,
    EA0_EB,
    EA1_EB,
    EA_EB,
    AUTO
};

// distance type deduction
static PointEdgeDistType
point_edge_dist_type(const Vector3r &p, const Vector3r &e0, const Vector3r &e1) {
    Vector3r edge = e1 - e0;
    const real edge_length_square = edge.squaredNorm();
    if (edge_length_square == 0) { return PointEdgeDistType::P_E0; }

    const real ratio = edge.dot(p - e0) / edge_length_square;
    if (ratio < 0) {
        return PointEdgeDistType::P_E0;
    } else if (ratio > 1) {
        return PointEdgeDistType::P_E1;
    } else {
        return PointEdgeDistType::P_E;
    }
}

static PointTriangleDistType point_triangle_dist_type(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2) {
    const Vector3r normal = (t1 - t0).cross(t2 - t0);
    Matrix2x3r basis, coeff;

    // project (p-t0) to the triangel plane
    // calculate coefficient of basis t1 - t0, (t1 - t0).cross(normal)
    basis.row(0) = t1 - t0;
    basis.row(1) = basis.row(0).cross(normal);
    coeff.col(0) = (basis * basis.transpose()).ldlt().solve(basis * (p - t0));
    if (coeff(0, 0) > 0.0 && coeff(0, 0) < 1.0 && coeff(1, 0) >= 0.0) {
        // coeff(0,0) is the component of vector t1 - t0
        // coeff(0,0) > 0 && < 1 means cloest point is on the t1 - t0
        // coeff(1,0) is the component of another vector
        // coeff(1,0) >= 0 means projection(p) is away from t2
        return PointTriangleDistType::P_T0T1;
    }

    basis.row(0) = t2 - t1;
    basis.row(1) = basis.row(0).cross(normal);
    coeff.col(1) = (basis * basis.transpose()).ldlt().solve(basis * (p - t1));
    if (coeff(0, 1) > 0.0 && coeff(0, 1) < 1.0 && coeff(1, 1) >= 0.0) {
        return PointTriangleDistType::P_T1T2;
    }

    basis.row(0) = t0 - t2;
    basis.row(1) = basis.row(0).cross(normal);
    coeff.col(2) = (basis * basis.transpose()).ldlt().solve(basis * (p - t2));
    if (coeff(0, 2) > 0.0 && coeff(0, 2) < 1.0 && coeff(1, 2) >= 0.0) {
        return PointTriangleDistType::P_T2T0;
    }

    if (coeff(0, 0) <= 0.0 && coeff(0, 2) >= 1.0) {
        return PointTriangleDistType::P_T0;
    } else if (coeff(0, 1) <= 0.0 && coeff(0, 0) >= 1.0) {
        return PointTriangleDistType::P_T1;
    } else if (coeff(0, 2) <= 0.0 && coeff(0, 1) >= 1.0) {
        return PointTriangleDistType::P_T2;
    } else {
        return PointTriangleDistType::P_T;
    }
}

static EdgeEdgeDistType parallel_edge_dist_type(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    const Vector3r ea = ea1 - ea0;
    const double alpha = (eb0 - ea0).dot(ea) / ea.squaredNorm();
    const double beta = (eb1 - ea0).dot(ea) / ea.squaredNorm();

    uint8_t eac;
    uint8_t ebc;
    if (alpha < 0) {
        eac = (0 <= beta && beta <= 1) ? 2 : 0;
        ebc = (beta <= alpha) ? 0 : (beta <= 1 ? 1 : 2);
    } else if (alpha > 1) {
        eac = (0 <= beta && beta <= 1) ? 2 : 1;
        ebc = (beta >= alpha) ? 0 : (0 <= beta ? 1 : 2);
    } else {
        eac = 2;
        ebc = 0;
    }
    cipc_assert(eac != 2 || ebc != 2, "degenerate case in parallel edges");
    return EdgeEdgeDistType(ebc < 2 ? (eac << 1 | ebc) : (6 + eac));
}

static EdgeEdgeDistType edge_edge_dist_type(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    constexpr double PARALLEL_THRESHOLD = 1.0e-15;
    const Vector3r u = ea1 - ea0;
    const Vector3r v = eb1 - eb0;
    const Vector3r w = ea0 - eb0;

    const double a = u.squaredNorm();
    const double b = u.dot(v);
    const double c = v.squaredNorm();
    const double d = u.dot(w);
    const double e = v.dot(w);
    const double D = a * c - b * b;

    const double parallel_tol = PARALLEL_THRESHOLD * std::max(1.0, a * c);
    if (u.cross(v).squaredNorm() < parallel_tol) {
        return parallel_edge_dist_type(ea0, ea1, eb0, eb1);
    }

    EdgeEdgeDistType default_case = EdgeEdgeDistType::EA_EB;
    const double sN = (b * e - c * d);
    double tN, tD;
    if (sN <= 0.0) {
        tN = e;
        tD = c;
        default_case = EdgeEdgeDistType::EA0_EB;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        default_case = EdgeEdgeDistType::EA1_EB;
    } else {
        tN = (a * e - b * d);
        tD = D;
        if (tN > 0.0 && tN < tD && u.cross(v).squaredNorm() < parallel_tol) {

            if (sN < D / 2) {
                tN = e;
                tD = c;
                default_case = EdgeEdgeDistType::EA0_EB;
            } else {
                tN = e + b;
                tD = c;
                default_case = EdgeEdgeDistType::EA1_EB;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return EdgeEdgeDistType::EA0_EB0;
        } else if (-d >= a) {
            return EdgeEdgeDistType::EA1_EB0;
        } else {
            return EdgeEdgeDistType::EA_EB0;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return EdgeEdgeDistType::EA0_EB1;
        } else if ((-d + b) >= a) {
            return EdgeEdgeDistType::EA1_EB1;
        } else {
            return EdgeEdgeDistType::EA_EB1;
        }
    }

    return default_case;
}

// IPC use distance d^2
static real point_point_distance(const Vector3r &p0, const Vector3r &p1) {
    return (p0 - p1).squaredNorm();
}

static real point_edge_distance(
    const Vector3r &p, const Vector3r &e0, const Vector3r &e1, PointEdgeDistType dist_type) {
    if (dist_type == PointEdgeDistType::AUTO) { dist_type = point_edge_dist_type(p, e0, e1); }
    switch (dist_type) {
    case PointEdgeDistType::P_E0: return point_point_distance(p, e0);
    case PointEdgeDistType::P_E1: return point_point_distance(p, e1);
    case PointEdgeDistType::P_E:
        return (e0 - p).cross(e1 - p).squaredNorm() / (e0 - e1).squaredNorm();
    default: printf("Invalid dist type for point edge\n"); std::abort();
    }
}

static real point_triangle_distance_in(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2) {
    const Vector3r normal = (t1 - t0).cross(t2 - t0);
    return std::pow((p - t0).dot(normal), 2.0) / normal.squaredNorm();
}

static real point_triangle_distance(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2,
    PointTriangleDistType dist_type) {
    if (dist_type == PointTriangleDistType::AUTO) {
        dist_type = point_triangle_dist_type(p, t0, t1, t2);
    }
    switch (dist_type) {
    case PointTriangleDistType::P_T0: return point_point_distance(p, t0);
    case PointTriangleDistType::P_T1: return point_point_distance(p, t1);
    case PointTriangleDistType::P_T2: return point_point_distance(p, t2);
    case PointTriangleDistType::P_T0T1:
        return (t0 - p).cross(t1 - p).squaredNorm() / (t0 - t1).squaredNorm();
    case PointTriangleDistType::P_T1T2:
        return (t1 - p).cross(t2 - p).squaredNorm() / (t1 - t2).squaredNorm();
    case PointTriangleDistType::P_T2T0:
        return (t0 - p).cross(t2 - p).squaredNorm() / (t0 - t2).squaredNorm();
    case PointTriangleDistType::P_T: return point_triangle_distance_in(p, t0, t1, t2);
    default: printf("Invalid dist type for point triangle\n"); std::abort();
    }
    return 0.0;
}

static real edge_edge_distance_in(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    const Vector3r normal = (ea1 - ea0).cross(eb1 - eb0);
    const double line_to_line = (eb0 - ea0).dot(normal);
    return line_to_line * line_to_line / normal.squaredNorm();
}

static real edge_edge_distance(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1,
    EdgeEdgeDistType dist_type) {
    if (dist_type == EdgeEdgeDistType::AUTO) {
        dist_type = edge_edge_dist_type(ea0, ea1, eb0, eb1);
    }
    switch (dist_type) {
    case EdgeEdgeDistType::EA0_EB0: return point_point_distance(ea0, eb0);
    case EdgeEdgeDistType::EA0_EB1: return point_point_distance(ea0, eb1);
    case EdgeEdgeDistType::EA1_EB0: return point_point_distance(ea1, eb0);
    case EdgeEdgeDistType::EA1_EB1: return point_point_distance(ea1, eb1);
    case EdgeEdgeDistType::EA0_EB:
        return (eb0 - ea0).cross(eb1 - ea0).squaredNorm() / (eb0 - eb1).squaredNorm();
    case EdgeEdgeDistType::EA1_EB:
        return (eb0 - ea1).cross(eb1 - ea1).squaredNorm() / (eb0 - eb1).squaredNorm();
    case EdgeEdgeDistType::EA_EB0:
        return (ea0 - eb0).cross(ea1 - eb0).squaredNorm() / (ea0 - ea1).squaredNorm();
    case EdgeEdgeDistType::EA_EB1:
        return (ea0 - eb1).cross(ea1 - eb1).squaredNorm() / (ea0 - ea1).squaredNorm();
    case EdgeEdgeDistType::EA_EB: return edge_edge_distance_in(ea0, ea1, eb0, eb1);
    default: printf("Invalid dist type for point triangle\n"); std::abort();
    }
}

// gradient part
static Matrix3x2r point_point_distance_gradient(const Vector3r &p0, const Vector3r &p1) {
    Matrix3x2r gradient = Matrix3x2r::Zero();
    gradient.col(0) = 2.0 * (p0 - p1);
    gradient.col(1) = -gradient.col(0);
    return gradient;
}

static Matrix3r
point_line_distance_gradient(const Vector3r &p, const Vector3r &e0, const Vector3r &e1) {
    Matrix3r gradient = Matrix3r::Zero();

    double v01 = p(0), v02 = p(1), v03 = p(2), v11 = e0(0), v12 = e0(1), v13 = e0(2), v21 = e1(0),
           v22 = e1(1), v23 = e1(2);
    double t17, t18, t19, t20, t21, t22, t23, t24, t25, t42, t44, t45, t46, t43, t50, t51, t52;

    t17 = -v11 + v01;
    t18 = -v12 + v02;
    t19 = -v13 + v03;
    t20 = -v21 + v01;
    t21 = -v22 + v02;
    t22 = -v23 + v03;
    t23 = -v21 + v11;
    t24 = -v22 + v12;
    t25 = -v23 + v13;
    t42 = 1.0 / ((t23 * t23 + t24 * t24) + t25 * t25);
    t44 = t17 * t21 + -(t18 * t20);
    t45 = t17 * t22 + -(t19 * t20);
    t46 = t18 * t22 + -(t19 * t21);
    t43 = t42 * t42;
    t50 = (t44 * t44 + t45 * t45) + t46 * t46;
    t51 = (v11 * 2.0 + -(v21 * 2.0)) * t43 * t50;
    t52 = (v12 * 2.0 + -(v22 * 2.0)) * t43 * t50;
    t43 = (v13 * 2.0 + -(v23 * 2.0)) * t43 * t50;
    gradient(0, 0) = t42 * (t24 * t44 * 2.0 + t25 * t45 * 2.0);
    gradient(1, 0) = -t42 * (t23 * t44 * 2.0 - t25 * t46 * 2.0);
    gradient(2, 0) = -t42 * (t23 * t45 * 2.0 + t24 * t46 * 2.0);
    gradient(0, 1) = -t51 - t42 * (t21 * t44 * 2.0 + t22 * t45 * 2.0);
    gradient(1, 1) = -t52 + t42 * (t20 * t44 * 2.0 - t22 * t46 * 2.0);
    gradient(2, 1) = -t43 + t42 * (t20 * t45 * 2.0 + t21 * t46 * 2.0);
    gradient(0, 2) = t51 + t42 * (t18 * t44 * 2.0 + t19 * t45 * 2.0);
    gradient(1, 2) = t52 - t42 * (t17 * t44 * 2.0 - t19 * t46 * 2.0);
    gradient(2, 2) = t43 - t42 * (t17 * t45 * 2.0 + t18 * t46 * 2.0);
    return gradient;
}

static Matrix3x4r line_line_distance_gradient(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    Matrix3x4r gradient = Matrix3x4r::Zero();
    double v01 = ea0(0), v02 = ea0(1), v03 = ea0(2), v11 = ea1(0), v12 = ea1(1), v13 = ea1(2),
           v21 = eb0(0), v22 = eb0(1), v23 = eb0(2), v31 = eb1(0), v32 = eb1(1), v33 = eb1(2);
    double t11, t12, t13, t14, t15, t16, t17, t18, t19, t32, t33, t34, t35, t36, t37, t44, t45, t46,
        t75, t77, t76, t78, t79, t80, t81, t83;

    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t14 = -v21 + v01;
    t15 = -v22 + v02;
    t16 = -v23 + v03;
    t17 = -v31 + v21;
    t18 = -v32 + v22;
    t19 = -v33 + v23;
    t32 = t14 * t18;
    t33 = t15 * t17;
    t34 = t14 * t19;
    t35 = t16 * t17;
    t36 = t15 * t19;
    t37 = t16 * t18;
    t44 = t11 * t18 + -(t12 * t17);
    t45 = t11 * t19 + -(t13 * t17);
    t46 = t12 * t19 + -(t13 * t18);
    t75 = 1.0 / ((t44 * t44 + t45 * t45) + t46 * t46);
    t77 = (t16 * t44 + t14 * t46) + -(t15 * t45);
    t76 = t75 * t75;
    t78 = t77 * t77;
    t79 = (t12 * t44 * 2.0 + t13 * t45 * 2.0) * t76 * t78;
    t80 = (t11 * t45 * 2.0 + t12 * t46 * 2.0) * t76 * t78;
    t81 = (t18 * t44 * 2.0 + t19 * t45 * 2.0) * t76 * t78;
    t18 = (t17 * t45 * 2.0 + t18 * t46 * 2.0) * t76 * t78;
    t83 = (t11 * t44 * 2.0 + -(t13 * t46 * 2.0)) * t76 * t78;
    t19 = (t17 * t44 * 2.0 + -(t19 * t46 * 2.0)) * t76 * t78;
    t76 = t75 * t77;
    gradient(0, 0) = -t81 + t76 * ((-t36 + t37) + t46) * 2.0;
    gradient(1, 0) = t19 - t76 * ((-t34 + t35) + t45) * 2.0;
    gradient(2, 0) = t18 + t76 * ((-t32 + t33) + t44) * 2.0;
    gradient(0, 1) = t81 + t76 * (t36 - t37) * 2.0;
    gradient(1, 1) = -t19 - t76 * (t34 - t35) * 2.0;
    gradient(2, 1) = -t18 + t76 * (t32 - t33) * 2.0;
    t17 = t12 * t16 + -(t13 * t15);
    gradient(0, 2) = t79 - t76 * (t17 + t46) * 2.0;
    t18 = t11 * t16 + -(t13 * t14);
    gradient(1, 2) = -t83 + t76 * (t18 + t45) * 2.0;
    t19 = t11 * t15 + -(t12 * t14);
    gradient(2, 2) = -t80 - t76 * (t19 + t44) * 2.0;
    gradient(0, 3) = -t79 + t76 * t17 * 2.0;
    gradient(1, 3) = t83 - t76 * t18 * 2.0;
    gradient(2, 3) = t80 + t76 * t19 * 2.0;
    return gradient;
}

static Matrix3x4r edge_edge_distance_gradient(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1,
    EdgeEdgeDistType dist_type) {
    if (dist_type == EdgeEdgeDistType::AUTO) {
        dist_type = edge_edge_dist_type(ea0, ea1, eb0, eb1);
    }

    Matrix3x4r gradient = Matrix3x4r::Zero();

    switch (dist_type) {
    case EdgeEdgeDistType::EA0_EB0: {
        const Matrix3x2r local_grad = point_point_distance_gradient(ea0, eb0);
        gradient.col(0) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        break;
    }
    case EdgeEdgeDistType::EA0_EB1: {
        const Matrix3x2r local_grad = point_point_distance_gradient(ea0, eb1);
        gradient.col(0) = local_grad.col(0);
        gradient.col(3) = local_grad.col(1);
        break;
    }
    case EdgeEdgeDistType::EA1_EB0: {
        const Matrix3x2r local_grad = point_point_distance_gradient(ea1, eb0);
        gradient.col(1) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        break;
    }
    case EdgeEdgeDistType::EA1_EB1: {
        const Matrix3x2r local_grad = point_point_distance_gradient(ea1, eb1);
        gradient.col(1) = local_grad.col(0);
        gradient.col(3) = local_grad.col(1);
        break;
    }
    case EdgeEdgeDistType::EA_EB0: {
        const Matrix3r local_grad = point_line_distance_gradient(eb0, ea0, ea1);
        gradient.col(2) = local_grad.col(0);
        gradient.col(0) = local_grad.col(1);
        gradient.col(1) = local_grad.col(2);
        break;
    }
    case EdgeEdgeDistType::EA_EB1: {
        const Matrix3r local_grad = point_line_distance_gradient(eb1, ea0, ea1);
        gradient.col(3) = local_grad.col(0);
        gradient.col(0) = local_grad.col(1);
        gradient.col(1) = local_grad.col(2);
        break;
    }
    case EdgeEdgeDistType::EA0_EB: {
        const Matrix3r local_grad = point_line_distance_gradient(ea0, eb0, eb1);
        gradient.col(0) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        gradient.col(3) = local_grad.col(2);
        break;
    }
    case EdgeEdgeDistType::EA1_EB: {
        const Matrix3r local_grad = point_line_distance_gradient(ea1, eb0, eb1);
        gradient.col(1) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        gradient.col(3) = local_grad.col(2);
        break;
    }
    case EdgeEdgeDistType::EA_EB: {
        gradient = line_line_distance_gradient(ea0, ea1, eb0, eb1);
        break;
    }
    default: break;
    }
    return gradient;
}

static Matrix3x4r point_plane_distance_gradient(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2) {
    Matrix3x4r gradient = Matrix3x4r::Zero();
    double v01 = p(0), v02 = p(1), v03 = p(2), v11 = t0(0), v12 = t0(1), v13 = t0(2), v21 = t1(0),
           v22 = t1(1), v23 = t1(2), v31 = t2(0), v32 = t2(1), v33 = t2(2);
    double t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t32, t33, t34, t43, t45, t44,
        t46;

    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t14 = -v21 + v11;
    t15 = -v22 + v12;
    t16 = -v23 + v13;
    t17 = -v31 + v11;
    t18 = -v32 + v12;
    t19 = -v33 + v13;
    t20 = -v31 + v21;
    t21 = -v32 + v22;
    t22 = -v33 + v23;
    t32 = t14 * t18 + -(t15 * t17);
    t33 = t14 * t19 + -(t16 * t17);
    t34 = t15 * t19 + -(t16 * t18);
    t43 = 1.0 / ((t32 * t32 + t33 * t33) + t34 * t34);
    t45 = (t13 * t32 + t11 * t34) + -(t12 * t33);
    t44 = t43 * t43;
    t46 = t45 * t45;
    gradient(0, 0) = t34 * t43 * t45 * 2.0;
    gradient(1, 0) = t33 * t43 * t45 * -2.0;
    gradient(2, 0) = t32 * t43 * t45 * 2.0;
    t45 *= t43;
    gradient(0, 1) = -t44 * t46 * (t21 * t32 * 2.0 + t22 * t33 * 2.0)
                     - t45 * ((t34 + t12 * t22) - t13 * t21) * 2.0;
    t43 = t44 * t46;
    gradient(1, 1) =
        t43 * (t20 * t32 * 2.0 - t22 * t34 * 2.0) + t45 * ((t33 + t11 * t22) - t13 * t20) * 2.0;
    gradient(2, 1) =
        t43 * (t20 * t33 * 2.0 + t21 * t34 * 2.0) - t45 * ((t32 + t11 * t21) - t12 * t20) * 2.0;
    gradient(0, 2) =
        t45 * (t12 * t19 - t13 * t18) * 2.0 + t43 * (t18 * t32 * 2.0 + t19 * t33 * 2.0);
    gradient(1, 2) =
        t45 * (t11 * t19 - t13 * t17) * -2.0 - t43 * (t17 * t32 * 2.0 - t19 * t34 * 2.0);
    gradient(2, 2) =
        t45 * (t11 * t18 - t12 * t17) * 2.0 - t43 * (t17 * t33 * 2.0 + t18 * t34 * 2.0);
    gradient(0, 3) =
        t45 * (t12 * t16 - t13 * t15) * -2.0 - t43 * (t15 * t32 * 2.0 + t16 * t33 * 2.0);
    gradient(1, 3) =
        t45 * (t11 * t16 - t13 * t14) * 2.0 + t43 * (t14 * t32 * 2.0 - t16 * t34 * 2.0);
    gradient(2, 3) =
        t45 * (t11 * t15 - t12 * t14) * -2.0 + t43 * (t14 * t33 * 2.0 + t15 * t34 * 2.0);
    return gradient;
}

static Matrix3x4r point_triangle_distance_gradient(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2,
    PointTriangleDistType dist_type) {
    Matrix3x4r gradient = Matrix3x4r::Zero();
    if (dist_type == PointTriangleDistType::AUTO) {
        dist_type = point_triangle_dist_type(p, t0, t1, t2);
    }

    switch (dist_type) {
    case PointTriangleDistType::P_T0: {
        Matrix3x2r local_grad = point_point_distance_gradient(p, t0);
        gradient.col(0) = local_grad.col(0);
        gradient.col(1) = local_grad.col(1);
        break;
    }
    case PointTriangleDistType::P_T1: {
        Matrix3x2r local_grad = point_point_distance_gradient(p, t1);
        gradient.col(0) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        break;
    }
    case PointTriangleDistType::P_T2: {
        Matrix3x2r local_grad = point_point_distance_gradient(p, t2);
        gradient.col(0) = local_grad.col(0);
        gradient.col(3) = local_grad.col(1);
        break;
    }
    case PointTriangleDistType::P_T0T1: {
        Matrix3r local_grad = point_line_distance_gradient(p, t0, t1);
        gradient.col(0) = local_grad.col(0);
        gradient.col(1) = local_grad.col(1);
        gradient.col(2) = local_grad.col(2);
        break;
    }
    case PointTriangleDistType::P_T1T2: {
        Matrix3r local_grad = point_line_distance_gradient(p, t1, t2);
        gradient.col(0) = local_grad.col(0);
        gradient.col(2) = local_grad.col(1);
        gradient.col(3) = local_grad.col(2);
        break;
    }
    case PointTriangleDistType::P_T2T0: {
        Matrix3r local_grad = point_line_distance_gradient(p, t2, t0);
        gradient.col(0) = local_grad.col(0);
        gradient.col(1) = local_grad.col(2); // t0
        gradient.col(3) = local_grad.col(1); // t2
        break;
    }
    case PointTriangleDistType::P_T: {
        gradient = point_plane_distance_gradient(p, t0, t1, t2);
        break;
    }
    default: break;
    }
    return gradient;
}

// hessian part
static Matrix6r point_point_distance_hessian(const Vector3r &p0, const Vector3r &p1) {
    Matrix6r hessian = Matrix6r::Zero();
    hessian.diagonal().setConstant(2.0);
    for (integer i = 0; i < 3; i++) { hessian(i, i + 3) = hessian(i + 3, i) = -2; }
    return hessian;
}

static Matrix9r
point_line_distance_hessian(const Vector3r &p, const Vector3r &e0, const Vector3r &e1) {
    Matrix9r hessian = Matrix9r::Zero();
    double v01 = p(0), v02 = p(1), v03 = p(2), v11 = e0(0), v12 = e0(1), v13 = e0(2), v21 = e1(0),
           v22 = e1(1), v23 = e1(2);
    double t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t35, t36, t37, t50, t51, t52,
        t53, t54, t55, t56, t62, t70, t71, t75, t79, t80, t84, t88, t38, t39, t40, t41, t42, t43,
        t44, t46, t48, t57, t58, t60, t63, t65, t67, t102, t103, t104, t162, t163, t164, t213, t214,
        t215, t216, t217, t218, t225, t226, t227, t229, t230, t311, t231, t232, t233, t234, t235,
        t236, t237, t238, t239, t240, t245, t279, t281, t282, t283, t287, t289, t247, t248, t249,
        t250, t251, t252, t253, t293, t295, t299, t300, t303, t304, t294, t297, t301, t302;

    t17 = -v11 + v01;
    t18 = -v12 + v02;
    t19 = -v13 + v03;
    t20 = -v21 + v01;
    t21 = -v22 + v02;
    t22 = -v23 + v03;
    t23 = -v21 + v11;
    t24 = -v22 + v12;
    t25 = -v23 + v13;
    t26 = v11 * 2.0 + -(v21 * 2.0);
    t27 = v12 * 2.0 + -(v22 * 2.0);
    t28 = v13 * 2.0 + -(v23 * 2.0);
    t35 = t23 * t23;
    t36 = t24 * t24;
    t37 = t25 * t25;
    t50 = t17 * t21;
    t51 = t18 * t20;
    t52 = t17 * t22;
    t53 = t19 * t20;
    t54 = t18 * t22;
    t55 = t19 * t21;
    t56 = t17 * t20 * 2.0;
    t62 = t18 * t21 * 2.0;
    t70 = t19 * t22 * 2.0;
    t71 = t17 * t23 * 2.0;
    t75 = t18 * t24 * 2.0;
    t79 = t19 * t25 * 2.0;
    t80 = t20 * t23 * 2.0;
    t84 = t21 * t24 * 2.0;
    t88 = t22 * t25 * 2.0;
    t38 = t17 * t17 * 2.0;
    t39 = t18 * t18 * 2.0;
    t40 = t19 * t19 * 2.0;
    t41 = t20 * t20 * 2.0;
    t42 = t21 * t21 * 2.0;
    t43 = t22 * t22 * 2.0;
    t44 = t35 * 2.0;
    t46 = t36 * 2.0;
    t48 = t37 * 2.0;
    t57 = t50 * 2.0;
    t58 = t51 * 2.0;
    t60 = t52 * 2.0;
    t63 = t53 * 2.0;
    t65 = t54 * 2.0;
    t67 = t55 * 2.0;
    t102 = 1.0 / ((t35 + t36) + t37);
    t36 = t50 + -t51;
    t35 = t52 + -t53;
    t37 = t54 + -t55;
    t103 = t102 * t102;
    t104 = std::pow(t102, 3.0);
    t162 = -(t23 * t24 * t102 * 2.0);
    t163 = -(t23 * t25 * t102 * 2.0);
    t164 = -(t24 * t25 * t102 * 2.0);
    t213 = t18 * t36 * 2.0 + t19 * t35 * 2.0;
    t214 = t17 * t35 * 2.0 + t18 * t37 * 2.0;
    t215 = t21 * t36 * 2.0 + t22 * t35 * 2.0;
    t216 = t20 * t35 * 2.0 + t21 * t37 * 2.0;
    t217 = t24 * t36 * 2.0 + t25 * t35 * 2.0;
    t218 = t23 * t35 * 2.0 + t24 * t37 * 2.0;
    t35 = (t36 * t36 + t35 * t35) + t37 * t37;
    t225 = t17 * t36 * 2.0 + -(t19 * t37 * 2.0);
    t226 = t20 * t36 * 2.0 + -(t22 * t37 * 2.0);
    t227 = t23 * t36 * 2.0 + -(t25 * t37 * 2.0);
    t36 = t26 * t103;
    t229 = t36 * t213;
    t37 = t27 * t103;
    t230 = t37 * t213;
    t311 = t28 * t103;
    t231 = t311 * t213;
    t232 = t36 * t214;
    t233 = t37 * t214;
    t234 = t311 * t214;
    t235 = t36 * t215;
    t236 = t37 * t215;
    t237 = t311 * t215;
    t238 = t36 * t216;
    t239 = t37 * t216;
    t240 = t311 * t216;
    t214 = t36 * t217;
    t215 = t37 * t217;
    t216 = t311 * t217;
    t217 = t36 * t218;
    t245 = t37 * t218;
    t213 = t311 * t218;
    t279 = t103 * t35 * 2.0;
    t281 = t26 * t26 * t104 * t35 * 2.0;
    t282 = t27 * t27 * t104 * t35 * 2.0;
    t283 = t28 * t28 * t104 * t35 * 2.0;
    t287 = t26 * t27 * t104 * t35 * 2.0;
    t218 = t26 * t28 * t104 * t35 * 2.0;
    t289 = t27 * t28 * t104 * t35 * 2.0;
    t247 = t36 * t225;
    t248 = t37 * t225;
    t249 = t311 * t225;
    t250 = t36 * t226;
    t251 = t37 * t226;
    t252 = t311 * t226;
    t253 = t36 * t227;
    t35 = t37 * t227;
    t36 = t311 * t227;
    t293 = t102 * (t75 + t79) + t214;
    t295 = -(t102 * (t80 + t84)) + t213;
    t299 = t102 * ((t63 + t22 * t23 * 2.0) + -t60) + t217;
    t300 = t102 * ((t67 + t22 * t24 * 2.0) + -t65) + t245;
    t303 = -(t102 * ((t57 + t17 * t24 * 2.0) + -t58)) + t215;
    t304 = -(t102 * ((t60 + t17 * t25 * 2.0) + -t63)) + t216;
    t294 = t102 * (t71 + t75) + -t213;
    t297 = -(t102 * (t80 + t88)) + t35;
    t88 = -(t102 * (t84 + t88)) + -t214;
    t301 = t102 * ((t58 + t21 * t23 * 2.0) + -t57) + t253;
    t302 = t102 * ((t65 + t21 * t25 * 2.0) + -t67) + t36;
    t84 = t102 * ((t57 + t20 * t24 * 2.0) + -t58) + -t215;
    t80 = t102 * ((t60 + t20 * t25 * 2.0) + -t63) + -t216;
    t75 = -(t102 * ((t63 + t19 * t23 * 2.0) + -t60)) + -t217;
    t227 = -(t102 * ((t67 + t19 * t24 * 2.0) + -t65)) + -t245;
    t311 = ((-(t17 * t19 * t102 * 2.0) + t231) + -t232) + t218;
    t245 = ((-(t20 * t22 * t102 * 2.0) + t237) + -t238) + t218;
    t226 = ((-t102 * (t67 - t54 * 4.0) + t233) + t252) + -t289;
    t28 = ((-t102 * (t63 - t52 * 4.0) + t232) + -t237) + -t218;
    t27 = ((-t102 * (t58 - t50 * 4.0) + t247) + -t236) + -t287;
    t225 = ((-(t102 * (t65 + -(t55 * 4.0))) + t239) + t249) + -t289;
    t26 = ((-(t102 * (t60 + -(t53 * 4.0))) + t238) + -t231) + -t218;
    t103 = ((-(t102 * (t57 + -(t51 * 4.0))) + t250) + -t230) + -t287;
    t104 = (((-(t102 * (t56 + t62)) + t234) + t240) + t279) + -t283;
    t218 = (((-(t102 * (t56 + t70)) + t248) + t251) + t279) + -t282;
    t217 = (((-(t102 * (t62 + t70)) + -t229) + -t235) + t279) + -t281;
    t216 = t102 * (t71 + t79) + -t35;
    t215 = -(t102 * ((t58 + t18 * t23 * 2.0) + -t57)) + -t253;
    t214 = -(t102 * ((t65 + t18 * t25 * 2.0) + -t67)) + -t36;
    t213 = ((-(t17 * t18 * t102 * 2.0) + t230) + -t247) + t287;
    t37 = ((-(t20 * t21 * t102 * 2.0) + t236) + -t250) + t287;
    t36 = ((-(t18 * t19 * t102 * 2.0) + -t233) + -t249) + t289;
    t35 = ((-(t21 * t22 * t102 * 2.0) + -t239) + -t252) + t289;

    hessian(0, 0) = t102 * (t46 + t48);
    hessian(1, 0) = t162;
    hessian(2, 0) = t163;
    hessian(3, 0) = t88;
    hessian(4, 0) = t84;
    hessian(5, 0) = t80;
    hessian(6, 0) = t293;
    hessian(7, 0) = t303;
    hessian(8, 0) = t304;

    hessian(0, 1) = t162;
    hessian(1, 1) = t102 * (t44 + t48);
    hessian(2, 1) = t164;
    hessian(3, 1) = t301;
    hessian(4, 1) = t297;
    hessian(5, 1) = t302;
    hessian(6, 1) = t215;
    hessian(7, 1) = t216;
    hessian(8, 1) = t214;

    hessian(0, 2) = t163;
    hessian(1, 2) = t164;
    hessian(2, 2) = t102 * (t44 + t46);
    hessian(3, 2) = t299;
    hessian(4, 2) = t300;
    hessian(5, 2) = t295;
    hessian(6, 2) = t75;
    hessian(7, 2) = t227;
    hessian(8, 2) = t294;

    hessian(0, 3) = t88;
    hessian(1, 3) = t301;
    hessian(2, 3) = t299;
    hessian(3, 3) = ((t235 * 2.0 + -t279) + t281) + t102 * (t42 + t43);
    hessian(4, 3) = t37;
    hessian(5, 3) = t245;
    hessian(6, 3) = t217;
    hessian(7, 3) = t27;
    hessian(8, 3) = t28;

    hessian(0, 4) = t84;
    hessian(1, 4) = t297;
    hessian(2, 4) = t300;
    hessian(3, 4) = t37;
    hessian(4, 4) = ((t251 * -2.0 + -t279) + t282) + t102 * (t41 + t43);
    hessian(5, 4) = t35;
    hessian(6, 4) = t103;
    hessian(7, 4) = t218;
    hessian(8, 4) = t226;

    hessian(0, 5) = t80;
    hessian(1, 5) = t302;
    hessian(2, 5) = t295;
    hessian(3, 5) = t245;
    hessian(4, 5) = t35;
    hessian(5, 5) = ((t240 * -2.0 + -t279) + t283) + t102 * (t41 + t42);
    hessian(6, 5) = t26;
    hessian(7, 5) = t225;
    hessian(8, 5) = t104;

    hessian(0, 6) = t293;
    hessian(1, 6) = t215;
    hessian(2, 6) = t75;
    hessian(3, 6) = t217;
    hessian(4, 6) = t103;
    hessian(5, 6) = t26;
    hessian(6, 6) = ((t229 * 2.0 + -t279) + t281) + t102 * (t39 + t40);
    hessian(7, 6) = t213;
    hessian(8, 6) = t311;

    hessian(0, 7) = t303;
    hessian(1, 7) = t216;
    hessian(2, 7) = t227;
    hessian(3, 7) = t27;
    hessian(4, 7) = t218;
    hessian(5, 7) = t225;
    hessian(6, 7) = t213;
    hessian(7, 7) = ((t248 * -2.0 + -t279) + t282) + t102 * (t38 + t40);
    hessian(8, 7) = t36;

    hessian(0, 8) = t304;
    hessian(1, 8) = t214;
    hessian(2, 8) = t294;
    hessian(3, 8) = t28;
    hessian(4, 8) = t226;
    hessian(5, 8) = t104;
    hessian(6, 8) = t311;
    hessian(7, 8) = t36;
    hessian(8, 8) = ((t234 * -2.0 + -t279) + t283) + t102 * (t38 + t39);
    return hessian;
}

static Matrix12r line_line_distance_hessian(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    Matrix12r hessian = Matrix12r::Zero();
    double v01 = ea0(0), v02 = ea0(1), v03 = ea0(2), v11 = ea1(0), v12 = ea1(1), v13 = ea1(2),
           v21 = eb0(0), v22 = eb0(1), v23 = eb0(2), v31 = eb1(0), v32 = eb1(1), v33 = eb1(2);
    double t11, t12, t13, t14, t15, t16, t26, t27, t28, t47, t48, t49, t50, t51, t52, t53, t54, t55,
        t56, t57, t58, t59, t65, t73, t35, t36, t37, t38, t39, t40, t98, t99, t100, t101, t103,
        t105, t107, t108, t109, t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147,
        t148, t156, t159, t157, t262, t263, t264, t265, t266, t267, t268, t269, t270, t271, t272,
        t273, t274, t275, t276, t277, t278, t279, t298, t299, t300, t301, t302, t303, t310, t311,
        t312, t313, t314, t315, t322, t323, t325, t326, t327, t328, t329, t330, t335, t337, t339,
        t340, t341, t342, t343, t345, t348, t353, t356, t358, t359, t360, t362, t367, t368, t369,
        t371, t374, t377, t382, t386, t387, t398, t399, t403, t408, t423, t424, t427, t428, t431,
        t432, t433, t434, t437, t438, t441, t442, t446, t451, t455, t456, t467, t468, t472, t477,
        t491, t492, t495, t497, t499, t500, t503, t504, t506, t508, t550, t568, t519_tmp,
        b_t519_tmp, t519, t520_tmp, b_t520_tmp, t520, t521_tmp, b_t521_tmp, t521, t522_tmp,
        b_t522_tmp, t522, t523_tmp, b_t523_tmp, t523, t524_tmp, b_t524_tmp, t524, t525, t526, t527,
        t528, t529, t530, t531, t532, t533, t534, t535, t536, t537, t538, t539, t540, t542, t543,
        t544;

    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t14 = -v21 + v01;
    t15 = -v22 + v02;
    t16 = -v23 + v03;
    t26 = -v31 + v21;
    t27 = -v32 + v22;
    t28 = -v33 + v23;
    t47 = t11 * t27;
    t48 = t12 * t26;
    t49 = t11 * t28;
    t50 = t13 * t26;
    t51 = t12 * t28;
    t52 = t13 * t27;
    t53 = t14 * t27;
    t54 = t15 * t26;
    t55 = t14 * t28;
    t56 = t16 * t26;
    t57 = t15 * t28;
    t58 = t16 * t27;
    t59 = t11 * t26 * 2.0;
    t65 = t12 * t27 * 2.0;
    t73 = t13 * t28 * 2.0;
    t35 = t11 * t11 * 2.0;
    t36 = t12 * t12 * 2.0;
    t37 = t13 * t13 * 2.0;
    t38 = t26 * t26 * 2.0;
    t39 = t27 * t27 * 2.0;
    t40 = t28 * t28 * 2.0;
    t98 = t11 * t15 + -(t12 * t14);
    t99 = t11 * t16 + -(t13 * t14);
    t100 = t12 * t16 + -(t13 * t15);
    t101 = t47 + -t48;
    t103 = t49 + -t50;
    t105 = t51 + -t52;
    t107 = t53 + -t54;
    t108 = t55 + -t56;
    t109 = t57 + -t58;
    t137 = t98 + t101;
    t138 = t99 + t103;
    t139 = t100 + t105;
    t140 = (t54 + -t53) + t101;
    t141 = (t56 + -t55) + t103;
    t142 = (t58 + -t57) + t105;
    t143 = t12 * t101 * 2.0 + t13 * t103 * 2.0;
    t144 = t11 * t103 * 2.0 + t12 * t105 * 2.0;
    t145 = t27 * t101 * 2.0 + t28 * t103 * 2.0;
    t146 = t26 * t103 * 2.0 + t27 * t105 * 2.0;
    t147 = t11 * t101 * 2.0 + -(t13 * t105 * 2.0);
    t148 = t26 * t101 * 2.0 + -(t28 * t105 * 2.0);
    t156 = 1.0 / ((t101 * t101 + t103 * t103) + t105 * t105);
    t159 = (t16 * t101 + t14 * t105) + -(t15 * t103);
    t157 = t156 * t156;
    t57 = pow(t156, 3.0);
    t58 = t159 * t159;
    t262 = t11 * t156 * t159 * 2.0;
    t263 = t12 * t156 * t159 * 2.0;
    t264 = t13 * t156 * t159 * 2.0;
    t265 = t14 * t156 * t159 * 2.0;
    t266 = t15 * t156 * t159 * 2.0;
    t267 = t16 * t156 * t159 * 2.0;
    t268 = (-v31 + v01) * t156 * t159 * 2.0;
    t269 = (-v21 + v11) * t156 * t159 * 2.0;
    t270 = (-v32 + v02) * t156 * t159 * 2.0;
    t271 = (-v22 + v12) * t156 * t159 * 2.0;
    t272 = (-v33 + v03) * t156 * t159 * 2.0;
    t273 = (-v23 + v13) * t156 * t159 * 2.0;
    t274 = (-v31 + v11) * t156 * t159 * 2.0;
    t275 = (-v32 + v12) * t156 * t159 * 2.0;
    t276 = (-v33 + v13) * t156 * t159 * 2.0;
    t277 = t26 * t156 * t159 * 2.0;
    t278 = t27 * t156 * t159 * 2.0;
    t279 = t28 * t156 * t159 * 2.0;
    t298 = t11 * t12 * t157 * t58 * 2.0;
    t299 = t11 * t13 * t157 * t58 * 2.0;
    t300 = t12 * t13 * t157 * t58 * 2.0;
    t301 = t26 * t27 * t157 * t58 * 2.0;
    t302 = t26 * t28 * t157 * t58 * 2.0;
    t303 = t27 * t28 * t157 * t58 * 2.0;
    t310 = (t35 + t36) * t157 * t58;
    t311 = (t35 + t37) * t157 * t58;
    t312 = (t36 + t37) * t157 * t58;
    t313 = (t38 + t39) * t157 * t58;
    t314 = (t38 + t40) * t157 * t58;
    t315 = (t39 + t40) * t157 * t58;
    t322 = (t59 + t65) * t157 * t58;
    t323 = (t59 + t73) * t157 * t58;
    t59 = (t65 + t73) * t157 * t58;
    t325 = (t47 * 2.0 + -(t48 * 4.0)) * t157 * t58;
    t53 = -t157 * t58;
    t56 = t48 * 2.0 - t47 * 4.0;
    t326 = t53 * t56;
    t327 = (t49 * 2.0 + -(t50 * 4.0)) * t157 * t58;
    t55 = t50 * 2.0 - t49 * 4.0;
    t328 = t53 * t55;
    t329 = (t51 * 2.0 + -(t52 * 4.0)) * t157 * t58;
    t54 = t52 * 2.0 - t51 * 4.0;
    t330 = t53 * t54;
    t53 = t157 * t58;
    t335 = t53 * t56;
    t337 = t53 * t55;
    t339 = t53 * t54;
    t340 = t143 * t143 * t57 * t58 * 2.0;
    t341 = t144 * t144 * t57 * t58 * 2.0;
    t342 = t145 * t145 * t57 * t58 * 2.0;
    t343 = t146 * t146 * t57 * t58 * 2.0;
    t345 = t147 * t147 * t57 * t58 * 2.0;
    t348 = t148 * t148 * t57 * t58 * 2.0;
    t36 = t98 * t143 * t157 * t159 * 2.0;
    t353 = t99 * t143 * t157 * t159 * 2.0;
    t356 = t99 * t144 * t157 * t159 * 2.0;
    t65 = t100 * t144 * t157 * t159 * 2.0;
    t358 = t107 * t143 * t157 * t159 * 2.0;
    t359 = t98 * t145 * t157 * t159 * 2.0;
    t360 = t108 * t143 * t157 * t159 * 2.0;
    t54 = t107 * t144 * t157 * t159 * 2.0;
    t362 = t99 * t145 * t157 * t159 * 2.0;
    t53 = t98 * t146 * t157 * t159 * 2.0;
    t56 = t109 * t143 * t157 * t159 * 2.0;
    t27 = t108 * t144 * t157 * t159 * 2.0;
    t55 = t100 * t145 * t157 * t159 * 2.0;
    t367 = t99 * t146 * t157 * t159 * 2.0;
    t368 = t109 * t144 * t157 * t159 * 2.0;
    t369 = t100 * t146 * t157 * t159 * 2.0;
    t38 = t107 * t145 * t157 * t159 * 2.0;
    t371 = t108 * t145 * t157 * t159 * 2.0;
    t374 = t108 * t146 * t157 * t159 * 2.0;
    t28 = t109 * t146 * t157 * t159 * 2.0;
    t377 = t98 * t147 * t157 * t159 * 2.0;
    t382 = t100 * t147 * t157 * t159 * 2.0;
    t386 = t107 * t147 * t157 * t159 * 2.0;
    t387 = t98 * t148 * t157 * t159 * 2.0;
    t103 = t108 * t147 * t157 * t159 * 2.0;
    t101 = t99 * t148 * t157 * t159 * 2.0;
    t398 = t109 * t147 * t157 * t159 * 2.0;
    t399 = t100 * t148 * t157 * t159 * 2.0;
    t403 = t107 * t148 * t157 * t159 * 2.0;
    t408 = t109 * t148 * t157 * t159 * 2.0;
    t73 = t137 * t143 * t157 * t159 * 2.0;
    t423 = t138 * t143 * t157 * t159 * 2.0;
    t424 = t138 * t144 * t157 * t159 * 2.0;
    t37 = t139 * t144 * t157 * t159 * 2.0;
    t427 = t140 * t143 * t157 * t159 * 2.0;
    t428 = t137 * t145 * t157 * t159 * 2.0;
    t16 = t140 * t144 * t157 * t159 * 2.0;
    t11 = t137 * t146 * t157 * t159 * 2.0;
    t431 = t141 * t143 * t157 * t159 * 2.0;
    t432 = t138 * t145 * t157 * t159 * 2.0;
    t433 = t141 * t144 * t157 * t159 * 2.0;
    t434 = t138 * t146 * t157 * t159 * 2.0;
    t105 = t142 * t143 * t157 * t159 * 2.0;
    t14 = t139 * t145 * t157 * t159 * 2.0;
    t437 = t142 * t144 * t157 * t159 * 2.0;
    t438 = t139 * t146 * t157 * t159 * 2.0;
    t35 = t140 * t145 * t157 * t159 * 2.0;
    t441 = t141 * t145 * t157 * t159 * 2.0;
    t442 = t141 * t146 * t157 * t159 * 2.0;
    t39 = t142 * t146 * t157 * t159 * 2.0;
    t446 = t137 * t147 * t157 * t159 * 2.0;
    t451 = t139 * t147 * t157 * t159 * 2.0;
    t455 = t140 * t147 * t157 * t159 * 2.0;
    t456 = t137 * t148 * t157 * t159 * 2.0;
    t13 = t141 * t147 * t157 * t159 * 2.0;
    t26 = t138 * t148 * t157 * t159 * 2.0;
    t467 = t142 * t147 * t157 * t159 * 2.0;
    t468 = t139 * t148 * t157 * t159 * 2.0;
    t472 = t140 * t148 * t157 * t159 * 2.0;
    t477 = t142 * t148 * t157 * t159 * 2.0;
    t47 = t143 * t144 * t57 * t58 * 2.0;
    t15 = t143 * t145 * t57 * t58 * 2.0;
    t491 = t143 * t146 * t57 * t58 * 2.0;
    t492 = t144 * t145 * t57 * t58 * 2.0;
    t12 = t144 * t146 * t57 * t58 * 2.0;
    t40 = t145 * t146 * t57 * t58 * 2.0;
    t495 = t143 * t147 * t57 * t58 * 2.0;
    t497 = t144 * t147 * t57 * t58 * 2.0;
    t499 = t143 * t148 * t57 * t58 * 2.0;
    t500 = t145 * t147 * t57 * t58 * 2.0;
    t503 = t146 * t147 * t57 * t58 * 2.0;
    t504 = t144 * t148 * t57 * t58 * 2.0;
    t506 = t145 * t148 * t57 * t58 * 2.0;
    t508 = t146 * t148 * t57 * t58 * 2.0;
    t57 = t147 * t148 * t57 * t58 * 2.0;
    t550 = ((((t98 * t109 * t156 * 2.0 + -t266) + t337) + t359) + t368) + t492;
    t568 = ((((t108 * t137 * t156 * 2.0 + -t268) + t330) + t27) + t456) + t504;
    t519_tmp = t139 * t143 * t157 * t159;
    b_t519_tmp = t100 * t143 * t157 * t159;
    t519 = (((-(t100 * t139 * t156 * 2.0) + t312) + -t340) + b_t519_tmp * 2.0) + t519_tmp * 2.0;
    t520_tmp = t140 * t146 * t157 * t159;
    b_t520_tmp = t107 * t146 * t157 * t159;
    t520 = (((t107 * t140 * t156 * 2.0 + t313) + -t343) + b_t520_tmp * 2.0) + -(t520_tmp * 2.0);
    t521_tmp = t142 * t145 * t157 * t159;
    b_t521_tmp = t109 * t145 * t157 * t159;
    t521 = (((t109 * t142 * t156 * 2.0 + t315) + -t342) + -(b_t521_tmp * 2.0)) + t521_tmp * 2.0;
    t522_tmp = t137 * t144 * t157 * t159;
    b_t522_tmp = t98 * t144 * t157 * t159;
    t522 =
        (((-(t98 * t137 * t156 * 2.0) + t310) + -t341) + -(b_t522_tmp * 2.0)) + -(t522_tmp * 2.0);
    t523_tmp = t138 * t147 * t157 * t159;
    b_t523_tmp = t99 * t147 * t157 * t159;
    t523 = (((-(t99 * t138 * t156 * 2.0) + t311) + -t345) + b_t523_tmp * 2.0) + t523_tmp * 2.0;
    t524_tmp = t141 * t148 * t157 * t159;
    b_t524_tmp = t108 * t148 * t157 * t159;
    t524 = (((t108 * t141 * t156 * 2.0 + t314) + -t348) + -(b_t524_tmp * 2.0)) + t524_tmp * 2.0;
    t525 = (((t98 * t100 * t156 * 2.0 + t299) + t65) + -t36) + -t47;
    t526 = (((t107 * t109 * t156 * 2.0 + t302) + t38) + -t28) + -t40;
    t527 = (((-(t98 * t99 * t156 * 2.0) + t300) + t377) + -t356) + t497;
    t528 = (((-(t99 * t100 * t156 * 2.0) + t298) + t353) + t382) + -t495;
    t529 = (((-(t107 * t108 * t156 * 2.0) + t303) + t374) + -t403) + t508;
    t530 = (((-(t108 * t109 * t156 * 2.0) + t301) + -t371) + -t408) + -t506;
    t531 = (((t98 * t107 * t156 * 2.0 + t322) + t54) + -t53) + -t12;
    t532 = (((t100 * t109 * t156 * 2.0 + t59) + t55) + -t56) + -t15;
    t533 = (((t99 * t108 * t156 * 2.0 + t323) + t101) + -t103) + -t57;
    t534 = (((t98 * t140 * t156 * 2.0 + -t322) + t53) + t16) + t12;
    t535 = (((-(t107 * t137 * t156 * 2.0) + -t322) + -t54) + t11) + t12;
    t536 = (((t100 * t142 * t156 * 2.0 + -t59) + -t55) + -t105) + t15;
    t537 = (((-(t109 * t139 * t156 * 2.0) + -t59) + t56) + -t14) + t15;
    t538 = (((t99 * t141 * t156 * 2.0 + -t323) + -t101) + -t13) + t57;
    t539 = (((-(t108 * t138 * t156 * 2.0) + -t323) + t103) + -t26) + t57;
    t540 = (((t137 * t139 * t156 * 2.0 + t299) + t37) + -t73) + -t47;
    t148 = (((t140 * t142 * t156 * 2.0 + t302) + t39) + -t35) + -t40;
    t542 = (((-(t137 * t138 * t156 * 2.0) + t300) + t446) + -t424) + t497;
    t543 = (((-(t138 * t139 * t156 * 2.0) + t298) + t423) + t451) + -t495;
    t544 = (((-(t140 * t141 * t156 * 2.0) + t303) + t472) + -t442) + t508;
    t53 = (((-(t141 * t142 * t156 * 2.0) + t301) + t441) + t477) + -t506;
    t157 = (((-(t139 * t142 * t156 * 2.0) + t59) + t105) + t14) + -t15;
    t159 = (((-(t137 * t140 * t156 * 2.0) + t322) + -t16) + -t11) + -t12;
    t147 = (((-(t138 * t141 * t156 * 2.0) + t323) + t13) + t26) + -t57;
    t146 = ((((t100 * t107 * t156 * 2.0 + t266) + t327) + -t358) + -t369) + t491;
    t145 = ((((-(t99 * t107 * t156 * 2.0) + -t265) + t329) + t367) + t386) + -t503;
    t144 = ((((-(t100 * t108 * t156 * 2.0) + -t267) + t325) + t360) + -t399) + t499;
    t143 = ((((-(t99 * t109 * t156 * 2.0) + t267) + t335) + -t362) + t398) + t500;
    t52 = ((((-(t98 * t108 * t156 * 2.0) + t265) + t339) + -t27) + -t387) + -t504;
    t51 = ((((t109 * t140 * t156 * 2.0 + -t278) + -t302) + t28) + t35) + t40;
    t50 = ((((-(t98 * t139 * t156 * 2.0) + t263) + -t299) + t36) + -t37) + t47;
    t49 = ((((t107 * t142 * t156 * 2.0 + t278) + -t302) + -t38) + -t39) + t40;
    t48 = ((((-(t100 * t137 * t156 * 2.0) + -t263) + -t299) + -t65) + t73) + t47;
    t47 = ((((t99 * t137 * t156 * 2.0 + t262) + -t300) + t356) + -t446) + -t497;
    t73 = ((((t100 * t138 * t156 * 2.0 + t264) + -t298) + -t382) + -t423) + t495;
    t65 = ((((-(t109 * t141 * t156 * 2.0) + t279) + -t301) + t408) + -t441) + t506;
    t59 = ((((t98 * t138 * t156 * 2.0 + -t262) + -t300) + -t377) + t424) + -t497;
    t40 = ((((t99 * t139 * t156 * 2.0 + -t264) + -t298) + -t353) + -t451) + t495;
    t39 = ((((-(t107 * t141 * t156 * 2.0) + -t277) + -t303) + t403) + t442) + -t508;
    t38 = ((((-(t108 * t142 * t156 * 2.0) + -t279) + -t301) + t371) + -t477) + t506;
    t37 = ((((-(t108 * t140 * t156 * 2.0) + t277) + -t303) + -t374) + -t472) + -t508;
    t36 = ((((t98 * t142 * t156 * 2.0 + t271) + t328) + -t359) + t437) + -t492;
    t35 = ((((-(t109 * t137 * t156 * 2.0) + t270) + t328) + -t368) + -t428) + -t492;
    t28 = ((((t100 * t140 * t156 * 2.0 + -t271) + -t327) + t369) + -t427) + -t491;
    t27 = ((((-(t98 * t141 * t156 * 2.0) + -t269) + t330) + t387) + -t433) + t504;
    t26 = ((((t109 * t138 * t156 * 2.0 + -t272) + t326) + -t398) + t432) + -t500;
    t13 = ((((-(t107 * t139 * t156 * 2.0) + -t270) + -t327) + t358) + t438) + -t491;
    t12 = ((((-(t99 * t142 * t156 * 2.0) + -t273) + t326) + t362) + t467) + -t500;
    t11 = ((((-(t99 * t140 * t156 * 2.0) + t269) + -t329) + -t367) + t455) + t503;
    t16 = ((((t107 * t138 * t156 * 2.0 + t268) + -t329) + -t386) + -t434) + t503;
    t15 = ((((-(t100 * t141 * t156 * 2.0) + t273) + -t325) + t399) + t431) + -t499;
    t14 = ((((t108 * t139 * t156 * 2.0 + t272) + -t325) + -t360) + t468) + -t499;
    t105 = ((((-(t139 * t140 * t156 * 2.0) + t275) + t327) + t427) + -t438) + t491;
    t103 = ((((t138 * t140 * t156 * 2.0 + -t274) + t329) + t434) + -t455) + -t503;
    t101 = ((((-(t137 * t142 * t156 * 2.0) + -t275) + t337) + t428) + -t437) + t492;
    t58 = ((((t139 * t141 * t156 * 2.0 + -t276) + t325) + -t431) + -t468) + t499;
    t57 = ((((t137 * t141 * t156 * 2.0 + t274) + t339) + t433) + -t456) + -t504;
    t56 = ((((t138 * t142 * t156 * 2.0 + t276) + t335) + -t432) + -t467) + t500;
    t55 = -t315 + t342;

    hessian(0, 0) = (t55 + t142 * t142 * t156 * 2.0) - t521_tmp * 4.0;
    hessian(1, 0) = t53;
    hessian(2, 0) = t148;
    hessian(3, 0) = t521;
    hessian(4, 0) = t38;
    hessian(5, 0) = t49;
    hessian(6, 0) = t157;
    hessian(7, 0) = t56;
    hessian(8, 0) = t101;
    hessian(9, 0) = t536;
    hessian(10, 0) = t12;
    hessian(11, 0) = t36;

    hessian(0, 1) = t53;
    t54 = -t314 + t348;
    hessian(1, 1) = (t54 + t141 * t141 * t156 * 2.0) - t524_tmp * 4.0;
    hessian(2, 1) = t544;
    hessian(3, 1) = t65;
    hessian(4, 1) = t524;
    hessian(5, 1) = t39;
    hessian(6, 1) = t58;
    hessian(7, 1) = t147;
    hessian(8, 1) = t57;
    hessian(9, 1) = t15;
    hessian(10, 1) = t538;
    hessian(11, 1) = t27;

    hessian(0, 2) = t148;
    hessian(1, 2) = t544;
    t53 = -t313 + t343;
    hessian(2, 2) = (t53 + t140 * t140 * t156 * 2.0) + t520_tmp * 4.0;
    hessian(3, 2) = t51;
    hessian(4, 2) = t37;
    hessian(5, 2) = t520;
    hessian(6, 2) = t105;
    hessian(7, 2) = t103;
    hessian(8, 2) = t159;
    hessian(9, 2) = t28;
    hessian(10, 2) = t11;
    hessian(11, 2) = t534;

    hessian(0, 3) = t521;
    hessian(1, 3) = t65;
    hessian(2, 3) = t51;
    hessian(3, 3) = (t55 + t109 * t109 * t156 * 2.0) + b_t521_tmp * 4.0;
    hessian(4, 3) = t530;
    hessian(5, 3) = t526;
    hessian(6, 3) = t537;
    hessian(7, 3) = t26;
    hessian(8, 3) = t35;
    hessian(9, 3) = t532;
    hessian(10, 3) = t143;
    hessian(11, 3) = t550;

    hessian(0, 4) = t38;
    hessian(1, 4) = t524;
    hessian(2, 4) = t37;
    hessian(3, 4) = t530;
    hessian(4, 4) = (t54 + t108 * t108 * t156 * 2.0) + b_t524_tmp * 4.0;
    hessian(5, 4) = t529;
    hessian(6, 4) = t14;
    hessian(7, 4) = t539;
    hessian(8, 4) = t568;
    hessian(9, 4) = t144;
    hessian(10, 4) = t533;
    hessian(11, 4) = t52;

    hessian(0, 5) = t49;
    hessian(1, 5) = t39;
    hessian(2, 5) = t520;
    hessian(3, 5) = t526;
    hessian(4, 5) = t529;
    hessian(5, 5) = (t53 + t107 * t107 * t156 * 2.0) - b_t520_tmp * 4.0;
    hessian(6, 5) = t13;
    hessian(7, 5) = t16;
    hessian(8, 5) = t535;
    hessian(9, 5) = t146;
    hessian(10, 5) = t145;
    hessian(11, 5) = t531;

    hessian(0, 6) = t157;
    hessian(1, 6) = t58;
    hessian(2, 6) = t105;
    hessian(3, 6) = t537;
    hessian(4, 6) = t14;
    hessian(5, 6) = t13;
    t55 = -t312 + t340;
    hessian(6, 6) = (t55 + t139 * t139 * t156 * 2.0) - t519_tmp * 4.0;
    hessian(7, 6) = t543;
    hessian(8, 6) = t540;
    hessian(9, 6) = t519;
    hessian(10, 6) = t40;
    hessian(11, 6) = t50;

    hessian(0, 7) = t56;
    hessian(1, 7) = t147;
    hessian(2, 7) = t103;
    hessian(3, 7) = t26;
    hessian(4, 7) = t539;
    hessian(5, 7) = t16;
    hessian(6, 7) = t543;
    t54 = -t311 + t345;
    hessian(7, 7) = (t54 + t138 * t138 * t156 * 2.0) - t523_tmp * 4.0;
    hessian(8, 7) = t542;
    hessian(9, 7) = t73;
    hessian(10, 7) = t523;
    hessian(11, 7) = t59;

    hessian(0, 8) = t101;
    hessian(1, 8) = t57;
    hessian(2, 8) = t159;
    hessian(3, 8) = t35;
    hessian(4, 8) = t568;
    hessian(5, 8) = t535;
    hessian(6, 8) = t540;
    hessian(7, 8) = t542;
    t53 = -t310 + t341;
    hessian(8, 8) = (t53 + t137 * t137 * t156 * 2.0) + t522_tmp * 4.0;
    hessian(9, 8) = t48;
    hessian(10, 8) = t47;
    hessian(11, 8) = t522;

    hessian(0, 9) = t536;
    hessian(1, 9) = t15;
    hessian(2, 9) = t28;
    hessian(3, 9) = t532;
    hessian(4, 9) = t144;
    hessian(5, 9) = t146;
    hessian(6, 9) = t519;
    hessian(7, 9) = t73;
    hessian(8, 9) = t48;
    hessian(9, 9) = (t55 + t100 * t100 * t156 * 2.0) - b_t519_tmp * 4.0;
    hessian(10, 9) = t528;
    hessian(11, 9) = t525;

    hessian(0, 10) = t12;
    hessian(1, 10) = t538;
    hessian(2, 10) = t11;
    hessian(3, 10) = t143;
    hessian(4, 10) = t533;
    hessian(5, 10) = t145;
    hessian(6, 10) = t40;
    hessian(7, 10) = t523;
    hessian(8, 10) = t47;
    hessian(9, 10) = t528;
    hessian(10, 10) = (t54 + t99 * t99 * t156 * 2.0) - b_t523_tmp * 4.0;
    hessian(11, 10) = t527;

    hessian(0, 11) = t36;
    hessian(1, 11) = t27;
    hessian(2, 11) = t534;
    hessian(3, 11) = t550;
    hessian(4, 11) = t52;
    hessian(5, 11) = t531;
    hessian(6, 11) = t50;
    hessian(7, 11) = t59;
    hessian(8, 11) = t522;
    hessian(9, 11) = t525;
    hessian(10, 11) = t527;
    hessian(11, 11) = (t53 + t98 * t98 * t156 * 2.0) + b_t522_tmp * 4.0;
    return hessian;
}

static Matrix12r edge_edge_distance_hessian(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1,
    EdgeEdgeDistType dist_type) {

    if (dist_type == EdgeEdgeDistType::AUTO) {
        dist_type = edge_edge_dist_type(ea0, ea1, eb0, eb1);
    }

    Matrix12r hessian = Matrix12r::Zero();
    switch (dist_type) {
    case EdgeEdgeDistType::EA0_EB0: {
        Matrix6r local_hess = point_point_distance_hessian(ea0, eb0);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.block<3, 3>(0, 6) = local_hess.topRightCorner<3, 3>();
        hessian.block<3, 3>(6, 0) = local_hess.bottomLeftCorner<3, 3>();
        hessian.block<3, 3>(6, 6) = local_hess.bottomRightCorner<3, 3>();
        break;
    }
    case EdgeEdgeDistType::EA0_EB1: {
        Matrix6r local_hess = point_point_distance_hessian(ea0, eb1);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.topRightCorner<3, 3>() = local_hess.topRightCorner<3, 3>();
        hessian.bottomLeftCorner<3, 3>() = local_hess.bottomLeftCorner<3, 3>();
        hessian.bottomRightCorner<3, 3>() = local_hess.bottomRightCorner<3, 3>();
        break;
    }
    case EdgeEdgeDistType::EA1_EB0: {
        Matrix6r local_hess = point_point_distance_hessian(ea1, eb0);
        hessian.block<6, 6>(3, 3) = local_hess;
        break;
    }
    case EdgeEdgeDistType::EA1_EB1: {
        Matrix6r local_hess = point_point_distance_hessian(ea1, eb1);
        hessian.block(3, 3, 3, 3) = local_hess.topLeftCorner<3, 3>();
        hessian.block(3, 9, 3, 3) = local_hess.topRightCorner<3, 3>();
        hessian.block(9, 3, 3, 3) = local_hess.bottomLeftCorner<3, 3>();
        hessian.block(9, 9, 3, 3) = local_hess.bottomRightCorner<3, 3>();
        break;
    }
    case EdgeEdgeDistType::EA0_EB: {
        Matrix9r local_hess = point_line_distance_hessian(ea0, eb0, eb1);
        hessian.block(0, 0, 3, 3) = local_hess.topLeftCorner<3, 3>();
        hessian.block(0, 6, 3, 6) = local_hess.topRightCorner<3, 6>();
        hessian.block(6, 0, 6, 3) = local_hess.bottomLeftCorner<6, 3>();
        hessian.block(6, 6, 6, 6) = local_hess.bottomRightCorner<6, 6>();
        break;
    }
    case EdgeEdgeDistType::EA1_EB: {
        Matrix9r local_hess = point_line_distance_hessian(ea1, eb0, eb1);
        hessian.block(3, 3, 9, 9) = local_hess;
        break;
    }
    case EdgeEdgeDistType::EA_EB0: {
        Matrix9r local_hess = point_line_distance_hessian(eb0, ea0, ea1);
        hessian.block(0, 0, 6, 6) = local_hess.bottomRightCorner<6, 6>();
        hessian.block(6, 0, 3, 6) = local_hess.topRightCorner<3, 6>();
        hessian.block(0, 6, 6, 3) = local_hess.bottomLeftCorner<6, 3>();
        hessian.block(6, 6, 3, 3) = local_hess.topLeftCorner<3, 3>();
        break;
    }
    case EdgeEdgeDistType::EA_EB1: {
        Matrix9r local_hess = point_line_distance_hessian(eb1, ea0, ea1);
        hessian.block(0, 0, 6, 6) = local_hess.bottomRightCorner<6, 6>();
        hessian.block(0, 9, 6, 3) = local_hess.bottomLeftCorner<6, 3>();
        hessian.block(9, 0, 3, 6) = local_hess.topRightCorner<3, 6>();
        hessian.block(9, 9, 3, 3) = local_hess.topLeftCorner<3, 3>();
        break;
    }
    case EdgeEdgeDistType::EA_EB: {
        hessian = line_line_distance_hessian(ea0, ea1, eb0, eb1);
        break;
    }
    default: break;
    }
    return hessian;
}

static Matrix12r point_plane_distance_hessian(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2) {
    Matrix12r hessian = Matrix12r::Zero();
    double v01 = p(0), v02 = p(1), v03 = p(2), v11 = t0(0), v12 = t0(1), v13 = t0(2), v21 = t1(0),
           v22 = t1(1), v23 = t1(2), v31 = t2(0), v32 = t2(1), v33 = t2(2);
    double t11, t12, t13, t18, t20, t22, t23, t24, t25, t26, t27, t28, t65, t66, t67, t68, t69, t70,
        t71, t77, t85, t86, t90, t94, t95, t99, t103, t38, t39, t40, t41, t42, t43, t44, t45, t46,
        t72, t73, t75, t78, t80, t82, t125, t126, t127, t128, t129, t130, t131, t133, t135, t149,
        t150, t151, t189, t190, t191, t192, t193, t194, t195, t196, t197, t198, t199, t200, t202,
        t205, t203, t204, t206, t241, t309, t310, t312, t313, t314, t315, t316, t317, t318, t319,
        t321, t322, t323, t324, t325, t261, t262, t599, t600, t602, t605, t609, t610, t611, t613,
        t615, t616, t621, t622, t623, t625, t645, t646_tmp, t646, t601, t603, t604, t606, t607,
        t608, t612, t614, t617, t618, t619, t620, t624, t626, t627, t628, t629, t630, t631, t632,
        t633, t634, t635, t636, t637, t638;

    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t18 = -v21 + v11;
    t20 = -v22 + v12;
    t22 = -v23 + v13;
    t23 = -v31 + v11;
    t24 = -v32 + v12;
    t25 = -v33 + v13;
    t26 = -v31 + v21;
    t27 = -v32 + v22;
    t28 = -v33 + v23;
    t65 = t18 * t24;
    t66 = t20 * t23;
    t67 = t18 * t25;
    t68 = t22 * t23;
    t69 = t20 * t25;
    t70 = t22 * t24;
    t71 = t18 * t23 * 2.0;
    t77 = t20 * t24 * 2.0;
    t85 = t22 * t25 * 2.0;
    t86 = t18 * t26 * 2.0;
    t90 = t20 * t27 * 2.0;
    t94 = t22 * t28 * 2.0;
    t95 = t23 * t26 * 2.0;
    t99 = t24 * t27 * 2.0;
    t103 = t25 * t28 * 2.0;
    t38 = t18 * t18 * 2.0;
    t39 = t20 * t20 * 2.0;
    t40 = t22 * t22 * 2.0;
    t41 = t23 * t23 * 2.0;
    t42 = t24 * t24 * 2.0;
    t43 = t25 * t25 * 2.0;
    t44 = t26 * t26 * 2.0;
    t45 = t27 * t27 * 2.0;
    t46 = t28 * t28 * 2.0;
    t72 = t65 * 2.0;
    t73 = t66 * 2.0;
    t75 = t67 * 2.0;
    t78 = t68 * 2.0;
    t80 = t69 * 2.0;
    t82 = t70 * 2.0;
    t125 = t11 * t20 + -(t12 * t18);
    t126 = t11 * t22 + -(t13 * t18);
    t127 = t12 * t22 + -(t13 * t20);
    t128 = t11 * t24 + -(t12 * t23);
    t129 = t11 * t25 + -(t13 * t23);
    t130 = t12 * t25 + -(t13 * t24);
    t131 = t65 + -t66;
    t133 = t67 + -t68;
    t135 = t69 + -t70;
    t149 = t131 * t131;
    t150 = t133 * t133;
    t151 = t135 * t135;
    t189 = (t11 * t27 + -(t12 * t26)) + t131;
    t190 = (t11 * t28 + -(t13 * t26)) + t133;
    t191 = (t12 * t28 + -(t13 * t27)) + t135;
    t192 = t20 * t131 * 2.0 + t22 * t133 * 2.0;
    t193 = t18 * t133 * 2.0 + t20 * t135 * 2.0;
    t194 = t24 * t131 * 2.0 + t25 * t133 * 2.0;
    t195 = t23 * t133 * 2.0 + t24 * t135 * 2.0;
    t196 = t27 * t131 * 2.0 + t28 * t133 * 2.0;
    t197 = t26 * t133 * 2.0 + t27 * t135 * 2.0;
    t198 = t18 * t131 * 2.0 + -(t22 * t135 * 2.0);
    t199 = t23 * t131 * 2.0 + -(t25 * t135 * 2.0);
    t200 = t26 * t131 * 2.0 + -(t28 * t135 * 2.0);
    t202 = 1.0 / ((t149 + t150) + t151);
    t205 = (t13 * t131 + t11 * t135) + -(t12 * t133);
    t203 = t202 * t202;
    t204 = pow(t202, 3.0);
    t206 = t205 * t205;
    t241 = t131 * t135 * t202 * 2.0;
    t309 = t11 * t202 * t205 * 2.0;
    t310 = t12 * t202 * t205 * 2.0;
    t13 = t13 * t202 * t205 * 2.0;
    t312 = (-v21 + v01) * t202 * t205 * 2.0;
    t313 = (-v22 + v02) * t202 * t205 * 2.0;
    t314 = (-v23 + v03) * t202 * t205 * 2.0;
    t315 = (-v31 + v01) * t202 * t205 * 2.0;
    t316 = t18 * t202 * t205 * 2.0;
    t317 = (-v32 + v02) * t202 * t205 * 2.0;
    t318 = t20 * t202 * t205 * 2.0;
    t319 = (-v33 + v03) * t202 * t205 * 2.0;
    t11 = t22 * t202 * t205 * 2.0;
    t321 = t23 * t202 * t205 * 2.0;
    t322 = t24 * t202 * t205 * 2.0;
    t323 = t25 * t202 * t205 * 2.0;
    t324 = t26 * t202 * t205 * 2.0;
    t325 = t27 * t202 * t205 * 2.0;
    t12 = t28 * t202 * t205 * 2.0;
    t261 = -(t131 * t133 * t202 * 2.0);
    t262 = -(t133 * t135 * t202 * 2.0);
    t599 = t130 * t135 * t202 * 2.0 + t135 * t194 * t203 * t205 * 2.0;
    t600 = -(t125 * t131 * t202 * 2.0) + t131 * t193 * t203 * t205 * 2.0;
    t602 = t129 * t133 * t202 * 2.0 + t133 * t199 * t203 * t205 * 2.0;
    t605 = -(t131 * t189 * t202 * 2.0) + t131 * t197 * t203 * t205 * 2.0;
    t609 = (t127 * t133 * t202 * 2.0 + -t11) + t133 * t192 * t203 * t205 * 2.0;
    t610 = (t126 * t135 * t202 * 2.0 + t11) + t135 * t198 * t203 * t205 * 2.0;
    t611 = (t130 * t131 * t202 * 2.0 + -t322) + t131 * t194 * t203 * t205 * 2.0;
    t613 = (t126 * t131 * t202 * 2.0 + -t316) + t131 * t198 * t203 * t205 * 2.0;
    t615 = (-(t125 * t135 * t202 * 2.0) + -t318) + t135 * t193 * t203 * t205 * 2.0;
    t616 = (-(t128 * t133 * t202 * 2.0) + -t321) + t133 * t195 * t203 * t205 * 2.0;
    t621 = (t133 * t191 * t202 * 2.0 + -t12) + t133 * t196 * t203 * t205 * 2.0;
    t622 = (t135 * t190 * t202 * 2.0 + t12) + t135 * t200 * t203 * t205 * 2.0;
    t623 = (t131 * t190 * t202 * 2.0 + -t324) + t131 * t200 * t203 * t205 * 2.0;
    t625 = (-(t135 * t189 * t202 * 2.0) + -t325) + t135 * t197 * t203 * t205 * 2.0;
    t645 = ((((t127 * t129 * t202 * 2.0 + -t13) + (t72 + -(t66 * 4.0)) * t203 * t206)
             + t129 * t192 * t203 * t205 * 2.0)
            + t127 * t199 * t203 * t205 * 2.0)
           + t192 * t199 * t204 * t206 * 2.0;
    t646_tmp = t203 * t206;
    t646 = ((((t126 * t130 * t202 * 2.0 + t13) + t646_tmp * (t73 - t65 * 4.0))
             + t126 * t194 * t203 * t205 * 2.0)
            + t130 * t198 * t203 * t205 * 2.0)
           + t194 * t198 * t204 * t206 * 2.0;
    t601 = t128 * t131 * t202 * 2.0 + -(t131 * t195 * t203 * t205 * 2.0);
    t603 = -(t127 * t135 * t202 * 2.0) + -(t135 * t192 * t203 * t205 * 2.0);
    t604 = -(t126 * t133 * t202 * 2.0) + -(t133 * t198 * t203 * t205 * 2.0);
    t606 = -(t135 * t191 * t202 * 2.0) + -(t135 * t196 * t203 * t205 * 2.0);
    t607 = -(t133 * t190 * t202 * 2.0) + -(t133 * t200 * t203 * t205 * 2.0);
    t608 = (t125 * t133 * t202 * 2.0 + t316) + -(t133 * t193 * t203 * t205 * 2.0);
    t612 = (t128 * t135 * t202 * 2.0 + t322) + -(t135 * t195 * t203 * t205 * 2.0);
    t614 = (-(t127 * t131 * t202 * 2.0) + t318) + -(t131 * t192 * t203 * t205 * 2.0);
    t617 = (-(t130 * t133 * t202 * 2.0) + t323) + -(t133 * t194 * t203 * t205 * 2.0);
    t618 = (-(t129 * t131 * t202 * 2.0) + t321) + -(t131 * t199 * t203 * t205 * 2.0);
    t619 = (-(t129 * t135 * t202 * 2.0) + -t323) + -(t135 * t199 * t203 * t205 * 2.0);
    t620 = (t133 * t189 * t202 * 2.0 + t324) + -(t133 * t197 * t203 * t205 * 2.0);
    t624 = (-(t131 * t191 * t202 * 2.0) + t325) + -(t131 * t196 * t203 * t205 * 2.0);
    t626 = (((t125 * t127 * t202 * 2.0 + t18 * t22 * t203 * t206 * 2.0)
             + t125 * t192 * t203 * t205 * 2.0)
            + -(t127 * t193 * t203 * t205 * 2.0))
           + -(t192 * t193 * t204 * t206 * 2.0);
    t627 = (((t128 * t130 * t202 * 2.0 + t23 * t25 * t203 * t206 * 2.0)
             + t128 * t194 * t203 * t205 * 2.0)
            + -(t130 * t195 * t203 * t205 * 2.0))
           + -(t194 * t195 * t204 * t206 * 2.0);
    t628 = (((-(t125 * t126 * t202 * 2.0) + t20 * t22 * t203 * t206 * 2.0)
             + t126 * t193 * t203 * t205 * 2.0)
            + -(t125 * t198 * t203 * t205 * 2.0))
           + t193 * t198 * t204 * t206 * 2.0;
    t629 = (((-(t128 * t129 * t202 * 2.0) + t24 * t25 * t203 * t206 * 2.0)
             + t129 * t195 * t203 * t205 * 2.0)
            + -(t128 * t199 * t203 * t205 * 2.0))
           + t195 * t199 * t204 * t206 * 2.0;
    t630 = (((-(t126 * t127 * t202 * 2.0) + t18 * t20 * t203 * t206 * 2.0)
             + -(t126 * t192 * t203 * t205 * 2.0))
            + -(t127 * t198 * t203 * t205 * 2.0))
           + -(t192 * t198 * t204 * t206 * 2.0);
    t631 = (((-(t129 * t130 * t202 * 2.0) + t23 * t24 * t203 * t206 * 2.0)
             + -(t129 * t194 * t203 * t205 * 2.0))
            + -(t130 * t199 * t203 * t205 * 2.0))
           + -(t194 * t199 * t204 * t206 * 2.0);
    t632 = (((-(t125 * t128 * t202 * 2.0) + (t71 + t77) * t203 * t206)
             + t128 * t193 * t203 * t205 * 2.0)
            + t125 * t195 * t203 * t205 * 2.0)
           + -(t193 * t195 * t204 * t206 * 2.0);
    t633 = (((-(t127 * t130 * t202 * 2.0) + (t77 + t85) * t203 * t206)
             + -(t130 * t192 * t203 * t205 * 2.0))
            + -(t127 * t194 * t203 * t205 * 2.0))
           + -(t192 * t194 * t204 * t206 * 2.0);
    t634 = (((-(t126 * t129 * t202 * 2.0) + (t71 + t85) * t203 * t206)
             + -(t129 * t198 * t203 * t205 * 2.0))
            + -(t126 * t199 * t203 * t205 * 2.0))
           + -(t198 * t199 * t204 * t206 * 2.0);
    t635 = (((t127 * t191 * t202 * 2.0 + -((t90 + t94) * t203 * t206))
             + t127 * t196 * t203 * t205 * 2.0)
            + t191 * t192 * t203 * t205 * 2.0)
           + t192 * t196 * t204 * t206 * 2.0;
    t636 = (((-(t128 * t189 * t202 * 2.0) + (t95 + t99) * t203 * t206)
             + t128 * t197 * t203 * t205 * 2.0)
            + t189 * t195 * t203 * t205 * 2.0)
           + -(t195 * t197 * t204 * t206 * 2.0);
    t637 = (((t125 * t189 * t202 * 2.0 + -((t86 + t90) * t203 * t206))
             + -(t125 * t197 * t203 * t205 * 2.0))
            + -(t189 * t193 * t203 * t205 * 2.0))
           + t193 * t197 * t204 * t206 * 2.0;
    t638 = (((-(t130 * t191 * t202 * 2.0) + (t99 + t103) * t203 * t206)
             + -(t130 * t196 * t203 * t205 * 2.0))
            + -(t191 * t194 * t203 * t205 * 2.0))
           + -(t194 * t196 * t204 * t206 * 2.0);
    t86 = (((t126 * t190 * t202 * 2.0 + -((t86 + t94) * t203 * t206))
            + t126 * t200 * t203 * t205 * 2.0)
           + t190 * t198 * t203 * t205 * 2.0)
          + t198 * t200 * t204 * t206 * 2.0;
    t71 = (((-(t129 * t190 * t202 * 2.0) + (t95 + t103) * t203 * t206)
            + -(t129 * t200 * t203 * t205 * 2.0))
           + -(t190 * t199 * t203 * t205 * 2.0))
          + -(t199 * t200 * t204 * t206 * 2.0);
    t85 = (((t189 * t191 * t202 * 2.0 + t26 * t28 * t203 * t206 * 2.0)
            + t189 * t196 * t203 * t205 * 2.0)
           + -(t191 * t197 * t203 * t205 * 2.0))
          + -(t196 * t197 * t204 * t206 * 2.0);
    t90 = (((-(t189 * t190 * t202 * 2.0) + t27 * t28 * t203 * t206 * 2.0)
            + t190 * t197 * t203 * t205 * 2.0)
           + -(t189 * t200 * t203 * t205 * 2.0))
          + t197 * t200 * t204 * t206 * 2.0;
    t99 = (((-(t190 * t191 * t202 * 2.0) + t26 * t27 * t203 * t206 * 2.0)
            + -(t190 * t196 * t203 * t205 * 2.0))
           + -(t191 * t200 * t203 * t205 * 2.0))
          + -(t196 * t200 * t204 * t206 * 2.0);
    t77 = ((((-(t127 * t128 * t202 * 2.0) + t310) + (t75 + -(t68 * 4.0)) * t203 * t206)
            + t127 * t195 * t203 * t205 * 2.0)
           + -(t128 * t192 * t203 * t205 * 2.0))
          + t192 * t195 * t204 * t206 * 2.0;
    t131 = ((((t126 * t128 * t202 * 2.0 + -t309) + (t80 + -(t70 * 4.0)) * t203 * t206)
             + t128 * t198 * t203 * t205 * 2.0)
            + -(t126 * t195 * t203 * t205 * 2.0))
           + -(t195 * t198 * t204 * t206 * 2.0);
    t133 = ((((-(t125 * t130 * t202 * 2.0) + -t310) + t646_tmp * (t78 - t67 * 4.0))
             + t130 * t193 * t203 * t205 * 2.0)
            + -(t125 * t194 * t203 * t205 * 2.0))
           + t193 * t194 * t204 * t206 * 2.0;
    t325 = ((((t125 * t129 * t202 * 2.0 + t309) + t646_tmp * (t82 - t69 * 4.0))
             + t125 * t199 * t203 * t205 * 2.0)
            + -(t129 * t193 * t203 * t205 * 2.0))
           + -(t193 * t199 * t204 * t206 * 2.0);
    t135 = ((((t125 * t191 * t202 * 2.0 + t313) + ((t75 + t18 * t28 * 2.0) + -t78) * t203 * t206)
             + t125 * t196 * t203 * t205 * 2.0)
            + -(t191 * t193 * t203 * t205 * 2.0))
           + -(t193 * t196 * t204 * t206 * 2.0);
    t324 = ((((t127 * t189 * t202 * 2.0 + -t313) + ((t78 + t22 * t26 * 2.0) + -t75) * t203 * t206)
             + -(t127 * t197 * t203 * t205 * 2.0))
            + t189 * t192 * t203 * t205 * 2.0)
           + -(t192 * t197 * t204 * t206 * 2.0);
    t318 = ((((-(t126 * t189 * t202 * 2.0) + t312) + ((t82 + t22 * t27 * 2.0) + -t80) * t203 * t206)
             + t126 * t197 * t203 * t205 * 2.0)
            + -(t189 * t198 * t203 * t205 * 2.0))
           + t197 * t198 * t204 * t206 * 2.0;
    t321 =
        ((((-(t130 * t189 * t202 * 2.0) + t317) + -(((t78 + t25 * t26 * 2.0) + -t75) * t203 * t206))
          + t130 * t197 * t203 * t205 * 2.0)
         + -(t189 * t194 * t203 * t205 * 2.0))
        + t194 * t197 * t204 * t206 * 2.0;
    t323 = ((((t129 * t191 * t202 * 2.0 + t319) + -(((t72 + t23 * t27 * 2.0) + -t73) * t203 * t206))
             + t129 * t196 * t203 * t205 * 2.0)
            + t191 * t199 * t203 * t205 * 2.0)
           + t196 * t199 * t204 * t206 * 2.0;
    t322 =
        ((((-(t125 * t190 * t202 * 2.0) + -t312) + ((t80 + t20 * t28 * 2.0) + -t82) * t203 * t206)
          + -(t125 * t200 * t203 * t205 * 2.0))
         + t190 * t193 * t203 * t205 * 2.0)
        + t193 * t200 * t204 * t206 * 2.0;
    t316 =
        ((((t130 * t190 * t202 * 2.0 + -t319) + -(((t73 + t24 * t26 * 2.0) + -t72) * t203 * t206))
          + t130 * t200 * t203 * t205 * 2.0)
         + t190 * t194 * t203 * t205 * 2.0)
        + t194 * t200 * t204 * t206 * 2.0;
    t65 = ((((-(t128 * t191 * t202 * 2.0) + -t317)
             + -(((t75 + t23 * t28 * 2.0) + -t78) * t203 * t206))
            + -(t128 * t196 * t203 * t205 * 2.0))
           + t191 * t195 * t203 * t205 * 2.0)
          + t195 * t196 * t204 * t206 * 2.0;
    t66 = ((((-(t127 * t190 * t202 * 2.0) + t314) + ((t73 + t20 * t26 * 2.0) + -t72) * t203 * t206)
            + -(t127 * t200 * t203 * t205 * 2.0))
           + -(t190 * t192 * t203 * t205 * 2.0))
          + -(t192 * t200 * t204 * t206 * 2.0);
    t13 = ((((t128 * t190 * t202 * 2.0 + t315) + -(((t80 + t24 * t28 * 2.0) + -t82) * t203 * t206))
            + t128 * t200 * t203 * t205 * 2.0)
           + -(t190 * t195 * t203 * t205 * 2.0))
          + -(t195 * t200 * t204 * t206 * 2.0);
    t12 = ((((-(t126 * t191 * t202 * 2.0) + -t314) + ((t72 + t18 * t27 * 2.0) + -t73) * t203 * t206)
            + -(t126 * t196 * t203 * t205 * 2.0))
           + -(t191 * t198 * t203 * t205 * 2.0))
          + -(t196 * t198 * t204 * t206 * 2.0);
    t11 = ((((t129 * t189 * t202 * 2.0 + -t315) + -(((t82 + t25 * t27 * 2.0) + -t80) * t203 * t206))
            + -(t129 * t197 * t203 * t205 * 2.0))
           + t189 * t199 * t203 * t205 * 2.0)
          + -(t197 * t199 * t204 * t206 * 2.0);
    hessian(0, 0) = t151 * t202 * 2.0;
    hessian(1, 0) = t262;
    hessian(2, 0) = t241;
    hessian(3, 0) = t606;
    hessian(4, 0) = t622;
    hessian(5, 0) = t625;
    hessian(6, 0) = t599;
    hessian(7, 0) = t619;
    hessian(8, 0) = t612;
    hessian(9, 0) = t603;
    hessian(10, 0) = t610;
    hessian(11, 0) = t615;

    hessian(0, 1) = t262;
    hessian(1, 1) = t150 * t202 * 2.0;
    hessian(2, 1) = t261;
    hessian(3, 1) = t621;
    hessian(4, 1) = t607;
    hessian(5, 1) = t620;
    hessian(6, 1) = t617;
    hessian(7, 1) = t602;
    hessian(8, 1) = t616;
    hessian(9, 1) = t609;
    hessian(10, 1) = t604;
    hessian(11, 1) = t608;

    hessian(0, 2) = t241;
    hessian(1, 2) = t261;
    hessian(2, 2) = t149 * t202 * 2.0;
    hessian(3, 2) = t624;
    hessian(4, 2) = t623;
    hessian(5, 2) = t605;
    hessian(6, 2) = t611;
    hessian(7, 2) = t618;
    hessian(8, 2) = t601;
    hessian(9, 2) = t614;
    hessian(10, 2) = t613;
    hessian(11, 2) = t600;

    hessian(0, 3) = t606;
    hessian(1, 3) = t621;
    hessian(2, 3) = t624;
    hessian(3, 3) =
        ((t191 * t191 * t202 * 2.0 + t196 * t196 * t204 * t206 * 2.0) - t646_tmp * (t45 + t46))
        + t191 * t196 * t203 * t205 * 4.0;
    hessian(4, 3) = t99;
    hessian(5, 3) = t85;
    hessian(6, 3) = t638;
    hessian(7, 3) = t323;
    hessian(8, 3) = t65;
    hessian(9, 3) = t635;
    hessian(10, 3) = t12;
    hessian(11, 3) = t135;

    hessian(0, 4) = t622;
    hessian(1, 4) = t607;
    hessian(2, 4) = t623;
    hessian(3, 4) = t99;
    hessian(4, 4) =
        ((t190 * t190 * t202 * 2.0 + t200 * t200 * t204 * t206 * 2.0) - t646_tmp * (t44 + t46))
        + t190 * t200 * t203 * t205 * 4.0;
    hessian(5, 4) = t90;
    hessian(6, 4) = t316;
    hessian(7, 4) = t71;
    hessian(8, 4) = t13;
    hessian(9, 4) = t66;
    hessian(10, 4) = t86;
    hessian(11, 4) = t322;

    hessian(0, 5) = t625;
    hessian(1, 5) = t620;
    hessian(2, 5) = t605;
    hessian(3, 5) = t85;
    hessian(4, 5) = t90;
    hessian(5, 5) =
        ((t189 * t189 * t202 * 2.0 + t197 * t197 * t204 * t206 * 2.0) - t646_tmp * (t44 + t45))
        - t189 * t197 * t203 * t205 * 4.0;
    hessian(6, 5) = t321;
    hessian(7, 5) = t11;
    hessian(8, 5) = t636;
    hessian(9, 5) = t324;
    hessian(10, 5) = t318;
    hessian(11, 5) = t637;

    hessian(0, 6) = t599;
    hessian(1, 6) = t617;
    hessian(2, 6) = t611;
    hessian(3, 6) = t638;
    hessian(4, 6) = t316;
    hessian(5, 6) = t321;
    hessian(6, 6) =
        ((t130 * t130 * t202 * 2.0 + t194 * t194 * t204 * t206 * 2.0) - t646_tmp * (t42 + t43))
        + t130 * t194 * t203 * t205 * 4.0;
    hessian(7, 6) = t631;
    hessian(8, 6) = t627;
    hessian(9, 6) = t633;
    hessian(10, 6) = t646;
    hessian(11, 6) = t133;

    hessian(0, 7) = t619;
    hessian(1, 7) = t602;
    hessian(2, 7) = t618;
    hessian(3, 7) = t323;
    hessian(4, 7) = t71;
    hessian(5, 7) = t11;
    hessian(6, 7) = t631;
    hessian(7, 7) =
        ((t129 * t129 * t202 * 2.0 + t199 * t199 * t204 * t206 * 2.0) - t646_tmp * (t41 + t43))
        + t129 * t199 * t203 * t205 * 4.0;
    hessian(8, 7) = t629;
    hessian(9, 7) = t645;
    hessian(10, 7) = t634;
    hessian(11, 7) = t325;

    hessian(0, 8) = t612;
    hessian(1, 8) = t616;
    hessian(2, 8) = t601;
    hessian(3, 8) = t65;
    hessian(4, 8) = t13;
    hessian(5, 8) = t636;
    hessian(6, 8) = t627;
    hessian(7, 8) = t629;
    hessian(8, 8) =
        ((t128 * t128 * t202 * 2.0 + t195 * t195 * t204 * t206 * 2.0) - t646_tmp * (t41 + t42))
        - t128 * t195 * t203 * t205 * 4.0;
    hessian(9, 8) = t77;
    hessian(10, 8) = t131;
    hessian(11, 8) = t632;

    hessian(0, 9) = t603;
    hessian(1, 9) = t609;
    hessian(2, 9) = t614;
    hessian(3, 9) = t635;
    hessian(4, 9) = t66;
    hessian(5, 9) = t324;
    hessian(6, 9) = t633;
    hessian(7, 9) = t645;
    hessian(8, 9) = t77;
    hessian(9, 9) =
        ((t127 * t127 * t202 * 2.0 + t192 * t192 * t204 * t206 * 2.0) - t646_tmp * (t39 + t40))
        + t127 * t192 * t203 * t205 * 4.0;
    hessian(10, 9) = t630;
    hessian(11, 9) = t626;

    hessian(0, 10) = t610;
    hessian(1, 10) = t604;
    hessian(2, 10) = t613;
    hessian(3, 10) = t12;
    hessian(4, 10) = t86;
    hessian(5, 10) = t318;
    hessian(6, 10) = t646;
    hessian(7, 10) = t634;
    hessian(8, 10) = t131;
    hessian(9, 10) = t630;
    hessian(10, 10) =
        ((t126 * t126 * t202 * 2.0 + t198 * t198 * t204 * t206 * 2.0) - t646_tmp * (t38 + t40))
        + t126 * t198 * t203 * t205 * 4.0;
    hessian(11, 10) = t628;

    hessian(0, 11) = t615;
    hessian(1, 11) = t608;
    hessian(2, 11) = t600;
    hessian(3, 11) = t135;
    hessian(4, 11) = t322;
    hessian(5, 11) = t637;
    hessian(6, 11) = t133;
    hessian(7, 11) = t325;
    hessian(8, 11) = t632;
    hessian(9, 11) = t626;
    hessian(10, 11) = t628;
    hessian(11, 11) =
        ((t125 * t125 * t202 * 2.0 + t193 * t193 * t204 * t206 * 2.0) - t646_tmp * (t38 + t39))
        - t125 * t193 * t203 * t205 * 4.0;
    return hessian;
}

static Matrix12r point_triangle_distance_hessian(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2,
    PointTriangleDistType dist_type) {
    Matrix12r hessian = Matrix12r::Zero();
    if (dist_type == PointTriangleDistType::AUTO) {
        dist_type = point_triangle_dist_type(p, t0, t1, t2);
    }

    switch (dist_type) {
    case PointTriangleDistType::P_T0: {
        Matrix6r local_hess = point_point_distance_hessian(p, t0);
        hessian.topLeftCorner<6, 6>() = local_hess;
        break;
    }
    case PointTriangleDistType::P_T1: {
        Matrix6r local_hess = point_point_distance_hessian(p, t1);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.block<3, 3>(0, 6) = local_hess.topRightCorner<3, 3>();
        hessian.block<3, 3>(6, 0) = local_hess.bottomLeftCorner<3, 3>();
        hessian.block<3, 3>(6, 6) = local_hess.bottomRightCorner<3, 3>();
        break;
    }
    case PointTriangleDistType::P_T2: {
        Matrix6r local_hess = point_point_distance_hessian(p, t2);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.topRightCorner<3, 3>() = local_hess.topRightCorner<3, 3>();
        hessian.bottomLeftCorner<3, 3>() = local_hess.bottomLeftCorner<3, 3>();
        hessian.bottomRightCorner<3, 3>() = local_hess.bottomRightCorner<3, 3>();
        break;
    }
    case PointTriangleDistType::P_T0T1: {
        Matrix9r local_hess = point_line_distance_hessian(p, t0, t1);
        hessian.topLeftCorner<9, 9>() = local_hess;
        break;
    }
    case PointTriangleDistType::P_T1T2: {
        Matrix9r local_hess = point_line_distance_hessian(p, t1, t2);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.topRightCorner<3, 6>() = local_hess.topRightCorner<3, 6>();
        hessian.bottomLeftCorner<6, 3>() = local_hess.bottomLeftCorner<6, 3>();
        hessian.bottomRightCorner<6, 6>() = local_hess.bottomRightCorner<6, 6>();
        break;
    }
    case PointTriangleDistType::P_T2T0: {
        Matrix9r local_hess = point_line_distance_hessian(p, t2, t0);
        hessian.topLeftCorner<3, 3>() = local_hess.topLeftCorner<3, 3>();
        hessian.block<3, 3>(0, 3) = local_hess.topRightCorner<3, 3>();
        hessian.topRightCorner<3, 3>() = local_hess.block<3, 3>(0, 3);
        hessian.block<3, 3>(3, 0) = local_hess.bottomLeftCorner<3, 3>();
        hessian.block<3, 3>(3, 3) = local_hess.bottomRightCorner<3, 3>();
        hessian.block<3, 3>(3, 9) = local_hess.block<3, 3>(6, 3);
        hessian.bottomLeftCorner<3, 3>() = local_hess.block<3, 3>(3, 0);
        hessian.block<3, 3>(9, 3) = local_hess.block<3, 3>(3, 6);
        hessian.bottomRightCorner<3, 3>() = local_hess.block<3, 3>(3, 3);
        break;
    }
    case PointTriangleDistType::P_T: {
        hessian = point_plane_distance_hessian(p, t0, t1, t2);
        break;
    }

    default: break;
    }
    return hessian;
}
} // namespace cipc