
// distance type deduction
PointEdgeDistType point_edge_dist_type(const Vector3r &p, const Vector3r &e0, const Vector3r &e1) {
    Vector3r edge = e1 - e0;
    const real edge_length_square = edge.squaredNorm();
    if (edge_length_square == 0) {
        return PointEdgeDistType::P_E0;
    }

    const real ratio = edge.dot(p - e0) / edge_length_square;
    if (ratio < 0) {
        return PointEdgeDistType::P_E0;
    } else if (ratio > 1) {
        return PointEdgeDistType::P_E1;
    } else {
        return PointEdgeDistType::P_E;
    }
}

PointTriangleDistType point_triangle_dist_type(
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

EdgeEdgeDistType parallel_edge_dist_type(
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

EdgeEdgeDistType edge_edge_dist_type(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    constexpr double PARALLEL_THRESHOLD = 1.0e-20;
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
    // compute the line parameters of the two closest points
    const double sN = (b * e - c * d);
    double tN, tD; // tc = tN / tD
    if (sN <= 0.0) { // sc < 0 ⟹ the s=0 edge is visible
        tN = e;
        tD = c;
        default_case = EdgeEdgeDistType::EA0_EB;
    } else if (sN >= D) { // sc > 1 ⟹ the s=1 edge is visible
        tN = e + b;
        tD = c;
        default_case = EdgeEdgeDistType::EA1_EB;
    } else {
        tN = (a * e - b * d);
        tD = D; // default tD = D ≥ 0
        if (tN > 0.0 && tN < tD && u.cross(v).squaredNorm() < parallel_tol) {
            // avoid coplanar or nearly parallel EE
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
        // else default_case stays EdgeEdgeDistanceType::EA_EB
    }

    if (tN <= 0.0) { // tc < 0 ⟹ the t=0 edge is visible
        // recompute sc for this edge
        if (-d <= 0.0) {
            return EdgeEdgeDistType::EA0_EB0;
        } else if (-d >= a) {
            return EdgeEdgeDistType::EA1_EB0;
        } else {
            return EdgeEdgeDistType::EA_EB0;
        }
    } else if (tN >= tD) { // tc > 1 ⟹ the t=1 edge is visible
        // recompute sc for this edge
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
real point_point_distance(const Vector3r &p0, const Vector3r &p1) {
    return (p0 - p1).squaredNorm();
}

real point_edge_distance(
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

real point_triangle_distance_in(
    const Vector3r &p, const Vector3r &t0, const Vector3r &t1, const Vector3r &t2) {
    Vector3r normal = (t2 - t0).cross(t1 - t0);
    return (p - t0).dot(normal) / normal.squaredNorm();
}

real point_triangle_distance(
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

real edge_edge_distance_in(
    const Vector3r &ea0, const Vector3r &ea1, const Vector3r &eb0, const Vector3r &eb1) {
    const Vector3r normal = (ea1 - ea0).cross(eb1 - eb0);
    const double line_to_line = (eb0 - ea0).dot(normal);
    return line_to_line * line_to_line / normal.squaredNorm();
}

real edge_edge_distance(
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
    case EdgeEdgeDistType::EA_EB:
        return edge_edge_distance_in(ea0, ea1, eb0, eb1);
    default: printf("Invalid dist type for point triangle\n"); std::abort();
    }
}
