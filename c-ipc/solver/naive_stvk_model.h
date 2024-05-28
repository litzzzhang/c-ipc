#pragma once

/* constitutive model from class homework3 */
#include "c-ipc/solver/eigen.h"
#include "c-ipc/backend/backend.h"

namespace cipc {
class NaiveStvK {
  public:
    const Matrix4r C = Eigen::Matrix<real, 16, 1>{2500., 0., 0.,   2000., 0.,    500., 0., 0.,
                                                  0.,    0., 500., 0.,    2000., 0.,   0., 2500.}
                           .reshaped(4, 4);
    // const real density_;
    // const real youngs_modulus_;
    // const real poisson_ratio_;
    // const real lambda_;
    // const real mu_;

    NaiveStvK() = default;
    // energy
    const real ComputeEnergyDensity(const Matrix3x2r &F) const;
    const real ComputeStretchingEnergy(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements,
        const std::vector<Matrix3x2r> &D_inv) const;

    // grad and force
    const Matrix3x2r ComputeStressTensor(const Matrix3x2r &F) const;
    const Matrix3Xr ComputeStretchingForce(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements, const std::vector<Matrix3x2r> &D_inv,
        const std::vector<std::vector<std::array<integer, 2>>> &gradient_map) const;

    // hessian
    const Matrix9r ComputeEnergyDensityHessian(const Matrix3x2r &F) const;
    const SparseMatrixXr ComputeStretchingHessian(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements, const std::vector<Matrix3x2r> &D_inv,
        const std::vector<std::vector<std::array<integer, 3>>> &hessian_map,
        const SparseMatrixXr &hessian_prev) const;

    // const std::pair<Matrix9r, Matrix9r>
    // ComputeEnergyDensityHessian_SPD(const Matrix3x2r &F) const override;
};

inline const real NaiveStvK::ComputeEnergyDensity(const Matrix3x2r &F) const {
    Matrix2r E = 0.5f * (F.transpose() * F - Matrix2r::Identity());
    return 0.5f * E.reshaped().dot(C * E.reshaped());
}

inline const real NaiveStvK::ComputeStretchingEnergy(
    const Matrix3Xr &curr_pos, const Matrix3Xi &elements,
    const std::vector<Matrix3x2r> &D_inv) const {
    const integer element_num = static_cast<integer>(elements.cols());

    real energy = 0.0f;
    energy = oneapi::tbb::parallel_deterministic_reduce(
        oneapi::tbb::blocked_range<integer>(0, element_num), 0.0,
        [&](oneapi::tbb::blocked_range<integer> r, real local) {
            for (integer e = r.begin(); e < r.end(); e++) {
                const Vector3r v0 = curr_pos.col(elements(0, e));
                const Vector3r v1 = curr_pos.col(elements(1, e));
                const Vector3r v2 = curr_pos.col(elements(2, e));
                const Matrix3x2r F = curr_pos(Eigen::all, elements.col(e)) * D_inv[e];
                real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
                local += element_area * ComputeEnergyDensity(F);
            }
            return local;
        },
        [](real x, real y) {
            return x + y;
        });

        return energy;
}

inline const Matrix3x2r NaiveStvK::ComputeStressTensor(const Matrix3x2r &F) const {
    const Matrix2r E = 0.5f * (F.transpose() * F - Matrix2r::Identity());
    const Vector4r dPsidE = C * E.reshaped();
    Matrix3x2r dPsidF = Matrix3x2r::Zero();
    Vector3r F1 = F.col(0), F2 = F.col(1);
    Eigen::Matrix<real, 6, 4> dEdF = Eigen::Matrix<real, 6, 4>::Zero();
    dEdF.block(0, 0, 3, 1) = F1;
    dEdF.block(0, 1, 3, 1) = 0.5 * F2;
    dEdF.block(0, 2, 3, 1) = 0.5 * F2;
    dEdF.block(3, 1, 3, 1) = 0.5 * F1;
    dEdF.block(3, 2, 3, 1) = 0.5 * F1;
    dEdF.block(3, 3, 3, 1) = F2;
    Vector6r vec_dPsidF = dEdF * dPsidE;
    dPsidF.block(0, 0, 3, 1) = vec_dPsidF(Vector3i(0, 1, 2));
    dPsidF.block(0, 1, 3, 1) = vec_dPsidF(Vector3i(3, 4, 5));
    return dPsidF;
}

inline const Matrix3Xr NaiveStvK::ComputeStretchingForce(
    const Matrix3Xr &curr_pos, const Matrix3Xi &elements, const std::vector<Matrix3x2r> &D_inv,
    const std::vector<std::vector<std::array<integer, 2>>> &gradient_map) const {
    const integer element_num = static_cast<integer>(elements.cols());
    std::vector<Matrix3r> gradient_per_element(element_num);

    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        gradient_per_element[e] = Matrix3r::Zero();
        const Vector3r v0 = curr_pos.col(elements.col(e)(0));
        const Vector3r v1 = curr_pos.col(elements.col(e)(1));
        const Vector3r v2 = curr_pos.col(elements.col(e)(2));
        const Matrix3x2r F = curr_pos(Eigen::all, elements.col(e)) * D_inv[e];
        const Matrix2r E = 0.5f * (F.transpose() * F - Matrix2r::Identity());
        const real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
        const Vector4r C_dot_E = C * E.reshaped();
        Vector4r dEdF[3][2];
        for (integer i = 0; i < 3; i++) {
            dEdF[i][0] << F(i, 0), F(i, 1) / 2, F(i, 1) / 2, 0;
            dEdF[i][1] << 0, F(i, 0) / 2, F(i, 0) / 2, F(i, 1);
        }
        Eigen::Matrix<real, 3, 2> dPsidF;
        dPsidF.setZero();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                for (int m = 0; m < 4; m++) { dPsidF(i, j) += C_dot_E(m) * (dEdF[i][j])(m); }
        // dE/dx = dPsi/dF * dF/dx * area
        gradient_per_element[e] = element_area * dPsidF * D_inv[e].transpose();
    });

    integer vertex_num = static_cast<integer>(curr_pos.cols());
    Matrix3Xr gradient = Matrix3Xr::Zero(3, vertex_num);
    for (integer v = 0; v < vertex_num; v++) {
        for (const auto &tuple : gradient_map[v]) {
            const integer e = tuple[0];
            const integer i = tuple[1];
            gradient.col(v) += gradient_per_element[e].col(i);
        }
    }

    return -gradient;
}

inline const Matrix9r NaiveStvK::ComputeEnergyDensityHessian(const Matrix3x2r &F) const {
    return Matrix9r();
}

inline const SparseMatrixXr NaiveStvK::ComputeStretchingHessian(
    const Matrix3Xr &curr_pos, const Matrix3Xi &elements, const std::vector<Matrix3x2r> &D_inv,
    const std::vector<std::vector<std::array<integer, 3>>> &hessian_map,
    const SparseMatrixXr &hessian_prev) const {
    const integer element_num = static_cast<integer>(elements.cols());
    std::vector<Matrix9r> hess_nonzeros;
    hess_nonzeros.assign(element_num, Matrix9r::Zero());

    oneapi::tbb::parallel_for(0, element_num, [&](integer e) {
        const Vector3r v0 = curr_pos.col(elements.col(e)(0));
        const Vector3r v1 = curr_pos.col(elements.col(e)(1));
        const Vector3r v2 = curr_pos.col(elements.col(e)(2));
        const Matrix3x2r F = curr_pos(Eigen::all, elements.col(e)) * D_inv[e];
        const real element_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
        const Matrix2r E = 0.5f * (F.transpose() * F - Matrix2r::Identity());
        // hess_nonzeros.push_back(Matrix9r::Zero());
        const Vector4r E_vec = E.reshaped();
        const Vector4r C_dot_E = C * E_vec;
        Vector4r dEdF[3][2];
        for (int i = 0; i < 3; i++) {
            dEdF[i][0] << F(i, 0), F(i, 1) / 2, F(i, 1) / 2, 0;
            dEdF[i][1] << 0, F(i, 0) / 2, F(i, 0) / 2, F(i, 1);
        }
        real ddPsiddF[3][2][3][2] = {0};
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 2; b++) {
                for (int c = 0; c < 3; c++) {
                    for (int d = 0; d < 2; d++) {
                        for (int m = 0; m < 4; m++) {
                            if (a == c && b == 0 && d == 0 && m == 0)
                                ddPsiddF[a][b][c][d] += C_dot_E(m);
                            if (a == c && b == 1 && d == 1 && m == 3)
                                ddPsiddF[a][b][c][d] += C_dot_E(m);
                            if (a == c && b != d && (m == 1 || m == 2))
                                ddPsiddF[a][b][c][d] += C_dot_E(m) / 2;
                            for (int n = 0; n < 4; n++) {
                                ddPsiddF[a][b][c][d] += C(m, n) * (dEdF[a][b])(m) * (dEdF[c][d])(n);
                            }
                        }
                    }
                }
            }
        }
        for (int i1 = 0; i1 < 3; i1++) {
            for (int j1 = 0; j1 < 3; j1++) {
                for (int i2 = 0; i2 < 3; i2++) {
                    for (int j2 = 0; j2 < 3; j2++) {
                        for (int a = 0; a < 3; a++)
                            for (int b = 0; b < 2; b++)
                                for (int c = 0; c < 3; c++)
                                    for (int d = 0; d < 2; d++) {
                                        if (a == i1 && c == i2)
                                            hess_nonzeros[e](3 * j1 + i1, 3 * j2 + i2) +=
                                                element_area * D_inv[e](j1, b) * D_inv[e](j2, d)
                                                * ddPsiddF[a][b][c][d];
                                    }
                    }
                }
            }
        }
    });

    SparseMatrixXr ret(hessian_prev);
    integer hessian_nonzero_num = static_cast<integer>(hessian_map.size());
    for (integer v = 0; v < hessian_nonzero_num; v++) {
        real val = 0;
        for (const auto &arr : hessian_map[v]) {
            const integer e = arr[0];
            const integer i = arr[1];
            const integer j = arr[2];
            val += hess_nonzeros[e](i, j);
        }
        ret.valuePtr()[v] = val;
    }
    return ret;
}
} // namespace cipc
