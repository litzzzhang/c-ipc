#pragma once



namespace cipc {
class Material {
    Material();
    // virtual ~Material() {}

    // energy
    virtual const real ComputeEnergyDensity(const Matrix3x2r &F) const = 0;
    virtual const real ComputeStretchingEnergy(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements,
        std::vector<Matrix3x2r> &D_inv) const = 0;

    // grad and force
    virtual const Matrix3x2r ComputeStressTensor(const Matrix3x2r &F) const = 0;
    virtual const Matrix3Xr ComputeStretchingForce(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements, std::vector<Matrix3x2r> &D_inv,
        std::vector<std::vector<std::array<integer, 2>>> &gradient_map) const = 0;

    // hessian
    virtual const Matrix9r ComputeEnergyDensityHessian(const Matrix3x2r &F) const = 0;
    virtual const SparseMatrixXr ComputeStretchingHessian(
        const Matrix3Xr &curr_pos, const Matrix3Xi &elements, std::vector<Matrix3x2r> &D_inv,
        std::vector<std::vector<std::array<integer, 3>>> &hessian_map,
        SparseMatrixXr &hessian_prev) const = 0;

    // virtual const std::pair<Matrix9r, Matrix9r>
    // ComputeEnergyDensityHessian_SPD(const Matrix3x2r &F) const = 0;

};

} // namespace cipc