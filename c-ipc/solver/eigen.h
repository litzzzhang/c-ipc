#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace cipc {

// using real = float;
using real = double;
using integer = int32_t;
// alias for basic vector and matrix
using MatrixXr = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
using Matrix2r = Eigen::Matrix<real, 2, 2>;
using Matrix3r = Eigen::Matrix<real, 3, 3>;
using Matrix4r = Eigen::Matrix<real, 4, 4>;
using Matrix6r = Eigen::Matrix<real, 6, 6>;
using Matrix8r = Eigen::Matrix<real, 8, 8>;
using Matrix9r = Eigen::Matrix<real, 9, 9>;
using Matrix10r = Eigen::Matrix<real, 10, 10>;
using Matrix12r = Eigen::Matrix<real, 12, 12>;
using Matrix18r = Eigen::Matrix<real, 18, 18>;
using Matrix27r = Eigen::Matrix<real, 27, 27>;
using Matrix2Xr = Eigen::Matrix<real, 2, Eigen::Dynamic>;
using MatrixX2r = Eigen::Matrix<real, Eigen::Dynamic, 2>;
using Matrix3Xr = Eigen::Matrix<real, 3, Eigen::Dynamic>;
using MatrixX3r = Eigen::Matrix<real, Eigen::Dynamic, 3>;
using Matrix4Xr = Eigen::Matrix<real, 4, Eigen::Dynamic>;
using MatrixX4r = Eigen::Matrix<real, Eigen::Dynamic, 4>;
using Matrix6Xr = Eigen::Matrix<real, 6, Eigen::Dynamic>;
using MatrixX6r = Eigen::Matrix<real, Eigen::Dynamic, 6>;

using Matrix3x2r = Eigen::Matrix<real, 3, 2>;

using SparseMatrixXr = Eigen::SparseMatrix<real>;

using VectorXr = Eigen::Matrix<real, Eigen::Dynamic, 1>;
using Vector1r = Eigen::Matrix<real, 1, 1>;
using Vector2r = Eigen::Matrix<real, 2, 1>;
using Vector3r = Eigen::Matrix<real, 3, 1>;
using Vector4r = Eigen::Matrix<real, 4, 1>;
using Vector5r = Eigen::Matrix<real, 5, 1>;
using Vector6r = Eigen::Matrix<real, 6, 1>;
using Vector7r = Eigen::Matrix<real, 7, 1>;
using Vector8r = Eigen::Matrix<real, 8, 1>;
using Vector9r = Eigen::Matrix<real, 9, 1>;
using Vector10r = Eigen::Matrix<real, 10, 1>;
using Vector18r = Eigen::Matrix<real, 18, 1>;
using Vector12r = Eigen::Matrix<real, 12, 1>;
using Vector27r = Eigen::Matrix<real, 27, 1>;

using RowVectorXr = Eigen::Matrix<real, 1, Eigen::Dynamic>;
using RowVector2r = Eigen::Matrix<real, 1, 2>;
using RowVector3r = Eigen::Matrix<real, 1, 3>;
using RowVector4r = Eigen::Matrix<real, 1, 4>;
using RowVector6r = Eigen::Matrix<real, 1, 6>;

using SparseVectorXr = Eigen::SparseVector<real>;

using MatrixXi = Eigen::Matrix<integer, Eigen::Dynamic, Eigen::Dynamic>;
using Matrix2i = Eigen::Matrix<integer, 2, 2>;
using Matrix3i = Eigen::Matrix<integer, 3, 3>;
using Matrix4i = Eigen::Matrix<integer, 4, 4>;
using Matrix2Xi = Eigen::Matrix<integer, 2, Eigen::Dynamic>;
using MatrixX2i = Eigen::Matrix<integer, Eigen::Dynamic, 2>;
using Matrix3Xi = Eigen::Matrix<integer, 3, Eigen::Dynamic>;
using MatrixX3i = Eigen::Matrix<integer, Eigen::Dynamic, 3>;
using Matrix4Xi = Eigen::Matrix<integer, 4, Eigen::Dynamic>;
using MatrixX4i = Eigen::Matrix<integer, Eigen::Dynamic, 4>;
using Matrix8Xi = Eigen::Matrix<integer, 8, Eigen::Dynamic>;
using MatrixX8i = Eigen::Matrix<integer, Eigen::Dynamic, 8>;

using SparseMatrixXi = Eigen::SparseMatrix<integer>;

using VectorXi = Eigen::Matrix<integer, Eigen::Dynamic, 1>;
using Vector1i = Eigen::Matrix<integer, 1, 1>;
using Vector2i = Eigen::Matrix<integer, 2, 1>;
using Vector3i = Eigen::Matrix<integer, 3, 1>;
using Vector4i = Eigen::Matrix<integer, 4, 1>;
using Vector8i = Eigen::Matrix<integer, 8, 1>;

using RowVectorXi = Eigen::Matrix<integer, 1, Eigen::Dynamic>;
using RowVector2i = Eigen::Matrix<integer, 1, 2>;
using RowVector3i = Eigen::Matrix<integer, 1, 3>;
using RowVector4i = Eigen::Matrix<integer, 1, 4>;

inline const SparseMatrixXr FromTriplet(
    const integer row_num, const integer col_num,
    const std::vector<Eigen::Triplet<real>> &nonzeros) {
    SparseMatrixXr mat(row_num, col_num);
    mat.setFromTriplets(nonzeros.begin(), nonzeros.end());
    mat.makeCompressed();
    return mat;
}

inline const std::vector<Eigen::Triplet<real>> ToTriplet(const SparseMatrixXr &mat) {
    SparseMatrixXr mat_compressed = mat;
    mat_compressed.makeCompressed();
    std::vector<Eigen::Triplet<real>> nonzeros;
    for (integer k = 0; k < static_cast<integer>(mat_compressed.outerSize()); ++k)
        for (SparseMatrixXr::InnerIterator it(mat_compressed, k); it; ++it)
            nonzeros.push_back(Eigen::Triplet<real>((int)it.row(), (int)it.col(), it.value()));
    return nonzeros;
}
} // namespace cipc