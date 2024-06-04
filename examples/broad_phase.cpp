/* Naive cloth simulator with dihedral angle based isotropic model*/

#include "c-ipc/c-ipc.h"
#include <spdlog/stopwatch.h>

#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cipc;

int main() {
    spdlog::set_pattern("[%m-%d %T] %^[%l]%$ %v");
    const std::string output_dir("./output/broad_phase");
    const std::string objfiles_dir("./obj_files");
    const std::string obj_file("mat2x2.obj");
    // print information
    spdlog::info("** Broad Phase Example **");
    if (!std::filesystem::exists(output_dir)) { std::filesystem::create_directories(output_dir); }
    spdlog::info("Output folder : {}", std::filesystem::absolute(output_dir).string());

    spdlog::stopwatch sw;
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", objfiles_dir, obj_file);
    load_obj(filepath, clothmesh);

    Matrix2Xi edges = Matrix2Xi::Zero(2, 5);

    edges.col(0) = Vector2i(0, 3);
    edges.col(1) = Vector2i(0, 1);
    edges.col(2) = Vector2i(1, 3);
    edges.col(3) = Vector2i(2, 3);
    edges.col(4) = Vector2i(1, 2);

    double dhat = 1e-3;
    double dmin = 1e-4;

    BarrierPotential B;
    B.build(clothmesh.vertices, clothmesh.rest_vertices, edges, clothmesh.indices, dhat, dmin);
    double energy = B.ComputeBarrierPotential(clothmesh.vertices, edges, clothmesh.indices, dmin);
    Matrix3Xr gradient =
        B.ComputeBarrierGradient(clothmesh.vertices, edges, clothmesh.indices, dmin);
    SparseMatrixXr hessian =
        B.ComputeBarrierHessian(clothmesh.vertices, edges, clothmesh.indices, dmin);

    spdlog::info("Frame 0, {:.2f}s elapsed", sw);

    return 0;
}