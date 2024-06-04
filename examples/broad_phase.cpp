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


    clothmesh.vertices.row(1) *= 0.1;

    ConstrainSet c;
    double t = c.compute_accd_timestep(clothmesh.rest_vertices, clothmesh.vertices, clothmesh.edges, clothmesh.indices, 1e-3);

    double dhat = 1e-3;
    double dmin = 1e-4;

    BarrierPotential B;
    B.build(clothmesh.vertices, clothmesh.rest_vertices, clothmesh.edges, clothmesh.indices, dhat, dmin);
    double energy = B.ComputeBarrierPotential(clothmesh.vertices, clothmesh.edges, clothmesh.indices, dmin);
    Matrix3Xr gradient =
        B.ComputeBarrierGradient(clothmesh.vertices, clothmesh.edges, clothmesh.indices, dmin);
    SparseMatrixXr hessian =
        B.ComputeBarrierHessian(clothmesh.vertices, clothmesh.edges, clothmesh.indices, dmin);

    std::cout << gradient;
    spdlog::info("Frame 0, {:.2f}s elapsed", sw);

    return 0;
}