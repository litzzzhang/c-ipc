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
    edges.col(1) = Vector2i(0, 2);
    edges.col(2) = Vector2i(1, 3);
    edges.col(3) = Vector2i(2, 3);
    edges.col(4) = Vector2i(1, 2);

    ConstrainSet collision_cadidates;

    collision_cadidates.build(clothmesh.vertices, edges, clothmesh.indices, 1e-3);

    spdlog::info("Frame 0, {:.2f}s elapsed", sw);

    return 0;
}