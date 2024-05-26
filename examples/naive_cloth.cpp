/* Naive cloth simulator with dihedral angle based isotropic model*/

#include "c-ipc/c-ipc.h"
#include <spdlog/stopwatch.h>

#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cipc;

int main() {
    spdlog::set_pattern("[%m-%d %T] %^[%l]%$ %v");
    const std::string output_dir("./output/naive_cloth");
    const std::string objfile_dir("./obj_files/");
    const std::string obj_file("./obj_files/mat20x30.obj");
    // print information
    spdlog::info("*** Naive Cloth Simulation without C-IPC **");
    spdlog::info("Output folder : {}", std::filesystem::absolute(output_dir).string());

    Mesh clothmesh;
    load_obj(obj_file, std::filesystem::absolute(objfile_dir).string(), clothmesh);

    spdlog::info(
        "Load mesh with {} vertices and {} elements", clothmesh.vertices_num, clothmesh.elem_num);
    
    Simulator<NaiveStvK> sim(clothmesh.vertices, clothmesh.indices, 0.1f, 0.1f, 1.0f);
    
    Matrix3Xr curr_pos = sim.get_position();
    // TO DO: set gravity
    for (integer i = 0; i < static_cast<integer>(curr_pos.cols()); i++){
        curr_pos.col(i)(2) = 2.0f;
    }
    sim.set_position(curr_pos);
    sim.Forward(0.001f);

    return 0;
}