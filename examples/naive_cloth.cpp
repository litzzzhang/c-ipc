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
    const std::string objfiles_dir("./obj_files");
    const std::string obj_file("mat10x10.obj");
    // print information
    spdlog::info("*** Naive Cloth Simulation without C-IPC **");
    if (!std::filesystem::exists(output_dir)) { std::filesystem::create_directories(output_dir); }
    spdlog::info("Output folder : {}", std::filesystem::absolute(output_dir).string());

    spdlog::stopwatch sw;
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", objfiles_dir, obj_file);
    load_obj(filepath, clothmesh);

    clothmesh.ComputeFaceNormals();

    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 0.0f, 1.0f);

    Matrix3Xr curr_pos = sim.get_position();
    Matrix3Xr gravity(curr_pos);
    gravity.setZero();
    // TO DO: set gravity
    for (integer i = 0; i < static_cast<integer>(curr_pos.cols()); i++) {
        curr_pos.col(i)(2) = 2.0f;
        gravity.col(i)(2) = -9.8f;
    }
    sim.set_position(curr_pos);
    sim.set_external_acceleration(gravity);
    real timestep = 0.001f;

    std::string framepath = std::format("{}/{}_{:04d}.obj", output_dir, "naive_cloth", 0);
    write_obj(framepath, Mesh(sim.get_position(), sim.get_indice()));
    spdlog::info("Frame 0, {:.2f}s elapsed", sw);
    for (integer iter = 1; iter < 201; iter++) {
        sim.Forward(timestep);
        std::string framepath = std::format("{}/{}_{:04d}.obj", output_dir, "naive_cloth", iter);
        write_obj(framepath, Mesh(sim.get_position(), sim.get_indice()));
        spdlog::info("Frame {}, {:.2f}s elapsed", iter, sw);
    }

    return 0;
}