/* Naive cloth simulator with dihedral angle based isotropic model*/

#include "c-ipc/c-ipc.h"
#include <spdlog/stopwatch.h>

#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cipc;

int main() {
    spdlog::set_pattern("[%m-%d %T] %^[%l]%$ %v");
    const std::string output_dir("./output/naive_cloth10");
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

    clothmesh.vertices *= 2.0;
    
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 0.1, 1.0f);

    Matrix3Xr curr_pos = sim.get_position();
    Matrix3Xr gravity(curr_pos), dirichlet(curr_pos);
    const real inf = std::numeric_limits<real>::infinity();
    gravity.setZero(), dirichlet.setConstant(inf);
    integer vertex_num = static_cast<integer>(curr_pos.cols());
    for (integer i = 0; i < vertex_num; i++) {
        curr_pos(2, i) = 2.0f;
        gravity(2, i) = -9.8f;
    }
    // fix some points
    dirichlet.col(0) = curr_pos.col(0);
    dirichlet.col(39) = curr_pos.col(39);
    sim.set_position(curr_pos);
    sim.set_external_acceleration(gravity);
    sim.set_dirichlet_boundary(dirichlet);
    real timestep = 0.001f;

    std::string framepath = std::format("{}/{}_{:04d}.obj", output_dir, "naive_cloth", 0);
    write_obj(framepath, Mesh(sim.get_position(), sim.get_indice()));
    spdlog::info("Frame 0, {:.2f}s elapsed", sw);
    for (integer iter = 1; iter < 1001; iter++) {
        sim.Forward(timestep);
        std::string framepath = std::format("{}/{}_{:04d}.obj", output_dir, "naive_cloth", iter);
        write_obj(framepath, Mesh(sim.get_position(), sim.get_indice()));
        spdlog::info("Frame {}, {:.2f}s elapsed", iter, sw);
    }

    return 0;
}