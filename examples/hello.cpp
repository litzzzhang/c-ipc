#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    spdlog::set_pattern("[%m-%d %T] %^[%l]%$ %v");
    spdlog::info("stretching energy and gradient test begin.");
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat4x4.obj");
    load_obj(filepath, clothmesh);
    clothmesh.ComputeFaceNormals();
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 0.5f, 1.0f);
    Matrix3Xr curr_pos = sim.get_position();
    Matrix3Xr gravity(curr_pos), dirichlet(curr_pos);
    const real inf = std::numeric_limits<real>::infinity();
    gravity.setZero(), dirichlet.setConstant(inf);
    for (integer i = 0; i < static_cast<integer>(curr_pos.cols()); i++) {
        curr_pos(2, i) = 2.0f;
        gravity(2, i) = -9.8f;
    }
    // fix a point
    dirichlet.col(0) = curr_pos.col(0);
    sim.set_position(curr_pos);
    sim.set_external_acceleration(gravity);
    sim.set_dirichlet_boundary(dirichlet);

    auto energyfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeStretchingAndShearingEnergy(pos);
    };

    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeStretchingAndShearingForce(pos);
    };

    auto hessianfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeStretchingAndShearingHessian(pos);
    };
    using EnergyFuncType = decltype(energyfunc);
    using GradFuncType = decltype(gradientfunc);
    using HessFuncType = decltype(hessianfunc);
    struct Model {
        EnergyFuncType energy;
        GradFuncType gradient;
        HessFuncType hessian;
        Model(EnergyFuncType energy_, GradFuncType gradient_, HessFuncType hessian_)
            : energy(energy_), gradient(gradient_), hessian(hessian_) {}
    };

    Model model(energyfunc, gradientfunc, hessianfunc);

    std::default_random_engine rng;

    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    real timestep = 0.001f;
    for (integer i = 0; i < 1000; i++) {
        sim.Forward(timestep);
        if (!gradient_checker(model, sim.get_position(), rng)) { printf("error\n"); }
    }

    return 0;
}
