#include <catch2/catch_test_macros.hpp>
#include "c-ipc/c-ipc.h"

using namespace cipc;

TEST_CASE("check the correctness of checkers", "meta check") {
    spdlog::info("checker check");
    integer dim = 9999;
    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<real> two_random(0.0, 2.0);

    auto energyfunc = [&](const Matrix3Xr &pos) {
        return pos.reshaped().dot(pos.reshaped().transpose());
    };
    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return 2.0 * pos;
    };
    auto hessianfunc = [&](const Matrix3Xr &pos) {
        integer hess_size = 3 * static_cast<integer>(pos.cols());
        std::vector<Eigen::Triplet<real>> triplist;
        for (integer i = 0; i < hess_size; i++) { triplist.push_back({i, i, 2.0}); }
        return FromTriplet(hess_size, hess_size, triplist);
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
    for (integer i = 0; i < 1000; i++) {
        VectorXr test_vec = VectorXr::Zero(dim);
        oneapi::tbb::parallel_for(0, dim, [&](integer i) {
            test_vec(i) = two_random(rng);
        });
        REQUIRE(gradient_checker(model, test_vec.reshaped(3, dim / 3), rng));
        REQUIRE(hessian_checker(model, test_vec.reshaped(3, dim / 3), rng));
    }
}

TEST_CASE("bending energy gradient check", "Dihedral Bend") {

    spdlog::info("bending energy and gradient test begin.");
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat30x30.obj");
    load_obj(filepath, clothmesh);
    clothmesh.ComputeFaceNormals();
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);
    auto energyfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeBendingEnergy(pos);
    };

    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeBendingForce(pos);
    };

    using EnergyFuncType = decltype(energyfunc);
    using GradFuncType = decltype(gradientfunc);
    struct Model {
        EnergyFuncType energy;
        GradFuncType gradient;
        Model(EnergyFuncType energy_, GradFuncType gradient_)
            : energy(energy_), gradient(gradient_) {}
    };

    Model model(energyfunc, gradientfunc);

    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    real timestep = 0.001f;
    for (integer i = 0; i < 10; i++) {
        sim.Forward(timestep);
        REQUIRE(gradient_checker(model, sim.get_position(), rng));
    }
}

// TO DO: set dirichlet boundary condition
// note the checker is right
TEST_CASE("stretching energy gradient check", "[NaiveStvK]") {

    spdlog::info("stretching energy and gradient test begin.");
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat30x30.obj");
    load_obj(filepath, clothmesh);
    clothmesh.ComputeFaceNormals();
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);
    auto energyfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeStretchingAndShearingEnergy(pos);
    };

    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeStretchingAndShearingForce(pos);
    };

    using EnergyFuncType = decltype(energyfunc);
    using GradFuncType = decltype(gradientfunc);
    struct Model {
        EnergyFuncType energy;
        GradFuncType gradient;
        Model(EnergyFuncType energy_, GradFuncType gradient_)
            : energy(energy_), gradient(gradient_) {}
    };

    Model model(energyfunc, gradientfunc);

    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    real timestep = 0.001f;
    for (integer i = 0; i < 10; i++) {
        sim.Forward(timestep);
        REQUIRE(gradient_checker(model, sim.get_position(), rng));
    }
}

TEST_CASE("stretching energy Hessian check", "[NaiveStvK]") {

    spdlog::info("stretching hessian test begin.");
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat20x20.obj");
    load_obj(filepath, clothmesh);
    clothmesh.ComputeFaceNormals();
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);

    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeStretchingAndShearingForce(pos);
    };
    auto hessianfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeStretchingAndShearingHessian(pos);
    };
    using HessFuncType = decltype(hessianfunc);
    using GradFuncType = decltype(gradientfunc);
    struct Model {
        HessFuncType hessian;
        GradFuncType gradient;
        Model(GradFuncType gradient_, HessFuncType hessian_)
            : gradient(gradient_), hessian(hessian_) {}
    };

    Model model(gradientfunc, hessianfunc);
    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    real timestep = 0.001f;
    for (integer i = 0; i < 1000; i++) {
        sim.Forward(timestep);
        REQUIRE(hessian_checker(model, sim.get_position(), rng));
    }
}

TEST_CASE("bending energy Hessian check", "Dihedral Bend") {

    spdlog::info("bending hessian test begin.");
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat20x20.obj");
    load_obj(filepath, clothmesh);
    clothmesh.ComputeFaceNormals();
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);

    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeBendingForce(pos);
    };
    auto hessianfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeBendingHessian(pos);
    };
    using HessFuncType = decltype(hessianfunc);
    using GradFuncType = decltype(gradientfunc);
    struct Model {
        HessFuncType hessian;
        GradFuncType gradient;
        Model(GradFuncType gradient_, HessFuncType hessian_)
            : gradient(gradient_), hessian(hessian_) {}
    };

    Model model(gradientfunc, hessianfunc);
    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    real timestep = 0.001f;
    for (integer i = 0; i < 1000; i++) {
        sim.Forward(timestep);
        REQUIRE(hessian_checker(model, sim.get_position(), rng));
    }
}

