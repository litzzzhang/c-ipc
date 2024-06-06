#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include "finitediff.hpp"
#include "c-ipc/c-ipc.h"

using namespace cipc;

TEST_CASE("Shearing and Stretching Energy", "[gradient][hessian]") {
    spdlog::set_level(spdlog::level::debug);
    Mesh clothmesh;
    std::string filepath = std::format("{}/{}", "./obj_files", "mat10x10.obj");
    load_obj(filepath, clothmesh);
    Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);
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
    // dirichlet.col(9) = curr_pos.col(9);
    sim.set_position(curr_pos);
    sim.set_external_acceleration(gravity);
    sim.set_dirichlet_boundary(dirichlet);
    real timestep = 0.001f;

    auto energyfunc = [&](const Matrix3Xr &pos) {
        return sim.ComputeBendingEnergy(pos);
    };
    auto gradientfunc = [&](const Matrix3Xr &pos) {
        return -sim.ComputeBendingForce(pos);
    };

    std::default_random_engine rng;
    rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    for (integer i = 0; i < 1000; i++) {
        sim.Forward(timestep);
        Matrix3Xr x = sim.get_position();
        auto grad_result = finite_gradient(x, rng, energyfunc, gradientfunc(x));
        printf("%.9f, %.9f\n", grad_result.analytic_diff, grad_result.numeric_diff);
        // REQUIRE_THAT(
        //     grad_result.numeric_diff,
        //     Catch::Matchers::WithinAbs(grad_result.analytic_diff, 0.1)
        //         || Catch::Matchers::WithinRel(grad_result.analytic_diff, 0.01));
    }
}

// TEST_CASE("check the correctness of checkers", "meta check") {
//     spdlog::info("checker check");
//     integer dim = 9999;
//     std::default_random_engine rng;
//     rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
//     static std::uniform_real_distribution<real> two_random(0.0, 2.0);

//     auto energyfunc = [&](const Matrix3Xr &pos) {
//         return pos.squaredNorm();
//     };
//     auto gradientfunc = [&](const Matrix3Xr &pos) {
//         return 2.0 * pos;
//     };
//     auto hessianfunc = [&](const Matrix3Xr &pos) {
//         integer hess_size = 3 * static_cast<integer>(pos.cols());
//         std::vector<Eigen::Triplet<real>> triplist;
//         for (integer i = 0; i < hess_size; i++) { triplist.push_back({i, i, 2.0}); }
//         return FromTriplet(hess_size, hess_size, triplist);
//     };

//     using EnergyFuncType = decltype(energyfunc);
//     using GradFuncType = decltype(gradientfunc);
//     using HessFuncType = decltype(hessianfunc);
//     struct Model {
//         EnergyFuncType energy;
//         GradFuncType gradient;
//         HessFuncType hessian;
//         Model(EnergyFuncType energy_, GradFuncType gradient_, HessFuncType hessian_)
//             : energy(energy_), gradient(gradient_), hessian(hessian_) {}
//     };

//     Model model(energyfunc, gradientfunc, hessianfunc);
//     for (integer i = 0; i < 1000; i++) {
//         VectorXr test_vec = VectorXr::Zero(dim);
//         oneapi::tbb::parallel_for(0, dim, [&](integer i) {
//             test_vec(i) = two_random(rng);
//         });
//         Matrix3Xr x = test_vec.reshaped(3, dim / 3);
//         auto grad_result = finite_gradient(x, rng, energyfunc, gradientfunc(x));
//         REQUIRE_THAT(
//             grad_result.numeric_diff,
//             Catch::Matchers::WithinAbs(grad_result.analytic_diff, 0.001)
//                 || Catch::Matchers::WithinRel(grad_result.analytic_diff, 0.001));
//         // CHECK(gradient_checker(model, test_vec.reshaped(3, dim / 3), rng));
//         // CHECK(hessian_checker(model, test_vec.reshaped(3, dim / 3), rng));
//     }
// }
// TEST_CASE("Shearing and Stretching energy gradient check", "Dihedral Bend") {

//     Mesh clothmesh;
//     std::string filepath = std::format("{}/{}", "./obj_files", "mat30x30.obj");
//     load_obj(filepath, clothmesh);
//     Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);

//     auto energyfunc = [&](const Matrix3Xr &pos) {
//         return sim.ComputeBendingEnergy(pos);
//     };

//     auto gradientfunc = [&](const Matrix3Xr &pos) {
//         return -sim.ComputeBendingForce(pos);
//     };

//     using EnergyFuncType = decltype(energyfunc);
//     using GradFuncType = decltype(gradientfunc);
//     struct Model {
//         EnergyFuncType energy;
//         GradFuncType gradient;
//         Model(EnergyFuncType energy_, GradFuncType gradient_)
//             : energy(energy_), gradient(gradient_) {}
//     };

//     Model model(energyfunc, gradientfunc);

//     std::default_random_engine rng;
//     rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
//     real timestep = 0.001f;
//     for (integer i = 0; i < 10; i++) {
//         sim.Forward(timestep);
//         REQUIRE(gradient_checker(model, sim.get_position(), rng));
//     }
// }

// // TO DO: set dirichlet boundary condition
// // note the checker is right
// TEST_CASE("stretching energy gradient check", "[NaiveStvK]") {

//     spdlog::info("stretching energy and gradient test begin.");
//     Mesh clothmesh;
//     std::string filepath = std::format("{}/{}", "./obj_files", "mat2x2.obj");
//     load_obj(filepath, clothmesh);
//     Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);
//     Matrix3Xr curr_pos = sim.get_position();
//     Matrix3Xr gravity(curr_pos), dirichlet(curr_pos);
//     const real inf = std::numeric_limits<real>::infinity();
//     gravity.setZero(), dirichlet.setConstant(inf);
//     integer vertex_num = static_cast<integer>(curr_pos.cols());
//     for (integer i = 0; i < vertex_num; i++) {
//         curr_pos(2, i) = 2.0f;
//         gravity(2, i) = -9.8f;
//     }
//     // fix some points
//     dirichlet.col(0) = curr_pos.col(0);
//     dirichlet.col(1) = curr_pos.col(1);
//     sim.set_position(curr_pos);
//     sim.set_external_acceleration(gravity);
//     sim.set_dirichlet_boundary(dirichlet);
//     auto energyfunc = [&](const Matrix3Xr &pos) {
//         return sim.ComputeStretchingAndShearingEnergy(pos);
//     };

//     auto gradientfunc = [&](const Matrix3Xr &pos) {
//         return -sim.ComputeStretchingAndShearingForce(pos);
//     };

//     using EnergyFuncType = decltype(energyfunc);
//     using GradFuncType = decltype(gradientfunc);
//     struct Model {
//         EnergyFuncType energy;
//         GradFuncType gradient;
//         Model(EnergyFuncType energy_, GradFuncType gradient_)
//             : energy(energy_), gradient(gradient_) {}
//     };

//     Model model(energyfunc, gradientfunc);

//     std::default_random_engine rng;
//     rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
//     real timestep = 0.001f;
//     for (integer i = 0; i < 10; i++) {
//         sim.Forward(timestep);
//         REQUIRE(gradient_checker(model, sim.get_position(), rng));
//     }
// }

// TEST_CASE("stretching energy Hessian check", "[NaiveStvK]") {

//     spdlog::info("stretching hessian test begin.");
//     Mesh clothmesh;
//     std::string filepath = std::format("{}/{}", "./obj_files", "mat20x20.obj");
//     load_obj(filepath, clothmesh);
//     clothmesh.ComputeFaceNormals();
//     Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);

//     auto gradientfunc = [&](const Matrix3Xr &pos) {
//         return -sim.ComputeStretchingAndShearingForce(pos);
//     };
//     auto hessianfunc = [&](const Matrix3Xr &pos) {
//         return sim.ComputeStretchingAndShearingHessian(pos);
//     };
//     using HessFuncType = decltype(hessianfunc);
//     using GradFuncType = decltype(gradientfunc);
//     struct Model {
//         HessFuncType hessian;
//         GradFuncType gradient;
//         Model(GradFuncType gradient_, HessFuncType hessian_)
//             : gradient(gradient_), hessian(hessian_) {}
//     };

//     Model model(gradientfunc, hessianfunc);
//     std::default_random_engine rng;
//     rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
//     real timestep = 0.001f;
//     for (integer i = 0; i < 1000; i++) {
//         sim.Forward(timestep);
//         REQUIRE(hessian_checker(model, sim.get_position(), rng));
//     }
// }

// TEST_CASE("bending energy Hessian check", "Dihedral Bend") {

//     spdlog::info("bending hessian test begin.");
//     Mesh clothmesh;
//     std::string filepath = std::format("{}/{}", "./obj_files", "mat20x20.obj");
//     load_obj(filepath, clothmesh);
//     clothmesh.ComputeFaceNormals();
//     Simulator<NaiveStvK> sim(clothmesh, 0.1f, 1.0f, 1.0f);

//     auto gradientfunc = [&](const Matrix3Xr &pos) {
//         return -sim.ComputeBendingForce(pos);
//     };
//     auto hessianfunc = [&](const Matrix3Xr &pos) {
//         return sim.ComputeBendingHessian(pos);
//     };
//     using HessFuncType = decltype(hessianfunc);
//     using GradFuncType = decltype(gradientfunc);
//     struct Model {
//         HessFuncType hessian;
//         GradFuncType gradient;
//         Model(GradFuncType gradient_, HessFuncType hessian_)
//             : gradient(gradient_), hessian(hessian_) {}
//     };

//     Model model(gradientfunc, hessianfunc);
//     std::default_random_engine rng;
//     rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
//     real timestep = 0.001f;
//     for (integer i = 0; i < 1000; i++) {
//         sim.Forward(timestep);
//         REQUIRE(hessian_checker(model, sim.get_position(), rng));
//     }

//////copy

// spdlog::set_pattern("[%m-%d %T] %^[%l]%$ %v");
// spdlog::info("stretching energy and gradient test begin.");
// Mesh clothmesh;
// std::string filepath = std::format("{}/{}", "./obj_files", "mat4x4.obj");
// load_obj(filepath, clothmesh);
// clothmesh.ComputeFaceNormals();
// Simulator<NaiveStvK> sim(clothmesh, 0.1f, 0.5f, 1.0f);
// Matrix3Xr curr_pos = sim.get_position();
// Matrix3Xr gravity(curr_pos), dirichlet(curr_pos);
// const real inf = std::numeric_limits<real>::infinity();
// gravity.setZero(), dirichlet.setConstant(inf);
// for (integer i = 0; i < static_cast<integer>(curr_pos.cols()); i++) {
//     curr_pos(2, i) = 2.0f;
//     gravity(2, i) = -9.8f;
// }
// // fix a point
// dirichlet.col(0) = curr_pos.col(0);
// sim.set_position(curr_pos);
// sim.set_external_acceleration(gravity);
// sim.set_dirichlet_boundary(dirichlet);

// auto energyfunc = [&](const Matrix3Xr &pos) {
//     return sim.ComputeStretchingAndShearingEnergy(pos);
// };

// auto gradientfunc = [&](const Matrix3Xr &pos) {
//     return -sim.ComputeStretchingAndShearingForce(pos);
// };

// auto hessianfunc = [&](const Matrix3Xr &pos) {
//     return sim.ComputeStretchingAndShearingHessian(pos);
// };
// using EnergyFuncType = decltype(energyfunc);
// using GradFuncType = decltype(gradientfunc);
// using HessFuncType = decltype(hessianfunc);
// struct Model {
//     EnergyFuncType energy;
//     GradFuncType gradient;
//     HessFuncType hessian;
//     Model(EnergyFuncType energy_, GradFuncType gradient_, HessFuncType hessian_)
//         : energy(energy_), gradient(gradient_), hessian(hessian_) {}
// };

// Model model(energyfunc, gradientfunc, hessianfunc);

// std::default_random_engine rng;

// rng.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
// real timestep = 0.001f;
// for (integer i = 0; i < 1000; i++) {
//     sim.Forward(timestep);
//     if (!gradient_checker(model, sim.get_position(), rng)) { printf("error\n"); }
// }
// }
