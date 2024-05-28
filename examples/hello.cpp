#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {
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
        if (!gradient_checker(model, test_vec.reshaped(3, dim / 3), rng)) {
            spdlog::error("gradient wrong");
        }
        if (!hessian_checker(model, test_vec.reshaped(3, dim / 3), rng)) {
            spdlog::error("hessian wrong");
        }
    }

    return 0;
}
