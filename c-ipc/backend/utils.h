#pragma once

#include "c-ipc/solver/eigen.h"
#include "c-ipc/backend/stl_port.h"

namespace cipc
{
    template<typename TestModel>
    bool gradient_checker(TestModel model, Matrix3Xr x){
        integer dim = 3 * static_cast<integer>(x.cols());
        real base = model.energy(x);
        Matrix3Xr gradient = model.gradient(x);
        return true;
    };
} // namespace cipc
