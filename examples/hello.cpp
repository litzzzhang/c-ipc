#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    Matrix3x4r test = Matrix3x4r::Zero();

    test(Eigen::all, Vector3i(0,2,3)) = Matrix3r::Identity();

    std::cout <<test;

    return 0;
}
