#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    Matrix3r a = Matrix3r::Identity();

    std::cout << a.col(a.cols() - 1);
    return 0;
}
