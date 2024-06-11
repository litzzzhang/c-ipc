#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    Matrix2r A = Matrix2r::Zero();
    A(0,0) = 16.0;
    A(1,1) = 1038.54;
    Vector2r b(8, -64.9);

    std::cout << A.ldlt().solve(b);
    return 0;
}
