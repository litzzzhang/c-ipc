#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    Matrix3r a;
    a << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;
    Vector2i b = {2,1};
    std::cout << a(b, Eigen::all);
    printf("\nhello world!\n");
    return 0;
}
