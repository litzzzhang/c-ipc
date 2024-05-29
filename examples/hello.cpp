#include <iostream>
#include "c-ipc/c-ipc.h"

using namespace cipc;

int main(int argc, char **argv) {

    Vector3r a(1.0, 2.0, -1.0);
    Vector3r b(2.0, 3.0, 1.0);
    std::array<long, 3> vertex_ids;
    
    vertex_ids = {1,1,1};

    std::cout << a.cwiseMin(b).cwiseEqual(a).all() << '\n';
    std::cout << a.cwiseMin(b).cwiseEqual(a).sum()<< '\n';

    return 0;
}
