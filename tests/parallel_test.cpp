#include <c-ipc/c-ipc.h>
#include <catch2/catch_test_macros.hpp>

using namespace cipc;

TEST_CASE("reduce test", "parallel test") {
    integer dim = 100;
    integer result = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<integer>(1, dim + 1), 0,
        [&](oneapi::tbb::blocked_range<integer> r, integer local) {
            for (integer i = r.begin(); i < r.end(); i++) { local += i; }
            return local;
        },
        [](integer x, integer y) {
            return x + y;
        });
    
    REQUIRE(result == 5050);
}