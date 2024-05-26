#pragma once

#include <spdlog/spdlog.h>
#include <format>

namespace cipc {
inline static void __massert(
    bool condition, const std::string str, const char *filename, int line, const char *funcname) {
    if (!condition) {
        spdlog::error("assertation failed: {} at {}:{} [{}]", str, filename, line, funcname);
        std::abort();
    }
}
#define cipc_massert(cond, str) __massert(cond, str, __FILE__, __LINE__, __func__)

template <typename...>
struct check_typeid; // for compile-time type checking

#define cipc_assert(cond, str, ...) cipc_massert(cond, std::format(str __VA_OPT__(, ) __VA_ARGS__))
} // namespace cipc