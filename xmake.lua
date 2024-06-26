set_project("c-ipc")
set_xmakever("2.8.5")

add_rules("mode.debug", "mode.release")
set_languages("c++20")
set_fpmodels("precise")

add_requires("spdlog", {configs = {std_format = true}})
add_requires("eigen")
add_requires("onetbb")
add_requires("catch2")

option("c-ipc_examples", {default = true, description = "Build examples."})
option("c-ipc_tests", {default = true, description = "Build tests."})

if is_plat("windows") then
    add_defines("_CRT_SECURE_NO_WARNINGS")
    add_cxflags("/EHsc")
    add_cxflags("/bigobj")
    add_cxflags("/Zc:preprocessor")
    add_cuflags("-Xcompiler /Zc:preprocessor")
end

set_warnings("all")

if has_config("vl_tests") then
    add_requires("catch2")
end

target("c-ipc")
    set_kind("headeronly")
    set_fpmodels("precise")
    add_includedirs(".", {public = true})
    add_headerfiles("(c-ipc/**.h)")
    add_packages("spdlog", "eigen", "onetbb", {public = true})
target_end()

if has_config("c-ipc_examples") then
    includes("examples")
end

if has_config("c-ipc_tests") then
    includes("tests")
end
--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

