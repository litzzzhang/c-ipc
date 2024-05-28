target("tinyobj")
    set_kind("static")
    add_deps("c-ipc")
    add_files("tiny_obj_loader.cc")

target("mytest")
    add_deps("c-ipc", "tinyobj")
    add_files("hello.cpp")

target("naive_cloth")
    add_deps("c-ipc", "tinyobj")
    add_files("naive_cloth.cpp")

