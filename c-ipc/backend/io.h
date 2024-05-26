#pragma once

#include <c-ipc/backend/stl_port.h>
#include <c-ipc/geometry/geometry.h>

namespace cipc {

inline void load_obj(const std::string filename, const std::string dirname, Mesh &mesh) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "";
    reader_config.vertex_color = false;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        cipc_assert(reader.Error().empty(), "Tinyobj load error");
        std::abort();
    }

    if (!reader.Warning().empty()) { std::cout << "TinyObjReader: " << reader.Warning(); }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    // only one shape in one .obj file
    mesh.clear();
    mesh.elem_num = static_cast<integer>(shapes[0].mesh.indices.size() / 3);
    mesh.vertices_num = static_cast<integer>(attrib.vertices.size() / 3);
    mesh.vertices.setZero(3, mesh.vertices_num);
    mesh.indices.setZero(3, mesh.elem_num);

    // loop over all vertices
    for (integer i = 0; i < mesh.vertices_num; i++) {
        tinyobj::real_t vx = attrib.vertices[3 * i + 0];
        tinyobj::real_t vy = attrib.vertices[3 * i + 1];
        tinyobj::real_t vz = attrib.vertices[3 * i + 2];
        Vector3r vertex(vx, vy, vz);
        mesh.vertices.col(i) = vertex;
    }

    // loop over all indices
    for (integer i = 0; i < mesh.elem_num; i++) {
        integer ix = shapes[0].mesh.indices[3 * i + 0].vertex_index;
        integer iy = shapes[0].mesh.indices[3 * i + 1].vertex_index;
        integer iz = shapes[0].mesh.indices[3 * i + 2].vertex_index;
        Vector3i indice(ix, iy, iz);
        mesh.indices.col(i) = indice;
    }
}
} // namespace cipc
// save frame as obj file