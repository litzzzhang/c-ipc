#pragma once

#include <c-ipc/backend/stl_port.h>
#include <c-ipc/geometry/geometry.h>

namespace cipc {

inline void load_obj(const std::string filepath, Mesh &mesh) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "";
    reader_config.vertex_color = false;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filepath, reader_config)) {
        cipc_assert(reader.Error().empty(), "Tinyobj load error");
        std::abort();
    }

    // if (!reader.Warning().empty()) { std::cout << "TinyObjReader: " << reader.Warning(); }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    // only one shape in one .obj file
    mesh.clear();
    integer elem_num = static_cast<integer>(shapes[0].mesh.indices.size() / 3);
    integer vertices_num = static_cast<integer>(attrib.vertices.size() / 3);
    mesh.vertices.setZero(3, vertices_num);
    mesh.indices.setZero(3, elem_num);

    // loop over all vertices
    for (integer i = 0; i < vertices_num; i++) {
        tinyobj::real_t vx = attrib.vertices[3 * i + 0];
        tinyobj::real_t vy = attrib.vertices[3 * i + 1];
        tinyobj::real_t vz = attrib.vertices[3 * i + 2];
        Vector3r vertex(vx, vy, vz);
        mesh.vertices.col(i) = vertex;
    }

    // loop over all indices
    for (integer i = 0; i < elem_num; i++) {
        integer ix = shapes[0].mesh.indices[3 * i + 0].vertex_index;
        integer iy = shapes[0].mesh.indices[3 * i + 1].vertex_index;
        integer iz = shapes[0].mesh.indices[3 * i + 2].vertex_index;
        Vector3i indice(ix, iy, iz);
        mesh.indices.col(i) = indice;
    }
}

inline void write_obj(const std::string filepath, const Mesh &mesh) {

    std::FILE *fp = std::fopen(filepath.c_str(), "w");
    cipc_assert(fp, "Cannot open file {}", filepath);

    integer v_size = static_cast<integer>(mesh.vertices.cols());
    for (integer i = 0; i < v_size; i++) {
        fprintf(fp, "v %f %f %f\n", mesh.vertices(0, i), mesh.vertices(1, i), mesh.vertices(2, i));
    }

    integer i_size = static_cast<integer>(mesh.indices.cols());
    for (integer i = 0; i < i_size; i++) {
        // obj file starts from 1, not 0
        fprintf(fp, "f %d %d %d\n", mesh.indices(0, i) + 1, mesh.indices(1, i) + 1, mesh.indices(2, i) + 1);
    }

    for (integer i = 0; i < v_size; i++) {
        fprintf(
            fp, "vn %f %f %f\n", mesh.vert_normals(0, i), mesh.vert_normals(1, i),
            mesh.vert_normals(2, i));
    }
    std::fclose(fp);
}
} // namespace cipc
// save frame as obj file