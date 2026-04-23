"""STL(s) -> optimized multi-mesh GLB.

Pure module: no I/O, no env vars, no network.

Takes one or more (name, stl_bytes) pairs, builds a single GLB containing each
STL as a named mesh node. The viewer uses those node names to generate toggles
for each anatomical structure (see medCaseViewer/case/main.js).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import fast_simplification
import trimesh

DEFAULT_TARGET_TRIANGLES = 300_000


@dataclass
class MeshStats:
    name: str
    input_triangles: int
    output_triangles: int
    decimated: bool


@dataclass
class ProcessStats:
    total_input_triangles: int
    total_output_triangles: int
    glb_size_bytes: int
    meshes: list[MeshStats] = field(default_factory=list)


def _load_and_decimate(
    stl_bytes: bytes, target_triangles: int
) -> tuple[trimesh.Trimesh, int, bool]:
    try:
        mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", process=True)
    except Exception as e:
        raise ValueError(f"STL inválido ou corrompido: {e}") from e

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Arquivo STL não pôde ser carregado como uma única malha.")

    input_tris = len(mesh.faces)
    if input_tris > target_triangles:
        points_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_triangles
        )
        mesh = trimesh.Trimesh(vertices=points_out, faces=faces_out, process=False)
        return mesh, input_tris, True
    return mesh, input_tris, False


def process_stls(
    files: list[tuple[str, bytes]],
    target_triangles_per_mesh: int = DEFAULT_TARGET_TRIANGLES,
) -> tuple[bytes, ProcessStats]:
    """Build a single multi-mesh GLB from a list of (name, stl_bytes)."""
    if not files:
        raise ValueError("Nenhum arquivo recebido.")

    scene = trimesh.Scene()
    mesh_stats: list[MeshStats] = []
    total_in = 0
    total_out = 0

    for name, stl_bytes in files:
        mesh, input_tris, decimated = _load_and_decimate(
            stl_bytes, target_triangles_per_mesh
        )
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        output_tris = len(mesh.faces)
        total_in += input_tris
        total_out += output_tris
        mesh_stats.append(
            MeshStats(
                name=name,
                input_triangles=input_tris,
                output_triangles=output_tris,
                decimated=decimated,
            )
        )

    glb_bytes = scene.export(file_type="glb")

    return glb_bytes, ProcessStats(
        total_input_triangles=total_in,
        total_output_triangles=total_out,
        glb_size_bytes=len(glb_bytes),
        meshes=mesh_stats,
    )
