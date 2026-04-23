"""STL(s) -> optimized multi-mesh GLB with per-structure colors.

Pure module: no I/O, no env vars, no network.

Takes one or more (name, stl_bytes) pairs, builds a single GLB containing each
STL as a named mesh node with its own PBR material (so the viewer's opacity
slider — which acts on materials — works per-structure). Coordinates are
rotated from RAS (Z-up, medical convention) to glTF (Y-up).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import fast_simplification
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

DEFAULT_TARGET_TRIANGLES = 300_000

# Keyword-to-color mapping — case-insensitive substring match against mesh name.
# Order matters only if keywords could overlap (here none do).
COLORS_BY_KEYWORD: dict[str, str] = {
    "art": "#BD0006",    # artéria: vermelho escuro
    "veia": "#458DE7",   # veia: azul
    "rim": "#BA5531",    # rim: marrom-alaranjado
    "lesao": "#08E700",  # lesão: verde brilhante
    "pele": "#C4908E",   # pele: rosado
}

# IBM Colorblind Safe palette (minus the orange, which overlaps kidney brown).
# Used for structures whose names don't match any keyword above.
FALLBACK_COLORS: list[str] = [
    "#648FFF",  # blue
    "#785EF0",  # purple
    "#DC267F",  # magenta
    "#FFB000",  # gold
    "#FE6100",  # orange
]

# RAS (Z-up) -> glTF (Y-up). Rotate -90° around X so Z becomes Y.
_RAS_TO_GLTF = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


@dataclass
class MeshStats:
    name: str
    input_triangles: int
    output_triangles: int
    decimated: bool
    color: str


@dataclass
class ProcessStats:
    total_input_triangles: int
    total_output_triangles: int
    glb_size_bytes: int
    meshes: list[MeshStats] = field(default_factory=list)


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def _pick_color(name: str, fallback_idx: int) -> str:
    lower = name.lower()
    for keyword, hex_color in COLORS_BY_KEYWORD.items():
        if keyword in lower:
            return hex_color
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


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
        # process=True recomputes per-vertex normals (needed for smooth shading after decimation)
        mesh = trimesh.Trimesh(vertices=points_out, faces=faces_out, process=True)
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
    fallback_idx = 0
    total_in = 0
    total_out = 0

    for name, stl_bytes in files:
        mesh, input_tris, decimated = _load_and_decimate(
            stl_bytes, target_triangles_per_mesh
        )

        mesh.apply_transform(_RAS_TO_GLTF)
        # Force vertex-normal compute AFTER the transform so the GLB exporter
        # includes the NORMAL attribute; without it the viewer renders flat-shaded.
        _ = mesh.vertex_normals

        matched_by_keyword = any(k in name.lower() for k in COLORS_BY_KEYWORD)
        color_hex = _pick_color(name, fallback_idx)
        if not matched_by_keyword:
            fallback_idx += 1

        r, g, b = _hex_to_rgb01(color_hex)
        mesh.visual = TextureVisuals(
            material=PBRMaterial(
                name=name,
                baseColorFactor=[r, g, b, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5,
            )
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
                color=color_hex,
            )
        )

    glb_bytes = scene.export(file_type="glb")

    return glb_bytes, ProcessStats(
        total_input_triangles=total_in,
        total_output_triangles=total_out,
        glb_size_bytes=len(glb_bytes),
        meshes=mesh_stats,
    )
