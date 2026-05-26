"""STL(s) -> optimized multi-mesh GLB with per-structure colors.

Pure module: no I/O, no env vars, no network.

Takes one or more (name, stl_bytes) pairs, builds a single GLB containing each
STL as a named mesh node with its own PBR material (so the viewer's opacity
slider — which acts on materials — works per-structure). Coordinates are
rotated from RAS (Z-up, medical convention) to glTF (Y-up).

Also exposes `process_obj_bundle` for OBJ + MTL + texture image uploads
(photogrammetry-style scans). That path skips decimation (decimation would
break UV mapping), skips the keyword color logic (the user wants the original
texture), and scales meters → millimeters (KIRI-style scanners output meters
while the viewer assumes 1 unit = 1 mm everywhere).
"""
from __future__ import annotations

import colorsys
import io
import os
import tempfile
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
    "art": "#BD0006",     # artéria: vermelho escuro
    "vei": "#477EFF",     # veia / vein: azul
    "rim": "#BA5531",     # rim: marrom-alaranjado
    "lesao": "#08E700",   # lesão: verde brilhante
    "tumor": "#08E700",   # tumor: mesmo verde da lesão (compartilha bucket → varia HSV)
    "pele": "#C4908E",    # pele: rosado
    "cortex": "#966830",  # córtex: marrom
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


def _vary_hsv(hex_color: str, index: int) -> str:
    """Walk HSV offsets so duplicates of the same base color stay distinguishable.

    Base colors here are typically near-max saturation/value, so going *up* on V
    clamps to no-op; we always darken and alternate saturation around the base.
    """
    if index == 0:
        return hex_color
    r, g, b = _hex_to_rgb01(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(0.30, v - 0.13 * index)
    s_step = ((index + 1) // 2) * 0.15
    s = max(0.30, min(1.0, s + (s_step if index % 2 == 0 else -s_step)))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _load_and_decimate(
    stl_bytes: bytes, target_triangles: int
) -> tuple[trimesh.Trimesh, int, bool]:
    try:
        loaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", process=True)
    except Exception as e:
        raise ValueError(f"STL inválido ou corrompido: {e}") from e

    # STL multi-body (vários `solid`/`endsolid`) volta como Scene. Concatenamos
    # todas as partes numa única Trimesh — para nós, um STL = uma estrutura
    # anatômica, mesmo quando o software de segmentação exporta em pedaços
    # desconectados. A decimação abaixo já cuida da contagem total de triângulos.
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise ValueError("STL não contém nenhuma malha utilizável.")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = loaded

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
    # Counts per base hex — when >1 mesh lands on the same color (same keyword,
    # tumor+lesão sharing green, or fallback palette wrapping), vary V/S so the
    # viewer's per-structure toggles remain visually distinguishable.
    bucket_counts: dict[str, int] = {}
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
        base_hex = _pick_color(name, fallback_idx)
        if not matched_by_keyword:
            fallback_idx += 1

        within = bucket_counts.get(base_hex, 0)
        bucket_counts[base_hex] = within + 1
        color_hex = _vary_hsv(base_hex, within)

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


# KIRI Engine and most photogrammetry exporters use meters. The viewer assumes
# 1 unit = 1 mm (measurement.js treats distanceTo() as mm directly, ar.js bakes
# a 0.001 scale into the USDZ). Multiply OBJ vertices by 1000 so a 0.7m skull
# scan ends up at 700mm — measurable in the same units as the medical STLs.
_OBJ_METERS_TO_MM = 1000.0


def process_obj_bundle(
    obj_bytes: bytes,
    mtl_bytes: bytes,
    textures: dict[str, bytes],
    name: str,
) -> tuple[bytes, ProcessStats]:
    """Build a GLB from an OBJ + MTL + texture image bundle.

    `textures` maps image filename (as referenced in the MTL's `map_Kd`, etc.)
    to bytes. trimesh resolves these via the filesystem, so we materialize the
    bundle in a temp dir, load by path, then drop the temp dir.

    No decimation (would break UVs). No color override (preserve the texture).
    No RAS→Y-up rotation (OBJ photogrammetry is already Y-up).
    """
    if not obj_bytes:
        raise ValueError("Arquivo OBJ vazio.")
    if not mtl_bytes:
        raise ValueError("Arquivo MTL ausente — necessário para a textura.")
    if not textures:
        raise ValueError("Imagem(s) de textura ausente(s).")

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "model.obj")
        # The OBJ references its MTL by the `mtllib` directive (which is a
        # filename, not a path). Rewriting that name to a known value lets us
        # control the layout without parsing the OBJ.
        rewritten_obj = _rewrite_mtllib(obj_bytes, "model.mtl")
        with open(obj_path, "wb") as f:
            f.write(rewritten_obj)
        with open(os.path.join(tmpdir, "model.mtl"), "wb") as f:
            f.write(mtl_bytes)
        for tex_name, tex_data in textures.items():
            # Sanitize: only the basename matters for trimesh's resolver, and
            # we don't want a malicious bundle to write outside tmpdir.
            safe = os.path.basename(tex_name)
            if not safe:
                continue
            with open(os.path.join(tmpdir, safe), "wb") as f:
                f.write(tex_data)

        try:
            loaded = trimesh.load(obj_path, process=True)
        except Exception as e:
            raise ValueError(f"OBJ inválido ou corrompido: {e}") from e

        # OBJ can be one Trimesh (single `o`) or a Scene (multiple `o` directives).
        # Multi-`o` with distinct materials would lose textures on concatenation,
        # so we reject it in v1 — KIRI and similar tools always export a single
        # object so this is a clear "you uploaded the wrong thing" signal.
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise ValueError("OBJ não contém nenhuma malha utilizável.")
            if len(parts) > 1:
                raise ValueError(
                    "OBJ com múltiplos objetos não suportado — envie um único modelo."
                )
            mesh = parts[0]
        else:
            mesh = loaded

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Arquivo OBJ não pôde ser carregado como uma única malha.")

        input_tris = len(mesh.faces)

        # Meters → millimeters. Applied via a uniform scale matrix so UVs and
        # the TextureVisuals stay intact (a direct vertex multiply would also
        # work but applying as a transform is the idiomatic trimesh way and
        # keeps any future per-mesh transforms composable).
        mesh.apply_scale(_OBJ_METERS_TO_MM)
        # Force vertex_normals compute *after* the transform so the GLB exporter
        # emits the NORMAL attribute; same rationale as process_stls — without
        # this the viewer renders flat-shaded.
        _ = mesh.vertex_normals

    # Keep the texture as-is: trimesh's GLB exporter automatically converts
    # SimpleMaterial → PBRMaterial with `baseColorTexture` set. Don't overwrite
    # mesh.visual — the user explicitly asked to preserve the embedded texture.

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=name, geom_name=name)
    glb_bytes = scene.export(file_type="glb")
    output_tris = len(mesh.faces)

    return glb_bytes, ProcessStats(
        total_input_triangles=input_tris,
        total_output_triangles=output_tris,
        glb_size_bytes=len(glb_bytes),
        meshes=[
            MeshStats(
                name=name,
                input_triangles=input_tris,
                output_triangles=output_tris,
                decimated=False,
                color="",  # texture, not a flat color — main.py serializes "" → null
            )
        ],
    )


def _rewrite_mtllib(obj_bytes: bytes, new_mtl_name: str) -> bytes:
    """Replace the `mtllib <name>` directive at the top of an OBJ.

    Avoids parsing the OBJ as text — uses byte-level scan of the first ~4KB
    where the directive always appears (the rest is megabytes of vertices).
    """
    head = obj_bytes[:4096]
    tail = obj_bytes[4096:]
    lines = head.split(b"\n")
    for i, line in enumerate(lines):
        if line.startswith(b"mtllib "):
            lines[i] = f"mtllib {new_mtl_name}".encode()
            break
    return b"\n".join(lines) + tail
