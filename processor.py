"""STL(s) -> optimized multi-mesh GLB with per-structure colors.

Pure module: no I/O, no env vars, no network.

Takes one or more (name, stl_bytes) pairs, builds a single GLB containing each
STL as a named mesh node with its own PBR material (so the viewer's opacity
slider — which acts on materials — works per-structure). Coordinates are
rotated from RAS (Z-up, medical convention) to glTF (Y-up).

Also exposes `process_obj_bundle` for OBJ uploads (single- or multi-object, with
an optional MTL + texture images). Coloring is material-aware, decided per object:
a textured object keeps its texture; an object with a flat MTL color keeps that
color; an object with no material at all falls through to the same keyword/palette
logic used for STL (one source of color config, in this module). That path skips
decimation (would break UV mapping) and scales meters → millimeters only when the
model looks like it's in meters (photogrammetry), leaving mm-authored OBJs alone.
"""
from __future__ import annotations

import colorsys
import io
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field

import fast_simplification
import numpy as np
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
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
    "pele": "#DC8576",    # pele: rosado avermelhado
    "cortex": "#966830",  # córtex: marrom
    "osso": "#EAE3D2",    # osso: marfim / off-white (mais claro que a pele)
}

# Structures whose name contains "metal" (implants, screws, plates, stents) get
# a polished silver/titanium PBR finish instead of a flat anatomical color:
# full metalness + low roughness so the scene's environment map reads as shiny
# metal in the viewer. The base hex still flows through the duplicate-bucket /
# HSV logic, so two distinct metal parts in one case stay distinguishable.
METAL_KEYWORD = "metal"
METAL_COLOR = "#C0C4C8"   # neutral steel/titanium gray
METAL_METALLIC = 1.0
METAL_ROUGHNESS = 0.25

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


def _name_based_material(
    name: str, bucket_counts: dict[str, int], fallback_idx: int
) -> tuple[str, PBRMaterial, int]:
    """Color + PBR finish for a mesh that has no material of its own, from its name.

    Single source of name-based coloring for BOTH STL meshes and material-less OBJ
    objects: `metal` → polished silver/titanium (full metalness, low roughness);
    other keyword → palette color; unmatched → cycling fallback palette. Duplicates
    of a base hex are HSV-varied via `bucket_counts` so they stay distinguishable.

    Returns (color_hex, material, next_fallback_idx).
    """
    lower = name.lower()
    is_metal = METAL_KEYWORD in lower
    # `metal` counts as a keyword match (so it doesn't consume/shift a fallback
    # slot) but uses a fixed silver base instead of an anatomical color.
    matched = is_metal or any(k in lower for k in COLORS_BY_KEYWORD)
    base_hex = METAL_COLOR if is_metal else _pick_color(name, fallback_idx)
    if not matched:
        fallback_idx += 1
    within = bucket_counts.get(base_hex, 0)
    bucket_counts[base_hex] = within + 1
    color_hex = _vary_hsv(base_hex, within)
    r, g, b = _hex_to_rgb01(color_hex)
    material = PBRMaterial(
        name=name,
        baseColorFactor=[r, g, b, 1.0],
        metallicFactor=METAL_METALLIC if is_metal else 0.0,
        roughnessFactor=METAL_ROUGHNESS if is_metal else 0.5,
    )
    return color_hex, material, fallback_idx


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

        color_hex, material, fallback_idx = _name_based_material(
            name, bucket_counts, fallback_idx
        )
        mesh.visual = TextureVisuals(material=material)

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
# a 0.001 scale into the USDZ). Multiply such OBJ vertices by 1000 so a 0.7m skull
# scan ends up at 700mm — measurable in the same units as the medical STLs.
_OBJ_METERS_TO_MM = 1000.0

# Below this max bounding-box extent we assume the OBJ is in meters (photogrammetry)
# and scale ×1000; above it we assume millimetres (a medically-authored OBJ) and
# leave it alone. 10 units is a safe gap: meter-scale scans sit under ~2, while
# anatomy in mm sits well above tens. Avoids 1000×-blowing-up an mm-authored OBJ.
_OBJ_UNIT_METERS_THRESHOLD = 10.0

# trimesh's SimpleMaterial default diffuse — a mesh carrying exactly this (and no
# texture) had no real material, so we treat it as "uncoloured" → keyword palette.
_TRIMESH_DEFAULT_DIFFUSE = (102, 102, 102)

# Keep objects separate (one geometry per `o`, keyed by object name) AND keep each
# object's own material — without merging by material or concatenating into one mesh.
_OBJ_LOAD_OPTS = dict(process=False, split_object=True, group_material=False)


def _clean_part_name(raw: str) -> str:
    """Transliterate accents + strip separators off an OBJ object name."""
    s = "".join(
        c for c in unicodedata.normalize("NFKD", raw or "") if not unicodedata.combining(c)
    )
    return s.strip(" _/") or "mesh"


def _part_material_kind(visual) -> tuple[str, object]:
    """Classify a loaded object's visual: ('texture'|'color'|'none', material)."""
    mat = getattr(visual, "material", None)
    if mat is None or isinstance(visual, ColorVisuals):
        return "none", None
    if getattr(mat, "image", None) is not None:
        return "texture", mat
    diffuse = getattr(mat, "diffuse", None)
    if diffuse is not None:
        if tuple(int(c) for c in diffuse[:3]) == _TRIMESH_DEFAULT_DIFFUSE:
            return "none", None   # trimesh default sentinel = effectively uncoloured
        return "color", mat
    if getattr(mat, "baseColorFactor", None) is not None:
        return "color", mat
    return "none", None


def _flat_material_rgb01(mat) -> tuple[float, float, float]:
    """Flat colour of a material as 0..1 RGB (from MTL Kd / baseColorFactor)."""
    diffuse = getattr(mat, "diffuse", None)
    if diffuse is not None:
        return (int(diffuse[0]) / 255.0, int(diffuse[1]) / 255.0, int(diffuse[2]) / 255.0)
    bcf = mat.baseColorFactor
    return (float(bcf[0]), float(bcf[1]), float(bcf[2]))


def _rgb01_to_hex(rgb01: tuple[float, float, float]) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, round(c * 255))) for c in rgb01)


def _flat_pbr_material(name: str, rgb01: tuple[float, float, float]) -> PBRMaterial:
    """A dielectric PBR material with the given colour — same finish as the STL path."""
    return PBRMaterial(
        name=name,
        baseColorFactor=[rgb01[0], rgb01[1], rgb01[2], 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.5,
    )


def process_obj_bundle(
    obj_bytes: bytes,
    mtl_bytes: bytes | None,
    textures: dict[str, bytes],
    name: str,
) -> tuple[bytes, ProcessStats]:
    """Build a GLB from an OBJ upload (single- or multi-object), colour per object.

    `mtl_bytes` is optional; `textures` maps image filename (as referenced in the
    MTL's `map_Kd`) to bytes. The bundle is materialized in a temp dir so trimesh
    can resolve the MTL/images by path. Per object: texture → kept; flat MTL colour
    → kept; nothing → keyword/palette colour (same source as STL). No decimation
    (would break UVs). No RAS→Y-up rotation (OBJ is already Y-up).
    """
    if not obj_bytes:
        raise ValueError("Arquivo OBJ vazio.")

    # Did the MTL ask for a texture? If so and none loads, that's the Pillow-missing
    # silent-texture-loss bug — fail loudly. (Without a map_Kd, "no image" is valid.)
    mtl_has_map_kd = bool(mtl_bytes) and re.search(rb"(?mi)^\s*map_Kd\b", mtl_bytes) is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, "model.obj")
        if mtl_bytes:
            # The OBJ references its MTL by the `mtllib` directive (a filename, not
            # a path). Rewrite it to a known value to control the temp-dir layout.
            rewritten_obj = _rewrite_mtllib(obj_bytes, "model.mtl")
            with open(os.path.join(tmpdir, "model.mtl"), "wb") as f:
                f.write(mtl_bytes)
        else:
            rewritten_obj = obj_bytes
        with open(obj_path, "wb") as f:
            f.write(rewritten_obj)
        for tex_name, tex_data in textures.items():
            # Sanitize: only the basename matters for trimesh's resolver, and we
            # don't want a malicious bundle to write outside tmpdir.
            safe = os.path.basename(tex_name)
            if not safe:
                continue
            with open(os.path.join(tmpdir, safe), "wb") as f:
                f.write(tex_data)

        try:
            loaded = trimesh.load(obj_path, **_OBJ_LOAD_OPTS)
        except Exception as e:
            raise ValueError(f"OBJ inválido ou corrompido: {e}") from e

        # Normalize to [(part_name, Trimesh)]. Multi-object OBJ → Scene keyed by the
        # `o` names; single object → a bare Trimesh that takes the upload filename.
        if isinstance(loaded, trimesh.Scene):
            parts = [
                (k, g) for k, g in loaded.geometry.items() if isinstance(g, trimesh.Trimesh)
            ]
            if not parts:
                raise ValueError("OBJ não contém nenhuma malha utilizável.")
            if len(parts) == 1:
                parts = [(name, parts[0][1])]
        elif isinstance(loaded, trimesh.Trimesh):
            parts = [(name, loaded)]
        else:
            raise ValueError("Arquivo OBJ não pôde ser carregado como malha.")

        # Unit heuristic over the whole model, so every part gets the SAME scale and
        # stays aligned. Meters (small) → ×1000; mm (large) → untouched.
        lo = np.min([g.bounds[0] for _, g in parts], axis=0)
        hi = np.max([g.bounds[1] for _, g in parts], axis=0)
        max_extent = float(np.max(hi - lo))
        scale = _OBJ_METERS_TO_MM if 0 < max_extent < _OBJ_UNIT_METERS_THRESHOLD else 1.0

        scene = trimesh.Scene()
        mesh_stats: list[MeshStats] = []
        bucket_counts: dict[str, int] = {}
        fallback_idx = 0
        used_names: set[str] = set()
        any_texture = False

        for raw_name, mesh in parts:
            if scale != 1.0:
                mesh.apply_scale(scale)
            # Force vertex_normals after the scale so the GLB exporter emits NORMAL.
            _ = mesh.vertex_normals

            part_name = _clean_part_name(raw_name)
            unique = part_name
            suffix = 2
            while unique in used_names:
                unique = f"{part_name}_{suffix}"
                suffix += 1
            part_name = unique
            used_names.add(part_name)

            kind, mat = _part_material_kind(mesh.visual)
            if kind == "texture":
                any_texture = True
                color_hex = ""   # the texture carries the look → null swatch in viewer
            elif kind == "color":
                rgb01 = _flat_material_rgb01(mat)
                color_hex = _rgb01_to_hex(rgb01)
                mesh.visual = TextureVisuals(material=_flat_pbr_material(part_name, rgb01))
            else:  # 'none' → same name-based color + finish as STL (incl. metal)
                color_hex, material, fallback_idx = _name_based_material(
                    part_name, bucket_counts, fallback_idx
                )
                mesh.visual = TextureVisuals(material=material)

            scene.add_geometry(mesh, node_name=part_name, geom_name=part_name)
            tris = len(mesh.faces)
            mesh_stats.append(
                MeshStats(
                    name=part_name,
                    input_triangles=tris,
                    output_triangles=tris,
                    decimated=False,
                    color=color_hex,
                )
            )

        # Silent-texture-loss guard (Pillow missing / unresolved image): the MTL
        # asked for a texture but none came through.
        if mtl_has_map_kd and not any_texture:
            raise ValueError(
                "Textura do OBJ não pôde ser carregada — verifique se o MTL referencia "
                "uma imagem (.jpg/.png) e se ela foi enviada junto."
            )

    glb_bytes = scene.export(file_type="glb")
    total_tris = sum(m.input_triangles for m in mesh_stats)

    return glb_bytes, ProcessStats(
        total_input_triangles=total_tris,
        total_output_triangles=total_tris,
        glb_size_bytes=len(glb_bytes),
        meshes=mesh_stats,
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
