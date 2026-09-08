"""Self-contained pytest for the generalized OBJ path: multi-object, material-aware.

Synthesizes OBJ/MTL/PNG in memory — no external fixtures. Asserts the per-object
rule: texture → keep texture; material (Kd) → keep color; nothing → keyword palette
(same as STL). Plus the bbox unit heuristic and the texture-loss guard.

Run: .venv/bin/python -m pytest test_obj_materials.py -q
"""
from __future__ import annotations

import io
import json
import struct

import pytest
import trimesh
from PIL import Image

from processor import process_obj_bundle, process_stls

# --- OBJ builders --------------------------------------------------------------

_CUBE_FACES = [
    (1, 2, 3), (1, 3, 4), (5, 7, 6), (5, 8, 7), (1, 6, 2), (1, 5, 6),
    (2, 7, 3), (2, 6, 7), (3, 8, 4), (3, 7, 8), (4, 5, 1), (4, 8, 5),
]


def _cube(name, base, offset, size, usemtl=None):
    ox, oy, oz = offset
    verts = [(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0),
             (0, 0, size), (size, 0, size), (size, size, size), (0, size, size)]
    lines = [f"o {name}"]
    if usemtl:
        lines.append(f"usemtl {usemtl}")
    lines += [f"v {x+ox} {y+oy} {z+oz}" for x, y, z in verts]
    lines += [f"f {a+base} {b+base} {c+base}" for a, b, c in _CUBE_FACES]
    return "\n".join(lines) + "\n", base + 8


def _textured_quad(name="quad", usemtl="TexMat"):
    return (
        f"o {name}\nusemtl {usemtl}\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
    )


def _png_bytes(color=(200, 30, 30), size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _reload(glb_bytes):
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    if isinstance(scene, trimesh.Scene):
        return list(scene.geometry.items())
    return [("<single>", scene)]


def _by_name(stats):
    return {m.name: m for m in stats.meshes}


def _glb_materials(glb_bytes):
    """Parse the GLB JSON chunk → {material_name: pbrMetallicRoughness dict}."""
    clen, _ = struct.unpack("<II", glb_bytes[12:20])
    g = json.loads(glb_bytes[20:20 + clen])
    return {m.get("name"): m.get("pbrMetallicRoughness", {}) for m in g.get("materials", [])}


# --- Tests ---------------------------------------------------------------------

def test_multiobject_named_meshes_with_distinct_colors():
    """Two objects, each its own MTL color → two named meshes preserving colors."""
    mtl = "newmtl Bone\nKd 0.90 0.90 0.85\n\nnewmtl Vessel\nKd 0.80 0.10 0.10\n"
    a, n = _cube("tibia", 0, (0, 0, 0), 50, usemtl="Bone")
    b, n = _cube("vaso", n, (80, 0, 0), 50, usemtl="Vessel")
    obj = "mtllib model.mtl\n" + a + b
    glb, stats = process_obj_bundle(obj.encode(), mtl.encode(), {}, "modelo")

    names = _by_name(stats)
    assert set(names) == {"tibia", "vaso"}, names
    # Bone Kd 0.90/0.90/0.85 -> #E6E6D9 ; Vessel 0.80/0.10/0.10 -> #CC1A1A
    assert names["tibia"].color.upper() == "#E6E6D9"
    assert names["vaso"].color.upper() == "#CC1A1A"
    assert len(_reload(glb)) == 2


def test_object_without_material_falls_back_to_keyword_skin():
    """Single no-material OBJ → named by the upload filename (like STL); 'pele*' → skin."""
    a, n = _cube("mesh1", 0, (0, 0, 0), 50)   # internal `o` name is irrelevant for a single object
    glb, stats = process_obj_bundle(a.encode(), None, {}, "pele_perna")  # filename drives it
    names = _by_name(stats)
    assert "pele_perna" in names
    assert names["pele_perna"].color.upper() == "#DC8576"  # COLORS_BY_KEYWORD["pele"]


def test_mixed_some_with_material_some_without():
    """Object with Kd preserved; sibling without any material → keyword/fallback."""
    mtl = "newmtl Vessel\nKd 0.80 0.10 0.10\n"
    # 'pele' object first (before any usemtl → no material), then a vessel with Kd.
    a, n = _cube("pele", 0, (0, 0, 0), 50)
    b, n = _cube("vaso", n, (80, 0, 0), 50, usemtl="Vessel")
    obj = "mtllib model.mtl\n" + a + b
    glb, stats = process_obj_bundle(obj.encode(), mtl.encode(), {}, "modelo")
    names = _by_name(stats)
    assert names["pele"].color.upper() == "#DC8576"      # keyword skin
    assert names["vaso"].color.upper() == "#CC1A1A"      # preserved Kd


def test_single_textured_object_preserves_texture():
    """Single textured OBJ (current KIRI case) still embeds the texture, color=null."""
    mtl = "newmtl TexMat\nKd 1 1 1\nmap_Kd tex.png\n"
    obj = "mtllib model.mtl\n" + _textured_quad()
    glb, stats = process_obj_bundle(obj.encode(), mtl.encode(), {"tex.png": _png_bytes()}, "scan")
    assert len(stats.meshes) == 1
    assert stats.meshes[0].color == ""  # texture, not a flat color
    geoms = _reload(glb)
    assert len(geoms) == 1
    mat = geoms[0][1].visual.material
    assert getattr(mat, "baseColorTexture", None) is not None


def test_unit_heuristic_meters_scaled_mm_kept():
    """~0.3-unit model (meters) → ~300mm; ~300-unit model (mm) → stays ~300mm."""
    meters_obj, _ = _cube("scan", 0, (0, 0, 0), 0.3)
    glb_m, _ = process_obj_bundle(meters_obj.encode(), None, {}, "scan")
    ext_m = _reload(glb_m)[0][1].extents.max()
    assert 250 < ext_m < 350, f"meters→mm broke: {ext_m}"

    mm_obj, _ = _cube("scan", 0, (0, 0, 0), 300.0)
    glb_mm, _ = process_obj_bundle(mm_obj.encode(), None, {}, "scan")
    ext_mm = _reload(glb_mm)[0][1].extents.max()
    assert 250 < ext_mm < 350, f"mm over-scaled: {ext_mm}"


def test_texture_guard_only_fires_when_map_kd_referenced():
    """map_Kd referenced but image missing → raise. Kd-only + no image → fine."""
    mtl_map = "newmtl TexMat\nKd 1 1 1\nmap_Kd missing.png\n"
    obj = "mtllib model.mtl\n" + _textured_quad()
    with pytest.raises(ValueError):
        process_obj_bundle(obj.encode(), mtl_map.encode(), {}, "scan")  # texture not supplied

    mtl_flat = "newmtl FlatMat\nKd 0.20 0.40 0.80\n"
    a, n = _cube("bloco", 0, (0, 0, 0), 50, usemtl="FlatMat")
    obj2 = "mtllib model.mtl\n" + a
    glb, stats = process_obj_bundle(obj2.encode(), mtl_flat.encode(), {}, "scan")
    assert stats.meshes[0].color.upper() == "#3366CC"  # 0.2/0.4/0.8


def test_stl_path_unchanged_keyword_color():
    """STL has no material → keyword palette, exactly as before."""
    stl_bytes = trimesh.creation.box(extents=(50, 50, 50)).export(file_type="stl")
    _, stats = process_stls([("pele", stl_bytes)])
    assert stats.meshes[0].color.upper() == "#DC8576"


def test_stl_bone_keyword_offwhite():
    """STL named '*osso*' → off-white bone tone (keyword palette)."""
    stl_bytes = trimesh.creation.box(extents=(50, 50, 50)).export(file_type="stl")
    _, stats = process_stls([("osso_cortical", stl_bytes)])
    assert stats.meshes[0].color.upper() == "#EAE3D2"


def _box_stl():
    return trimesh.creation.box(extents=(50, 50, 50)).export(file_type="stl")


def test_stl_kidney_keyword_matches_rim_brown():
    """STL named '*kidney*' → mesmo marrom de '*rim*'; par esquerdo/direito varia HSV."""
    stl_bytes = _box_stl()
    _, stats = process_stls([("Left Kidney", stl_bytes)])
    assert stats.meshes[0].color.upper() == "#BA5531"  # COLORS_BY_KEYWORD["rim"]

    _, stats2 = process_stls([("rim_direito", stl_bytes), ("Left Kidney", stl_bytes)])
    colors = [m.color.upper() for m in stats2.meshes]
    assert colors[0] == "#BA5531"          # primeiro do bucket = hex base
    assert colors[1] != colors[0]          # segundo varia (mesmo bucket que 'rim')


@pytest.mark.parametrize(
    "name",
    ["Rim Direito", "Rins", "Kidney", "Kidneys", "Parenquima Renal", "Vasos Renais"],
)
def test_kidney_variants_all_land_on_the_same_brown(name):
    """Todas as variações PT/EN caem no mesmo marrom base (bucket compartilhado)."""
    _, stats = process_stls([(name, _box_stl())])
    assert stats.meshes[0].color.upper() == "#BA5531"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Arteria Renal", "#BD0006"),   # 'art' vem antes → vaso, não rim
        ("Veia Renal", "#477EFF"),      # 'vei' vem antes → vaso
        ("Tumor Renal", "#08E700"),     # 'tumor' vem antes → verde de lesão
        ("Lesao Renal", "#08E700"),
        ("Cortex Renal", "#966830"),    # 'cortex' vem antes → marrom do córtex
    ],
)
def test_renal_adjective_does_not_steal_more_specific_keywords(name, expected):
    """`renal`/`renais` ficam por último: keyword mais específica vence."""
    _, stats = process_stls([(name, _box_stl())])
    assert stats.meshes[0].color.upper() == expected


@pytest.mark.parametrize(
    "name",
    [
        "Tumor",
        "Tumor de Rim",       # 'rim' casaria antes se `tumor` não estivesse no topo
        "Kidney Tumor",
        "Tumor Renal",
        "Tumor no Osso",
        "Tumor de Pele",
        "Lesao de Rim",
        "Lesao Renal",
    ],
)
def test_tumor_always_wins_over_the_host_organ(name):
    """Regra de produto: nome com 'tumor' (ou 'lesao') é sempre verde."""
    _, stats = process_stls([(name, _box_stl())])
    assert stats.meshes[0].color.upper() == "#08E700"


def test_tumor_outranks_vessel_keywords_too():
    """Consequência deliberada de "tumor ⇒ verde": vale até sobre 'art'/'vei'."""
    stl_bytes = _box_stl()
    _, stats = process_stls([("Arteria do Tumor", stl_bytes), ("Veia do Tumor", stl_bytes)])
    assert [m.color.upper() for m in stats.meshes][0] == "#08E700"
    assert all(m.color.upper() != "#BD0006" for m in stats.meshes)


@pytest.mark.parametrize("name", ["Adrenal", "Glandula Suprarrenal", "Adrenais"])
def test_adrenal_gland_is_not_painted_kidney_brown(name):
    """Glândula adrenal contém 'renal' mas é outro órgão → veto, cai no fallback."""
    _, stats = process_stls([(name, _box_stl())])
    assert stats.meshes[0].color.upper() != "#BA5531"


def test_vetoed_names_still_advance_the_fallback_palette():
    """Veto devolve ao fallback E consome slot — duas adrenais não saem iguais."""
    stl_bytes = _box_stl()
    _, stats = process_stls([("Adrenal Direita", stl_bytes), ("Adrenal Esquerda", stl_bytes)])
    colors = [m.color.upper() for m in stats.meshes]
    assert colors[0] != colors[1]


def test_stl_metal_name_gets_metallic_finish():
    """STL named '*metal*' → silver base + metallicFactor=1 (b400060 behavior preserved)."""
    stl_bytes = trimesh.creation.box(extents=(20, 20, 20)).export(file_type="stl")
    glb, stats = process_stls([("placa_metal", stl_bytes)])
    assert stats.meshes[0].color.upper() == "#C0C4C8"
    pbr = _glb_materials(glb)["placa_metal"]
    assert pbr["metallicFactor"] == 1.0
    assert pbr["roughnessFactor"] == 0.25


def test_obj_no_material_metal_name_gets_metallic_finish():
    """OBJ object with no material, named '*metal*' → same metallic finish (shared path)."""
    a, n = _cube("mesh1", 0, (0, 0, 0), 50)
    glb, stats = process_obj_bundle(a.encode(), None, {}, "parafuso_metal")
    assert stats.meshes[0].color.upper() == "#C0C4C8"
    pbr = _glb_materials(glb)["parafuso_metal"]
    assert pbr["metallicFactor"] == 1.0
    assert pbr["roughnessFactor"] == 0.25


def test_obj_with_mtl_color_on_metal_name_keeps_material_not_finish():
    """Precedence: an object that HAS an MTL color keeps it (dielectric), even if
    named 'metal' — file material wins over name-based finish."""
    mtl = "newmtl Custom\nKd 0.20 0.40 0.80\n"
    a, n = _cube("mesh1", 0, (0, 0, 0), 50, usemtl="Custom")
    obj = "mtllib model.mtl\n" + a
    glb, stats = process_obj_bundle(obj.encode(), mtl.encode(), {}, "metal_pintado")  # filename has 'metal'
    assert stats.meshes[0].color.upper() == "#3366CC"   # the MTL color, not silver
    pbr = _glb_materials(glb)["metal_pintado"]
    assert pbr.get("metallicFactor", 0.0) == 0.0        # dielectric — material wins


def test_routing_accepts_obj_without_mtl_or_texture():
    import os
    os.environ["DRY_RUN"] = "true"
    from main import _extract_obj_bundle
    from fastapi import HTTPException

    obj, _ = _cube("m", 0, (0, 0, 0), 50)
    # OBJ alone (no MTL, no texture) is now valid.
    ob, mb, tex, nm = _extract_obj_bundle([("m.obj", obj.encode())])
    assert mb is None and tex == {} and nm == "m.obj"

    # OBJ + MTL, no texture → valid.
    ob, mb, tex, nm = _extract_obj_bundle([("m.obj", obj.encode()), ("m.mtl", b"newmtl X\nKd 1 1 1\n")])
    assert mb is not None and tex == {}

    # Two OBJs → rejected.
    with pytest.raises(HTTPException):
        _extract_obj_bundle([("a.obj", obj.encode()), ("b.obj", obj.encode())])
