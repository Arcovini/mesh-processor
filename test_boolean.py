"""Testes das interações booleanas do pipeline STL (A intacta, B−A, B∩A).

Run with: .venv/bin/python -m pytest test_boolean.py -q
"""
from __future__ import annotations

import io
import os

import pytest
import trimesh

os.environ.setdefault("DRY_RUN", "true")  # main.py exige credenciais no import sem isso

from fastapi import HTTPException

from main import _parse_boolean_ops
from processor import process_stls


def _sphere_stl(radius: float, center=(0.0, 0.0, 0.0), subdivisions: int = 3) -> bytes:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.apply_translation(center)
    return m.export(file_type="stl")


RIM = _sphere_stl(10)                        # A: esfera na origem
TUMOR = _sphere_stl(6, center=(8, 0, 0))     # B: sobrepõe A parcialmente
LONGE = _sphere_stl(2, center=(100, 0, 0))   # não toca A
DENTRO = _sphere_stl(2)                      # inteiramente dentro de A


def test_boolean_splits_secondary_and_keeps_principal():
    files = [("Rim", RIM), ("Tumor", TUMOR)]
    _, stats_plain = process_stls(files)
    glb, stats = process_stls(files, boolean_ops=[("Rim", "Tumor")])

    assert [m.name for m in stats.meshes] == ["Rim", "Tumor", "Intersecao Tumor x Rim"]

    # A (principal) intacta: mesma contagem de triângulos do run sem booleana
    rim_plain = next(m for m in stats_plain.meshes if m.name == "Rim")
    rim = next(m for m in stats.meshes if m.name == "Rim")
    assert rim.output_triangles == rim_plain.output_triangles

    # Conservação: Volume(B−A) + Volume(B∩A) ≈ Volume(B original)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    vol_original_b = trimesh.load(io.BytesIO(TUMOR), file_type="stl").volume
    vol_cut = scene.geometry["Tumor"].volume
    vol_inter = scene.geometry["Intersecao Tumor x Rim"].volume
    assert vol_inter > 0
    assert vol_cut > 0
    assert abs((vol_cut + vol_inter) - vol_original_b) / vol_original_b < 0.01


def test_intersection_gets_highlight_color_over_tumor_green():
    # O nome composto contém "tumor" e "rim"; a keyword "intersec" deve vencer.
    _, stats = process_stls(
        [("Rim", RIM), ("Tumor", TUMOR)], boolean_ops=[("Rim", "Tumor")]
    )
    inter = next(m for m in stats.meshes if m.name.startswith("Intersecao"))
    assert inter.color == "#FFE100"


def test_disjoint_structures_raise():
    with pytest.raises(ValueError, match="não se tocam"):
        process_stls([("Rim", RIM), ("Osso", LONGE)], boolean_ops=[("Rim", "Osso")])


def test_secondary_fully_inside_principal_is_replaced_by_intersection():
    _, stats = process_stls(
        [("Rim", RIM), ("Lesao", DENTRO)], boolean_ops=[("Rim", "Lesao")]
    )
    assert [m.name for m in stats.meshes] == ["Rim", "Intersecao Lesao x Rim"]


def test_non_watertight_mesh_raises():
    m = trimesh.creation.icosphere(subdivisions=2, radius=6)
    m.apply_translation((8, 0, 0))
    m.faces = m.faces[:-1]  # abre um buraco
    furado = m.export(file_type="stl")
    with pytest.raises(ValueError, match="fechada"):
        process_stls([("Rim", RIM), ("Tumor", furado)], boolean_ops=[("Rim", "Tumor")])


# --- validação do form field boolean_ops (main._parse_boolean_ops) ---

FILENAMES = ["a.stl", "b.stl", "c.stl"]


def test_parse_ok_maps_filenames_to_indices():
    raw = '[{"principal": "a.stl", "secondary": "b.stl"}]'
    assert _parse_boolean_ops(raw, FILENAMES) == [(0, 1)]


def test_parse_empty_field_means_no_ops():
    assert _parse_boolean_ops("", FILENAMES) == []
    assert _parse_boolean_ops("  ", FILENAMES) == []


def test_parse_bad_json_raises_400():
    with pytest.raises(HTTPException) as e:
        _parse_boolean_ops("not json", FILENAMES)
    assert e.value.status_code == 400


def test_parse_unknown_filename_raises_400():
    with pytest.raises(HTTPException) as e:
        _parse_boolean_ops('[{"principal": "x.stl", "secondary": "b.stl"}]', FILENAMES)
    assert e.value.status_code == 400


def test_parse_same_structure_raises_400():
    with pytest.raises(HTTPException) as e:
        _parse_boolean_ops('[{"principal": "a.stl", "secondary": "a.stl"}]', FILENAMES)
    assert e.value.status_code == 400


def test_parse_duplicate_pair_raises_400():
    raw = (
        '[{"principal": "a.stl", "secondary": "b.stl"},'
        ' {"principal": "a.stl", "secondary": "b.stl"}]'
    )
    with pytest.raises(HTTPException) as e:
        _parse_boolean_ops(raw, FILENAMES)
    assert e.value.status_code == 400
