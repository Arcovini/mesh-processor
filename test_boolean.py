"""Testes da divisão de estruturas (dentro/fora) do pipeline STL.

Semântica: a referência A fica inteira; B é separada em "B fora de A" e
"B dentro de A" (destacada). Run with: .venv/bin/python -m pytest test_boolean.py -q
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

    assert [m.name for m in stats.meshes] == [
        "Rim",
        "Tumor fora de Rim",
        "Tumor dentro de Rim",
    ]

    # A (referência) fica inteira: mesma contagem de triângulos do run sem divisão
    rim_plain = next(m for m in stats_plain.meshes if m.name == "Rim")
    rim = next(m for m in stats.meshes if m.name == "Rim")
    assert rim.output_triangles == rim_plain.output_triangles

    # Conservação: Volume(fora) + Volume(dentro) ≈ Volume(B original)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    vol_original_b = trimesh.load(io.BytesIO(TUMOR), file_type="stl").volume
    vol_fora = scene.geometry["Tumor fora de Rim"].volume
    vol_dentro = scene.geometry["Tumor dentro de Rim"].volume
    assert vol_dentro > 0
    assert vol_fora > 0
    assert abs((vol_fora + vol_dentro) - vol_original_b) / vol_original_b < 0.01


def test_inner_piece_gets_highlight_color_over_tumor_green():
    # O nome composto contém "tumor" e "rim"; a keyword "dentro de" deve vencer.
    # Já "Tumor fora de Rim" mantém o verde do tumor de propósito.
    _, stats = process_stls(
        [("Rim", RIM), ("Tumor", TUMOR)], boolean_ops=[("Rim", "Tumor")]
    )
    dentro = next(m for m in stats.meshes if "dentro de" in m.name)
    fora = next(m for m in stats.meshes if "fora de" in m.name)
    assert dentro.color == "#FFE100"
    assert fora.color == "#08E700"


def test_disjoint_structures_raise():
    with pytest.raises(ValueError, match="não se sobrepõem"):
        process_stls([("Rim", RIM), ("Osso", LONGE)], boolean_ops=[("Rim", "Osso")])


def test_secondary_fully_inside_principal_is_replaced_by_inner_piece():
    _, stats = process_stls(
        [("Rim", RIM), ("Lesao", DENTRO)], boolean_ops=[("Rim", "Lesao")]
    )
    assert [m.name for m in stats.meshes] == ["Rim", "Lesao dentro de Rim"]


def test_chained_divisions_reuse_previous_result():
    """A peça externa de uma divisão pode ser dividida de novo.

    Regressão: o manifold3d devolve faces degeneradas que faziam o resultado
    parecer não-estanque, então a segunda divisão falhava na checagem de malha
    fechada (ver `_clean_boolean_result`).
    """
    coluna = trimesh.creation.box(extents=[8, 8, 40])
    coluna.apply_translation((16, 0, 0))
    files = [("Rim", RIM), ("Tumor", TUMOR), ("Coluna", coluna.export(file_type="stl"))]
    _, stats = process_stls(
        files, boolean_ops=[("Rim", "Tumor"), ("Coluna", "Tumor")]
    )
    names = [m.name for m in stats.meshes]
    assert names == [
        "Rim",
        "Tumor fora de Rim fora de Coluna",
        "Tumor fora de Rim dentro de Coluna",
        "Tumor dentro de Rim",
        "Coluna",
    ]
    # Ambas as peças "dentro de" são destaque: a segunda varia o HSV do amarelo
    # (mesmo bucket de cor), então basta que nenhuma caia na paleta de fallback.
    dentro = [m.color for m in stats.meshes if "dentro de" in m.name]
    assert dentro[0] == "#FFE100"
    assert len(set(dentro)) == 2


def test_boolean_results_are_watertight_so_volume_stays_measurable():
    # O viewer mede volume por soma de tetraedros: peças abertas dariam medida
    # errada (e o modo volume marcaria com "~").
    glb, _ = process_stls(
        [("Rim", RIM), ("Tumor", TUMOR)], boolean_ops=[("Rim", "Tumor")]
    )
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    for name in ("Tumor fora de Rim", "Tumor dentro de Rim"):
        assert scene.geometry[name].is_watertight, f"{name} não é estanque"


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
