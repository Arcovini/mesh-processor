"""Testes de segmentation.py e do caminho NRRD de segmentação → GLB.

Volumes sintéticos com geometria de propósito difícil (eixos oblíquos, origem
fora de zero, tamanhos e espaçamentos diferentes por eixo): uma transposição,
um eixo trocado ou um espelho LPS/RAS desloca a malha em dezenas de mm, e o
teste pega pelo centro da estrutura em LPS.
Run with: .venv/bin/python -m pytest test_segmentation.py -q
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
import trimesh

os.environ.setdefault("DRY_RUN", "true")

import nrrd  # noqa: E402

from processor import _RAS_TO_GLTF, process_stls  # noqa: E402
from segmentation import MAX_SEGMENTS, segments_from_nrrd  # noqa: E402

A = np.deg2rad(12)
ROT = np.array([[1, 0, 0], [0, np.cos(A), -np.sin(A)], [0, np.sin(A), np.cos(A)]])
SPACING = np.array([0.8, 1.0, 1.3])
DIRECTIONS = (ROT @ np.diag(SPACING)).T  # linhas = vetores de i, j, k (LPS)
SHAPE_IJK = (70, 60, 50)
ORIGIN = np.array([-25.0, 40.0, 100.0])

BALL_1 = (np.array([-5.0, 60.0, 125.0]), 12.0)
BALL_2 = (np.array([12.0, 75.0, 140.0]), 7.0)


def _world_grid(directions=DIRECTIONS, origin=ORIGIN, shape=SHAPE_IJK):
    ni, nj, nk = shape
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(float)
    return (origin + ijk @ directions).reshape(nk, nj, ni, 3)


def _labelmap(balls=(BALL_1, BALL_2), dtype=np.uint8):
    world = _world_grid()
    data = np.zeros(world.shape[:3], dtype=dtype)
    for value, (center, r) in enumerate(balls, start=1):
        data[np.linalg.norm(world - center, axis=-1) <= r] = value
    return data


def _nrrd_bytes(data, header=None, space="left-posterior-superior", directions=DIRECTIONS,
                origin=ORIGIN, encoding="gzip"):
    h = {
        "space": space,
        "space directions": directions,
        "space origin": origin,
        "kinds": ["domain"] * 3,
        "encoding": encoding,
    }
    h.update(header or {})
    buf = io.BytesIO()
    nrrd.write(buf, data, h, index_order="C")
    return buf.getvalue()


def _voxel_volume():
    return abs(np.linalg.det(DIRECTIONS))


# ---- Labelmap simples ----------------------------------------------------------


def test_plain_labelmap_one_closed_mesh_per_label_in_place():
    data = _labelmap()
    segs = segments_from_nrrd(_nrrd_bytes(data), "caso.nrrd")
    assert [s.name for s in segs] == ["Segmento 1", "Segmento 2"]
    for seg, (center, r), value in zip(segs, (BALL_1, BALL_2), (1, 2)):
        assert seg.mesh.is_watertight
        assert seg.mesh.volume > 0  # normais para fora
        assert seg.color is None
        assert seg.voxels == int((data == value).sum())
        # Mesmo lugar no espaço do paciente: o centro da malha cai no centro da bola.
        assert np.linalg.norm(seg.mesh.center_mass - center) < 0.5
        # O volume da malha fica perto do volume dos voxels.
        vox = seg.voxels * _voxel_volume()
        assert abs(seg.mesh.volume - vox) / vox < 0.01
        assert abs(seg.mesh.volume - 4 / 3 * np.pi * r**3) / (4 / 3 * np.pi * r**3) < 0.05


def test_single_label_is_named_after_the_file():
    data = _labelmap(balls=(BALL_1,))
    for filename, expected in [
        ("feto-label.nrrd", "feto"),
        ("Rim_esquerdo.seg.nrrd", "Rim_esquerdo"),
        ("Fígado mask.nrrd", "Figado"),
        ("label.nrrd", "Segmento 1"),
    ]:
        (seg,) = segments_from_nrrd(_nrrd_bytes(data), filename)
        assert seg.name == expected, filename


def test_ras_labelmap_lands_in_lps():
    """Mesmo volume declarado em RAS: x e y trocam de sinal na leitura."""
    data = _labelmap(balls=(BALL_1,))
    flip = np.array([-1.0, -1.0, 1.0])
    blob = _nrrd_bytes(data, space="right-anterior-superior",
                       directions=DIRECTIONS * flip, origin=ORIGIN * flip)
    (seg,) = segments_from_nrrd(blob, "ras.nrrd")
    assert np.linalg.norm(seg.mesh.center_mass - BALL_1[0]) < 0.5
    assert seg.mesh.volume > 0


def test_negative_determinant_geometry_is_not_inside_out():
    """Eixo k invertido (det < 0): a malha não pode sair do avesso."""
    dirs = DIRECTIONS.copy()
    dirs[2] *= -1
    world = _world_grid(directions=dirs)
    center, r = world[25, 30, 35], 9.0
    data = (np.linalg.norm(world - center, axis=-1) <= r).astype(np.uint8)
    (seg,) = segments_from_nrrd(_nrrd_bytes(data, directions=dirs), "m.nrrd")
    assert seg.mesh.volume > 0 and seg.mesh.is_watertight
    assert np.linalg.norm(seg.mesh.center_mass - center) < 0.5


def test_structure_touching_the_border_is_closed():
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[:, 5:15, 5:15] = 1  # atravessa o volume de ponta a ponta em k
    (seg,) = segments_from_nrrd(_nrrd_bytes(data, directions=np.eye(3), origin=np.zeros(3)), "b.nrrd")
    assert seg.mesh.is_watertight


def test_float_mask_is_accepted():
    data = _labelmap(balls=(BALL_1,)).astype(np.float32)
    (seg,) = segments_from_nrrd(_nrrd_bytes(data), "mask.nrrd")
    assert seg.mesh.is_watertight


@pytest.mark.parametrize("data,msg", [
    (np.full((10, 10, 10), -1000, dtype=np.int16), "negativos"),
    (np.random.default_rng(0).random((10, 10, 10)).astype(np.float32), "não inteiros"),
    (np.arange(1000, dtype=np.int16).reshape(10, 10, 10), "valores diferentes"),
    (np.zeros((10, 10, 10), dtype=np.uint8), "vazia"),
])
def test_image_or_empty_volume_is_refused(data, msg):
    with pytest.raises(ValueError, match=msg):
        segments_from_nrrd(_nrrd_bytes(data, directions=np.eye(3), origin=np.zeros(3)), "x.nrrd")


def test_limit_is_on_structures_not_on_dtype():
    data = np.zeros((MAX_SEGMENTS + 2, 4, 4), dtype=np.uint16)
    for v in range(1, MAX_SEGMENTS + 2):
        data[v, 1:3, 1:3] = v
    with pytest.raises(ValueError, match="valores diferentes"):
        segments_from_nrrd(_nrrd_bytes(data, directions=np.eye(3), origin=np.zeros(3)), "x.nrrd")


def test_not_a_nrrd_is_refused():
    with pytest.raises(ValueError, match="NRRD inválido"):
        segments_from_nrrd(b"solid nada\nendsolid\n", "x.nrrd")


# ---- .seg.nrrd do 3D Slicer -------------------------------------------------------


def _slicer_header(segments):
    h = {}
    for n, (name, color, layer, value) in enumerate(segments):
        h[f"Segment{n}_ID"] = f"Segment_{n + 1}"
        h[f"Segment{n}_Name"] = name
        h[f"Segment{n}_Color"] = " ".join(str(c) for c in color)
        h[f"Segment{n}_Layer"] = str(layer)
        h[f"Segment{n}_LabelValue"] = str(value)
        h[f"Segment{n}_Extent"] = "0 1 0 1 0 1"
    return h


def test_slicer_seg_names_colors_and_label_values():
    data = _labelmap()
    data[data == 2] = 7  # o LabelValue não precisa ser sequencial
    header = _slicer_header([
        ("UTERO", (0.9, 0.6, 0.5), 0, 1),
        ("CORACAO", (0.8, 0.1, 0.1), 0, 7),
        ("Sem voxels", (0.1, 0.1, 0.1), 0, 3),
    ])
    # O Slicer grava os nomes em UTF-8 (o pynrrd só escreve ASCII).
    blob = (_nrrd_bytes(data, header)
            .replace(b":=UTERO", ":=Útero".encode())
            .replace(b":=CORACAO", ":=Coração fetal".encode()))
    segs = segments_from_nrrd(blob, "Segmentation.seg.nrrd")
    assert [s.name for s in segs] == ["Utero", "Coracao fetal"]
    assert [s.color for s in segs] == ["#E69980", "#CC1A1A"]
    assert np.linalg.norm(segs[1].mesh.center_mass - BALL_2[0]) < 0.5


def test_slicer_layered_seg_with_overlapping_segments():
    """Segmentos que se sobrepõem: o Slicer grava 4D, camada no primeiro eixo."""
    world = _world_grid()
    big = (np.linalg.norm(world - BALL_1[0], axis=-1) <= BALL_1[1]).astype(np.uint8)
    small = (np.linalg.norm(world - BALL_1[0], axis=-1) <= 5.0).astype(np.uint8)
    data = np.stack([big, small], axis=-1)  # (k, j, i, camada) em ordem C
    header = _slicer_header([("Saco gestacional", (0.2, 0.4, 0.9), 0, 1), ("Embriao", (0.9, 0.9, 0.1), 1, 1)])
    header["kinds"] = ["list", "domain", "domain", "domain"]
    dirs4 = np.vstack([np.full(3, np.nan), DIRECTIONS])
    segs = segments_from_nrrd(_nrrd_bytes(data, header, directions=dirs4), "s.seg.nrrd")
    assert [s.name for s in segs] == ["Saco gestacional", "Embriao"]
    for s in segs:
        assert s.mesh.is_watertight
        assert np.linalg.norm(s.mesh.center_mass - BALL_1[0]) < 0.5
    assert segs[0].mesh.volume > 5 * segs[1].mesh.volume


def test_old_slicer_format_one_segment_per_layer():
    """Antes do Slicer 4.11: sem Layer/LabelValue, cada segmento é a sua camada."""
    world = _world_grid()
    a = (np.linalg.norm(world - BALL_1[0], axis=-1) <= BALL_1[1]).astype(np.uint8)
    b = (np.linalg.norm(world - BALL_2[0], axis=-1) <= BALL_2[1]).astype(np.uint8)
    header = {"Segment0_Name": "A", "Segment1_Name": "B", "kinds": ["list", "domain", "domain", "domain"]}
    dirs4 = np.vstack([np.full(3, np.nan), DIRECTIONS])
    segs = segments_from_nrrd(_nrrd_bytes(np.stack([a, b], axis=-1), header, directions=dirs4), "old.seg.nrrd")
    assert [s.name for s in segs] == ["A", "B"]
    assert np.linalg.norm(segs[1].mesh.center_mass - BALL_2[0]) < 0.5


# ---- GLB -----------------------------------------------------------------------------


def _glb_nodes(glb: bytes):
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    return {name: scene.geometry[g] for name, g in
            ((n, scene.graph[n][1]) for n in scene.graph.nodes_geometry)}


def test_process_stls_with_segmentation_builds_the_glb():
    data = _labelmap()
    header = _slicer_header([("Arteria uterina", (0.1, 0.9, 0.1), 0, 1), ("Placenta", (0.2, 0.4, 0.9), 0, 2)])
    glb, stats = process_stls([], segmentations=[("caso.seg.nrrd", _nrrd_bytes(data, header))])
    names = [m.name for m in stats.meshes]
    assert names == ["Arteria uterina", "Placenta"]
    # A keyword vence a cor do arquivo (artéria é vermelha em todo caso); sem
    # keyword vale a cor do Slicer, convertida de sRGB para o fator linear.
    assert stats.meshes[0].color == "#BD0006"
    assert stats.meshes[1].color == "#0822CA"
    nodes = _glb_nodes(glb)
    assert set(nodes) == set(names)
    # Mesma rotação dos STLs: o centro no GLB é o centro LPS girado.
    center_gltf = trimesh.transform_points([BALL_2[0]], _RAS_TO_GLTF)[0]
    assert np.linalg.norm(nodes["Placenta"].center_mass - center_gltf) < 0.5


def test_stl_and_segmentation_together_and_duplicate_names():
    m = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    stl = bytes(m.export(file_type="stl"))
    data = _labelmap(balls=(BALL_1,))
    _, stats = process_stls(
        [("Segmento 1", stl)],
        segmentations=[("a.nrrd", _nrrd_bytes(data)), ("b.nrrd", _nrrd_bytes(data))],
    )
    # "a" e "b" são de uma estrutura só: levam o nome do arquivo.
    assert [s.name for s in stats.meshes] == ["Segmento 1", "a", "b"]


def test_large_label_is_decimated():
    data = np.zeros((120, 120, 120), dtype=np.uint8)
    world = _world_grid(directions=np.eye(3) * 0.5, origin=np.zeros(3), shape=(120, 120, 120))
    data[np.linalg.norm(world - 30.0, axis=-1) <= 28] = 1
    _, stats = process_stls([], target_triangles_per_mesh=20_000,
                            segmentations=[("pele.nrrd", _nrrd_bytes(data, directions=np.eye(3) * 0.5,
                                                                     origin=np.zeros(3)))])
    (m,) = stats.meshes
    assert m.decimated and m.output_triangles <= 20_000 < m.input_triangles
    assert m.name == "pele"
    assert m.color == "#DC8576"  # "pele" casa com a paleta


# ---- Endpoint --------------------------------------------------------------------------


def test_upload_segmentation_and_image_nrrd_only(tmp_path, monkeypatch):
    """O pedido do caso: só NRRDs — a segmentação vira o 3D, a imagem o exame."""
    monkeypatch.setenv("DRY_RUN_GLB_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    seg = _nrrd_bytes(_labelmap(), _slicer_header([("Feto", (0.9, 0.7, 0.6), 0, 1), ("Placenta", (0.6, 0.3, 0.3), 0, 2)]))
    world = _world_grid()
    image = (1000 * np.exp(-np.linalg.norm(world - BALL_1[0], axis=-1) / 20)).astype(np.int16)
    r = client.post("/upload", files=[
        ("files", ("Segmentation.seg.nrrd", seg, "application/octet-stream")),
        ("exam", ("US 3D.nrrd", _nrrd_bytes(image), "application/octet-stream")),
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    assert [m["name"] for m in body["stats"]["meshes"]] == ["Feto", "Placenta"]
    assert body["exam"]["stored"] and body["exam"]["label"] == "US 3D"
    uid = body["uid"]
    assert (tmp_path / "cases" / f"{uid}.glb").exists()
    meta = json.loads((tmp_path / "cases" / f"{uid}.exam-0.json").read_text("utf-8"))
    assert meta["shape"] == list(SHAPE_IJK)


def test_upload_rejects_isolation_with_segmentation(monkeypatch, tmp_path):
    monkeypatch.setenv("DRY_RUN_GLB_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    stl = bytes(trimesh.creation.icosphere(radius=5.0).export(file_type="stl"))
    r = client.post("/upload", files=[
        ("files", ("rim.stl", stl, "application/octet-stream")),
        ("files", ("seg.nrrd", _nrrd_bytes(_labelmap()), "application/octet-stream")),
    ], data={"boolean_ops": json.dumps([{"principal": "rim.stl", "secondary": "seg.nrrd"}])})
    assert r.status_code == 400
    assert "só de arquivos STL" in r.json()["detail"]


def test_upload_image_sent_as_structures_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("DRY_RUN_GLB_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    ct = np.full((10, 10, 10), -1000, dtype=np.int16)
    r = client.post("/upload", files=[("files", ("tc.nrrd", _nrrd_bytes(ct), "application/octet-stream"))])
    assert r.status_code == 400
    assert "envie-o como exame" in r.json()["detail"]
