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

from processor import _RAS_TO_GLTF, _decimate, _is_closed, process_stls  # noqa: E402
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


# ---- Estruturas finas e em ilhas ----------------------------------------------
# Casos de uma RM fetal real (2026-10-04) que o desfoque e a busca do nível
# erravam: uma casca de 1 voxel de espessura saía com 45 % a menos de volume,
# e as ilhas pequenas de uma estrutura sumiam (o volume ia para as outras). As
# máscaras são construídas nos índices: "1 voxel" é a espessura que importa.


def _index_mask(fn, shape=SHAPE_IJK):
    ni, nj, nk = shape
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    return fn(i.astype(float), j.astype(float), k.astype(float)).astype(np.uint8)


def _one_segment(data):
    (seg,) = segments_from_nrrd(_nrrd_bytes(data), "estrutura.nrrd")
    vox = seg.voxels * _voxel_volume()
    return seg, (seg.mesh.volume - vox) / vox


def test_one_voxel_shell_keeps_its_volume_and_its_wall():
    # Esfera oca, parede de 1 voxel (|r − R| ≤ ½): ligada só em diagonal em
    # boa parte dela — a 0,8 de desfoque a parede sai furada.
    data = _index_mask(lambda i, j, k: np.abs(np.sqrt((i - 34) ** 2 + (j - 30) ** 2 + (k - 25) ** 2) - 20) <= 0.5)
    seg, error = _one_segment(data)
    assert abs(error) < 0.05
    assert _is_closed(seg.mesh.vertices, seg.mesh.faces)
    # Parede inteira: a superfície de fora e a de dentro, sem furo entre elas.
    assert seg.mesh.body_count == 2 and seg.mesh.euler_number == 4


@pytest.mark.parametrize("name, fn", [
    ("placa de 1 voxel", lambda i, j, k: (k == 25) & (np.abs(i - 34) < 18) & (np.abs(j - 30) < 18)),
    ("fio de 1 voxel", lambda i, j, k: (i == 34) & (j == 30) & (np.abs(k - 25) < 20)),
])
def test_thin_structure_keeps_its_volume(name, fn):
    seg, error = _one_segment(_index_mask(fn))
    assert abs(error) < 0.02, name
    assert _is_closed(seg.mesh.vertices, seg.mesh.faces), name
    assert seg.mesh.body_count == 1, name


def test_small_islands_are_all_kept():
    """Uma estrutura em pedaços (o cordão em ilhas): cada ilha fica, no lugar."""
    ball = lambda i, j, k: (i - 34) ** 2 + (j - 30) ** 2 + (k - 25) ** 2 <= 10 ** 2  # noqa: E731
    data = _index_mask(ball)
    corners = [(i, j, k) for i in (8, 58) for j in (8, 50) for k in (6, 42)]
    for i, j, k in corners:
        data[k:k + 2, j:j + 2, i:i + 2] = 1  # 8 voxels cada
    seg, error = _one_segment(data)
    assert abs(error) < 0.01
    assert seg.mesh.body_count == 1 + len(corners)
    world = _world_grid()
    centroid = world[data.astype(bool)].mean(axis=0)
    assert np.linalg.norm(seg.mesh.center_mass - centroid) < 0.5


# ---- Decimação mantém a malha fechada -------------------------------------------
# O fast_simplification não checa topologia: onde a estrutura é mais fina que
# as arestas novas, as duas faces dela colapsam uma na outra e a malha abre
# (num caso real, 618 mil → 300 mil triângulos). Aberta, o volume do
# visualizador vira "~" e o Cortar recusa a estrutura.


def _thin_blob_mesh():
    """Ruído desfocado e cortado: paredes e pontes de 1–3 voxels. Com esta
    semente, decimar para a metade sem cuidado abre a malha."""
    from scipy.ndimage import gaussian_filter

    from segmentation import _surface

    field = gaussian_filter(np.random.default_rng(2).random((40, 40, 40)), 2)
    mask = field > field.min() + 0.5 * (field.max() - field.min())
    mask[:2] = mask[-2:] = False
    mask[:, :2] = mask[:, -2:] = False
    mask[:, :, :2] = mask[:, :, -2:] = False
    return _surface(mask, [0, 0, 0], np.eye(3), np.zeros(3))


def test_is_closed_counts_edges_by_position_like_the_viewer():
    a = trimesh.creation.box()
    assert _is_closed(a.vertices, a.faces)
    # Duas caixas encostadas por uma aresta, cada uma com os seus vértices:
    # fechadas pelos índices, mas a aresta comum tem 4 triângulos na posição.
    b = trimesh.creation.box()
    b.apply_translation([1.0, 1.0, 0.0])
    both = trimesh.util.concatenate([a, b])
    assert both.is_watertight
    assert not _is_closed(both.vertices, both.faces)


def test_decimation_keeps_a_closed_mesh_closed():
    mesh = _thin_blob_mesh()
    assert _is_closed(mesh.vertices, mesh.faces)
    target = len(mesh.faces) // 2
    out, input_tris, decimated = _decimate(mesh, target)
    assert decimated and input_tris == len(mesh.faces)
    assert len(out.faces) <= target
    assert _is_closed(out.vertices, out.faces)
    assert abs(out.volume - mesh.volume) / mesh.volume < 0.01


def test_decimation_falls_back_when_simplify_opens_the_mesh(monkeypatch):
    """Se o colapso deixa a malha aberta mesmo sem as faces dobradas, o plano B
    (Manifold.simplify) decima sem abrir."""
    import processor

    real = processor.fast_simplification.simplify

    def opening(points, faces, **kw):
        p, f = real(points, faces, **kw)
        return p, f[3:]  # três triângulos a menos: um furo

    monkeypatch.setattr(processor.fast_simplification, "simplify", opening)
    mesh = trimesh.creation.icosphere(subdivisions=6, radius=30.0)
    target = len(mesh.faces) // 4
    out, _, decimated = _decimate(mesh, target)
    assert decimated
    assert _is_closed(out.vertices, out.faces)
    assert len(out.faces) <= target
    assert abs(out.volume - mesh.volume) / mesh.volume < 0.01


def test_decimated_segmentation_is_closed_in_the_glb():
    """De ponta a ponta: o que o visualizador lê do GLB (float32) fecha."""
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(np.random.default_rng(2).random((40, 40, 40)), 2)
    data = (field > field.min() + 0.5 * (field.max() - field.min())).astype(np.uint8)
    data[:2] = data[-2:] = 0
    blob = _nrrd_bytes(data, directions=np.eye(3) * 0.5, origin=np.zeros(3))
    glb, stats = process_stls([], target_triangles_per_mesh=22_000,
                              segmentations=[("pontes.nrrd", blob)])
    (m,) = stats.meshes
    assert m.decimated
    (mesh,) = _glb_nodes(glb).values()
    assert _is_closed(mesh.vertices, mesh.faces)


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
