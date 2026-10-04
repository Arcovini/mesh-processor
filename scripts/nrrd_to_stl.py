"""Gera STLs de validação a partir de um exame. Ferramenta de desenvolvimento.

Dois usos:

1) Um STL da "pele" de um exame real, por marching cubes, para conferir se os
   cortes do visualizador caem em cima do 3D (o DICOM e o NRRD de exemplo são
   da mesma paciente, mas não temos a segmentação dela):

       .venv/bin/python scripts/nrrd_to_stl.py ~/Downloads/11.nrrd --out feto.stl

2) O par sintético usado nos testes do visualizador: uma esfera de raio 20 mm
   em (10, −30, 50) LPS dentro de um volume com eixos oblíquos, origem fora de
   zero e tamanhos diferentes por eixo (pega transposição e deslocamento):

       .venv/bin/python scripts/nrrd_to_stl.py --sphere <pasta>

   Grava exam-sphere.nrrd (já no formato canônico do exam.py),
   exam-sphere.stl e exam-sphere.glb (passado pelo processor.py, igual a um
   upload), e exam-sphere-fase2.nrrd: a mesma esfera no mesmo espaço do
   paciente, como outra série do exame (cortes de 4 mm, outra origem e valor
   600) — para o teste de troca de série abrir no mesmo ponto em mm —, e
   exam-ct.nrrd: uma TC mínima em Hounsfield (ar −1000, corpo 40, vaso 300,
   osso 800) para os presets de janela do visualizador.

3) A segmentação da mesma esfera, para o caminho "só NRRD" (segmentation.py):

       .venv/bin/python scripts/nrrd_to_stl.py --seg <pasta>

   Grava exam-sphere.seg.nrrd (labelmap no espaço de exam-sphere.nrrd, com o
   cabeçalho de segmentos do 3D Slicer: "Esfera" = 1 e "Núcleo" = 2, uma
   esfera de 8 mm no centro, nome em UTF-8 como o Slicer grava),
   exam-sphere-seg.glb (essa segmentação passada pelo processor.py, igual a um
   upload) e esfera-rotulos.nrrd (o mesmo labelmap sem cabeçalho nem nome que
   diga "segmentação": a página tem de reconhecer pelos valores).

O STL sai com o cabeçalho `3D Slicer output. SPACE=LPS`, como o Slicer grava.
Precisa de requirements-dev.txt (scikit-image).
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exam import _read_nrrd, normalize_exam  # noqa: E402
from processor import process_stls  # noqa: E402

SPHERE_CENTER_LPS = np.array([10.0, -30.0, 50.0])
SPHERE_RADIUS_MM = 20.0


def _stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    data = bytearray(mesh.export(file_type="stl"))
    header = b"3D Slicer output. SPACE=LPS"
    data[:80] = header.ljust(80, b" ")
    return bytes(data)


def surface_from_volume(
    data: np.ndarray, directions: np.ndarray, origin: np.ndarray, level: float, step: int
) -> trimesh.Trimesh:
    """Marching cubes em (k, j, i) → vértices em LPS mm, maior componente."""
    from skimage.measure import marching_cubes

    verts_kji, faces, _, _ = marching_cubes(data.astype(np.float32), level=level, step_size=step)
    ijk = verts_kji[:, ::-1]  # (k, j, i) → (i, j, k)
    world = origin + ijk @ directions  # linhas de `directions` = vetores de i, j, k
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    mesh.fix_normals()
    return mesh


def from_nrrd(path: str, out: str, percentile: float, level: float | None, step: int) -> None:
    with open(path, "rb") as f:
        vol, _ = _read_nrrd(f.read())
    if level is None:
        level = float(np.percentile(vol.data, percentile))
    mesh = surface_from_volume(vol.data, vol.directions, vol.origin, level, step)
    with open(out, "wb") as f:
        f.write(_stl_bytes(mesh))
    print(f"{out}: {len(mesh.faces)} triângulos, limiar {level:.1f}, "
          f"fechada={mesh.is_watertight}, centro LPS {np.round(mesh.centroid, 1)}")


def sphere(out_dir: str) -> None:
    import nrrd

    os.makedirs(out_dir, exist_ok=True)
    # Tamanhos e espaçamentos diferentes por eixo, e eixos girados 12° em torno
    # de x (como a RM de exemplo): um volume transposto ou um eixo trocado
    # desloca a esfera e o teste pega.
    ni, nj, nk = 64, 56, 48
    sp = np.array([1.2, 1.5, 2.0])
    a = np.deg2rad(12)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    directions = (rot @ np.diag(sp)).T  # linhas = vetores de i, j, k
    center_idx = np.array([(ni - 1) / 2, (nj - 1) / 2, (nk - 1) / 2]) + np.array([3, -2, 1])
    origin = SPHERE_CENTER_LPS - center_idx @ directions

    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(np.float64)
    world = origin + ijk @ directions
    dist = np.linalg.norm(world - SPHERE_CENTER_LPS, axis=1)
    # Borda suave de ~1 mm: o limiar 500 cai exatamente no raio.
    vals = np.clip((SPHERE_RADIUS_MM - dist) / 1.0 + 0.5, 0, 1) * 1000
    data = np.rint(vals).astype(np.int16).reshape(nk, nj, ni)

    raw = io.BytesIO()
    nrrd.write(raw, data, {
        "space": "left-posterior-superior",
        "space directions": directions,
        "space origin": origin,
        "kinds": ["domain"] * 3,
        "encoding": "raw",
    }, index_order="C")
    nrrd_bytes, stats = normalize_exam([("exam-sphere.nrrd", raw.getvalue())])
    with open(os.path.join(out_dir, "exam-sphere.nrrd"), "wb") as f:
        f.write(nrrd_bytes)

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=SPHERE_RADIUS_MM)
    mesh.apply_translation(SPHERE_CENTER_LPS)
    stl = _stl_bytes(mesh)
    with open(os.path.join(out_dir, "exam-sphere.stl"), "wb") as f:
        f.write(stl)
    glb, _ = process_stls([("Esfera", stl)], target_triangles_per_mesh=300_000)
    with open(os.path.join(out_dir, "exam-sphere.glb"), "wb") as f:
        f.write(glb)
    print(f"{out_dir}: exam-sphere.nrrd ({len(nrrd_bytes)} B, {stats.shape}), "
          f"exam-sphere.stl, exam-sphere.glb ({len(glb)} B); "
          f"centro LPS {SPHERE_CENTER_LPS}, raio {SPHERE_RADIUS_MM} mm")
    sphere_series2(out_dir)
    ct_fixture(out_dir)


def ct_fixture(out_dir: str) -> None:
    """TC mínima em Hounsfield, 40×40×20 (1 × 1 × 2 mm): ar fora de um
    cilindro de corpo (40 HU), com um vaso (300 HU) e um osso (800 HU)."""
    import nrrd

    nk, nj, ni = 20, 40, 40
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    r = np.hypot(i - 19.5, j - 19.5)
    data = np.full((nk, nj, ni), -1000, dtype=np.int16)
    data[r < 16] = 40
    data[np.hypot(i - 12, j - 20) < 3] = 300
    data[np.hypot(i - 26, j - 26) < 4] = 800
    raw = io.BytesIO()
    nrrd.write(raw, data, {
        "space": "left-posterior-superior",
        "space directions": np.diag([1.0, 1.0, 2.0]),
        "space origin": np.array([-20.0, -20.0, 0.0]),
        "kinds": ["domain"] * 3,
        "encoding": "raw",
    }, index_order="C")
    out, stats = normalize_exam([("exam-ct.nrrd", raw.getvalue())])
    with open(os.path.join(out_dir, "exam-ct.nrrd"), "wb") as f:
        f.write(out)
    print(f"{out_dir}: exam-ct.nrrd ({len(out)} B, {stats.shape})")


def sphere_series2(out_dir: str) -> None:
    """Segunda série da esfera: mesmos eixos, cortes de 4 mm (24 em vez de 48),
    origem deslocada meio corte e esfera com valor 600. Mesmo espaço LPS."""
    import nrrd

    ni, nj, nk = 64, 56, 24
    sp = np.array([1.2, 1.5, 4.0])
    a = np.deg2rad(12)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    directions = (rot @ np.diag(sp)).T
    center_idx = np.array([(ni - 1) / 2, (nj - 1) / 2, (nk - 1) / 2]) + np.array([3, -2, 0.25])
    origin = SPHERE_CENTER_LPS - center_idx @ directions

    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(np.float64)
    world = origin + ijk @ directions
    dist = np.linalg.norm(world - SPHERE_CENTER_LPS, axis=1)
    vals = np.clip((SPHERE_RADIUS_MM - dist) / 1.0 + 0.5, 0, 1) * 600
    data = np.rint(vals).astype(np.int16).reshape(nk, nj, ni)

    raw = io.BytesIO()
    nrrd.write(raw, data, {
        "space": "left-posterior-superior",
        "space directions": directions,
        "space origin": origin,
        "kinds": ["domain"] * 3,
        "encoding": "raw",
    }, index_order="C")
    out, stats = normalize_exam([("exam-sphere-fase2.nrrd", raw.getvalue())])
    with open(os.path.join(out_dir, "exam-sphere-fase2.nrrd"), "wb") as f:
        f.write(out)
    print(f"{out_dir}: exam-sphere-fase2.nrrd ({len(out)} B, {stats.shape})")


def seg_fixture(out_dir: str) -> None:
    import nrrd

    os.makedirs(out_dir, exist_ok=True)
    ni, nj, nk = 64, 56, 48  # a geometria de exam-sphere.nrrd
    sp = np.array([1.2, 1.5, 2.0])
    a = np.deg2rad(12)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    directions = (rot @ np.diag(sp)).T
    center_idx = np.array([(ni - 1) / 2, (nj - 1) / 2, (nk - 1) / 2]) + np.array([3, -2, 1])
    origin = SPHERE_CENTER_LPS - center_idx @ directions

    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ijk = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(np.float64)
    dist = np.linalg.norm(origin + ijk @ directions - SPHERE_CENTER_LPS, axis=1).reshape(nk, nj, ni)
    labels = np.zeros((nk, nj, ni), dtype=np.uint8)
    labels[dist <= SPHERE_RADIUS_MM] = 1
    labels[dist <= 8.0] = 2

    def write(header_extra: dict) -> bytes:
        raw = io.BytesIO()
        nrrd.write(raw, labels, {
            "space": "left-posterior-superior",
            "space directions": directions,
            "space origin": origin,
            "kinds": ["domain"] * 3,
            "encoding": "gzip",
            **header_extra,
        }, index_order="C")
        return raw.getvalue()

    seg = write({
        "Segment0_ID": "Segment_1", "Segment0_Name": "Esfera", "Segment0_Color": "0.85 0.55 0.45",
        "Segment0_LabelValue": "1", "Segment0_Layer": "0",
        "Segment1_ID": "Segment_2", "Segment1_Name": "NUCLEO", "Segment1_Color": "0.3 0.5 0.9",
        "Segment1_LabelValue": "2", "Segment1_Layer": "0",
    }).replace(b":=NUCLEO", ":=Núcleo".encode())  # o pynrrd só escreve ASCII
    with open(os.path.join(out_dir, "exam-sphere.seg.nrrd"), "wb") as f:
        f.write(seg)
    with open(os.path.join(out_dir, "esfera-rotulos.nrrd"), "wb") as f:
        f.write(write({}))
    glb, stats = process_stls([], segmentations=[("exam-sphere.seg.nrrd", seg)])
    with open(os.path.join(out_dir, "exam-sphere-seg.glb"), "wb") as f:
        f.write(glb)
    print(f"{out_dir}: exam-sphere.seg.nrrd ({len(seg)} B), esfera-rotulos.nrrd, "
          f"exam-sphere-seg.glb ({len(glb)} B): {[(m.name, m.output_triangles) for m in stats.meshes]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nrrd", nargs="?", help="exame .nrrd de entrada")
    p.add_argument("--out", help="STL de saída")
    p.add_argument("--percentile", type=float, default=60.0,
                   help="limiar como percentil dos voxels (padrão 60)")
    p.add_argument("--level", type=float, help="limiar absoluto (ignora --percentile)")
    p.add_argument("--step", type=int, default=2, help="passo do marching cubes (padrão 2)")
    p.add_argument("--sphere", metavar="PASTA", help="gera o par sintético da esfera")
    p.add_argument("--seg", metavar="PASTA", help="gera a segmentação NRRD da esfera")
    args = p.parse_args()
    if args.sphere:
        sphere(args.sphere)
    elif args.seg:
        seg_fixture(args.seg)
    elif args.nrrd and args.out:
        from_nrrd(args.nrrd, args.out, args.percentile, args.level, args.step)
    else:
        p.error("informe um .nrrd e --out, ou --sphere PASTA")


if __name__ == "__main__":
    main()
