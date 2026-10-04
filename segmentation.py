"""Segmentação em NRRD (labelmap) → uma malha fechada por estrutura. Módulo puro.

bytes in → malhas out, sem I/O, sem env, sem rede — como `processor.py` e
`exam.py`. Quem chama é `processor.process_stls`, que trata cada estrutura que
sai daqui exatamente como um STL: decimação, divisões, rotação para o glTF, cor.

Dois formatos de entrada:

* **`.seg.nrrd` do 3D Slicer.** O cabeçalho traz, por segmento,
  `SegmentN_Name`, `SegmentN_Color` (sRGB 0..1), `SegmentN_LabelValue` e
  `SegmentN_Layer`. Com segmentos que se sobrepõem o Slicer grava um volume 4D:
  o primeiro eixo (`kinds: list …`) é a camada. Arquivos de antes do Slicer
  4.11 não têm `LabelValue`/`Layer`: cada segmento é a sua camada, valor 1.
* **Labelmap simples** (ITK-SNAP, nnU-Net, TotalSegmentator, uma máscara
  binária): um volume 3D de inteiros, 0 = fundo, cada outro valor = uma
  estrutura. Sem nomes no arquivo: uma estrutura só leva o nome do arquivo;
  várias viram "Segmento <valor>".

Geometria: a mesma do exame (`exam._read_nrrd`) — `space directions` e
`space origin`, convertidos para LPS. Por isso a malha cai exatamente em cima
dos cortes quando o volume de imagem vem junto como exame: os dois estão no
mesmo espaço do paciente, e o `processor` aplica a mesma rotação de sempre.

Superfície: marching cubes (scikit-image) na máscara desfocada de cada
estrutura, com uma borda de zeros em volta para a malha sempre fechar (o volume
do visualizador precisa de malha fechada). Ao contrário do STL — em que o
processor não suaviza nada, para não arredondar anatomia —, aqui o degrau que a
suavização tira é artefato do voxel, não anatomia: sem ela cada estrutura vira
uma escada de cubinhos. É o papel do "Smoothing factor" que o 3D Slicer aplica
ao gerar a superfície. O nível da superfície é ajustado para o volume da malha
bater com o dos voxels (ver as constantes abaixo); Taubin foi medido e
descartado: encolhia 25 % uma esfera de 5 voxels.

Erro de entrada → `ValueError` em pt-BR, a mesma regra dos outros módulos.
"""
from __future__ import annotations

import io
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import trimesh

from exam import MAX_NRRD_INPUT_BYTES, _SPACES, _to_lps_sign

# Mais que isto não é uma segmentação: é um exame de imagem mandado no lugar
# errado. O TotalSegmentator completo tem 117 estruturas.
MAX_SEGMENTS = 128

# Suavização: desfoque gaussiano da máscara (σ em voxels, por eixo — cada eixo
# perde o degrau do seu próprio voxel) antes do marching cubes. O desfoque
# encolhe estrutura pequena ou fina (uma esfera de 3 voxels perde 27 % a 0,5),
# então o nível da superfície é ajustado até o volume da malha bater com o dos
# voxels (± LEVEL_TOLERANCE) — o mesmo número que a "Segment Statistics" do
# Slicer mostra, e o que a medida de volume do visualizador vai dar. Uma
# estrutura de um voxel de espessura sobrevive (o nível desce) em vez de sumir.
# Estrutura fina (costela de 1–2 voxels) perde mais da metade no desfoque de
# 0,8: abaixo de THIN_FRACTION do volume no nível 0,5, tenta o σ seguinte.
SMOOTH_SIGMAS_VOXELS = (0.8, 0.5)
THIN_FRACTION = 0.6
SMOOTH_PAD = 3
LEVEL_STEPS = 4
LEVEL_TOLERANCE = 0.005

_SEGMENT_KEY = re.compile(r"^Segment(\d+)_(\w+)$")
# "feto-label", "Segmentation", "rim_mask"… → a palavra que só diz "isto é uma
# segmentação" sai do nome da estrutura.
_SEG_WORD_TAIL = re.compile(r"[\s._-]*(seg|segmentation|segmentacao|label|labels|labelmap|mask)$", re.I)


@dataclass
class Segment:
    name: str
    mesh: trimesh.Trimesh   # em LPS, mm
    color: str | None       # "#RRGGBB" sRGB, como o Slicer mostra; None sem cor no arquivo
    voxels: int


def is_segmentation_header(header: dict, filename: str = "") -> bool:
    """O cabeçalho (ou o nome) diz que é segmentação? Sem olhar os voxels."""
    if any(_SEGMENT_KEY.match(str(k)) for k in header):
        return True
    return filename.lower().endswith(".seg.nrrd")


def segments_from_nrrd(data: bytes, filename: str = "") -> list[Segment]:
    """Lê um NRRD de segmentação e devolve uma malha por estrutura (LPS, mm)."""
    import nrrd  # import tardio, como em exam.py

    fh = io.BytesIO(data)
    end = data.find(b"\n\n", 0, 4 * 1024 * 1024)
    raw_header = data[: end if end > 0 else 0]
    try:
        header = nrrd.read_header(fh)
    except Exception as e:
        raise ValueError(f"Arquivo NRRD inválido ou corrompido ({filename}): {e}")
    if "data file" in header or "datafile" in header:
        raise ValueError(
            "Este NRRD tem os dados num arquivo separado (.nhdr + .raw). "
            "Exporte como um único arquivo .nrrd."
        )

    dim = int(header.get("dimension", 0))
    kinds = [str(k).lower() for k in header.get("kinds", [])]
    layered = dim == 4 and kinds[:1] and kinds[0] in ("list", "vector", "point", "covariant-vector")
    if dim != 3 and not layered:
        raise ValueError(
            f"A segmentação {filename} precisa ser um volume 3D (este NRRD tem dimensão {dim})."
        )

    sizes = [int(s) for s in header["sizes"]]
    try:
        itemsize = np.dtype(nrrd.reader._determine_datatype(header)).itemsize
    except Exception:
        itemsize = 8
    if int(np.prod(sizes)) * itemsize > MAX_NRRD_INPUT_BYTES:
        raise ValueError(
            f"A segmentação {filename} é grande demais para processar "
            f"(mais de {MAX_NRRD_INPUT_BYTES // (1024 * 1024)} MB descompactada)."
        )
    try:
        arr = nrrd.read_data(header, fh, index_order="C")
    except Exception as e:
        raise ValueError(f"Não foi possível ler os dados da segmentação {filename}: {e}")
    del data, fh

    # Ordem C: o eixo que varia mais rápido (o primeiro de `sizes`) fica por
    # último. 3D → (k, j, i); 4D com a camada primeiro → (k, j, i, camada).
    if layered:
        layers = [arr[..., n] for n in range(arr.shape[-1])]
        spatial = sizes[1:]
    else:
        layers = [arr]
        spatial = sizes
    if layers[0].shape != (spatial[2], spatial[1], spatial[0]):
        raise ValueError(f"Segmentação {filename} inconsistente: dados e cabeçalho não batem.")

    directions, origin = _geometry(header, layered)
    layers = [_as_labels(layer, filename) for layer in layers]

    meta = _segment_meta(header, raw_header)
    if meta:
        wanted = _slicer_segments(meta, layered, len(layers))
    else:
        wanted = _plain_segments(layers, filename)
    if not wanted:
        raise ValueError(f"A segmentação {filename} está vazia: nenhum voxel marcado.")
    if len(wanted) > MAX_SEGMENTS:
        raise ValueError(
            f"O arquivo {filename} tem {len(wanted)} valores diferentes — parece um exame de "
            f"imagem, não uma segmentação (o limite é {MAX_SEGMENTS} estruturas). "
            "Se for o exame, envie-o como exame."
        )

    out: list[Segment] = []
    used: set[str] = set()
    boxes = [_label_boxes(layer) for layer in layers]
    for name, layer_i, value, color in wanted:
        box = boxes[layer_i].get(value)
        if box is None:
            continue  # segmento declarado no cabeçalho, mas sem nenhum voxel
        mask = layers[layer_i][box] == value
        voxels = int(mask.sum())
        mesh = _surface(mask, [s.start for s in box], directions, origin)
        if mesh is None:
            continue
        name = _unique(name, used)
        out.append(Segment(name=name, mesh=mesh, color=color, voxels=voxels))
    if not out:
        raise ValueError(f"A segmentação {filename} está vazia: nenhum voxel marcado.")
    return out


# ---- Cabeçalho ----------------------------------------------------------------


def _geometry(header: dict, layered: bool) -> tuple[np.ndarray, np.ndarray]:
    dirs = header.get("space directions")
    if dirs is not None:
        dirs = np.asarray(dirs, dtype=float)
        if layered and dirs.shape == (4, 3):
            dirs = dirs[1:]  # a linha da camada é "none" (NaN)
    else:
        spacings = header.get("spacings")
        if spacings is None:
            spacings = [1.0] * (4 if layered else 3)
        sp = [float(s) for s in spacings][-3:]
        dirs = np.diag(sp)
    if dirs.shape != (3, 3) or not np.all(np.isfinite(dirs)):
        raise ValueError("A segmentação precisa ter geometria 3D (space directions) nos três eixos.")
    origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)

    space = str(header.get("space", "left-posterior-superior")).strip().lower()
    letters = _SPACES.get(space)
    if letters is None:
        raise ValueError(
            f"Espaço de coordenadas da segmentação não suportado: '{space}'. "
            "Exporte em LPS ou RAS (padrão do 3D Slicer)."
        )
    sign = _to_lps_sign(letters)
    return dirs * sign, origin * sign


def _as_labels(layer: np.ndarray, filename: str) -> np.ndarray:
    """Inteiros ≥ 0. Máscara salva como float (0.0/1.0) também serve."""
    if not np.issubdtype(layer.dtype, np.integer):
        if not np.all(np.isfinite(layer)) or not np.all(layer == np.round(layer)):
            raise ValueError(
                f"O arquivo {filename} tem valores não inteiros — parece um exame de imagem, "
                "não uma segmentação. Se for o exame, envie-o como exame."
            )
        layer = layer.astype(np.int64)
    if layer.size and layer.min() < 0:
        raise ValueError(
            f"O arquivo {filename} tem valores negativos — parece um exame de imagem "
            "(uma TC, por exemplo), não uma segmentação. Se for o exame, envie-o como exame."
        )
    return layer


def _segment_meta(header: dict, raw_header: bytes) -> dict[int, dict[str, str]]:
    """{n: {"Name": …, "Color": …, …}} dos campos `SegmentN_*` do cabeçalho.

    Lidos dos bytes crus, e não do `header` do pynrrd: o pynrrd decodifica o
    cabeçalho como ASCII descartando o resto, e o Slicer grava os nomes em
    UTF-8 — "Útero" viraria "tero".
    """
    try:
        text = raw_header.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_header.decode("latin-1")
    pairs = (line.split(":=", 1) for line in text.splitlines() if ":=" in line)
    meta: dict[int, dict[str, str]] = {}
    for k, v in list(pairs) or header.items():
        m = _SEGMENT_KEY.match(str(k).strip())
        if m:
            meta.setdefault(int(m.group(1)), {})[m.group(2)] = str(v).strip()
    return meta


def _slicer_segments(meta, layered: bool, n_layers: int):
    """[(nome, camada, valor, cor)] na ordem dos segmentos do Slicer."""
    out = []
    for n in sorted(meta):
        m = meta[n]
        try:
            value = int(m.get("LabelValue", "1"))
            # Antes do Slicer 4.11: sem Layer, um segmento por camada.
            layer = int(m["Layer"]) if "Layer" in m else (n if layered else 0)
        except ValueError:
            raise ValueError(f"Segmento {n} com LabelValue/Layer inválido no cabeçalho.")
        if not 0 <= layer < n_layers:
            raise ValueError(f"Segmento {n} aponta para uma camada que não existe.")
        name = _clean_name(m.get("Name", "")) or f"Segmento {n + 1}"
        out.append((name, layer, value, _slicer_color(m.get("Color"))))
    return out


def _plain_segments(layers: list[np.ndarray], filename: str):
    values = [(i, v) for i, layer in enumerate(layers) for v in _nonzero_values(layer)]
    if len(values) > MAX_SEGMENTS:
        return values  # quem chama recusa pelo tamanho; não vale a pena nomear
    if len(values) == 1:
        stem = _file_stem(filename)
        return [(stem or "Segmento 1", values[0][0], values[0][1], None)]
    multi_layer = len(layers) > 1
    return [
        (f"Segmento {i + 1}.{v}" if multi_layer else f"Segmento {v}", i, v, None)
        for i, v in values
    ]


def _nonzero_values(layer: np.ndarray) -> list[int]:
    if layer.dtype.itemsize <= 2 and layer.size:
        counts = np.bincount(layer.ravel().astype(np.int64, copy=False))
        return [int(v) for v in np.nonzero(counts)[0] if v != 0]
    return [int(v) for v in np.unique(layer) if v != 0]


def _slicer_color(text: str | None) -> str | None:
    if not text:
        return None
    try:
        rgb = [float(c) for c in text.split()[:3]]
    except ValueError:
        return None
    if len(rgb) != 3 or not all(0 <= c <= 1 for c in rgb):
        return None
    return "#%02X%02X%02X" % tuple(round(c * 255) for c in rgb)


def _clean_name(text: str) -> str:
    """Como o nome de um STL (main.clean_mesh_names): sem acento, sem controle."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in text)
    return re.sub(r"\s+", " ", text).strip(" _")[:64]


def _file_stem(filename: str) -> str:
    base = os.path.basename(filename or "")
    base = re.sub(r"(\.seg)?\.nrrd$", "", base, flags=re.I)
    base = _SEG_WORD_TAIL.sub("", base)
    return _clean_name(base)


def _unique(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate in used:
        candidate = f"{name} {n}"
        n += 1
    used.add(candidate)
    return candidate


# ---- Superfície ----------------------------------------------------------------


def _label_boxes(layer: np.ndarray) -> dict[int, tuple[slice, slice, slice]]:
    """Caixa de cada valor numa passada só (scipy.ndimage.find_objects)."""
    from scipy import ndimage

    if not layer.size or layer.max() == 0:
        return {}
    if layer.max() > 65535:
        # find_objects aloca uma entrada por valor até o máximo.
        return {
            v: ndimage.find_objects((layer == v).astype(np.uint8))[0]
            for v in _nonzero_values(layer)
        }
    objs = ndimage.find_objects(layer.astype(np.int32, copy=False))
    return {i + 1: box for i, box in enumerate(objs) if box is not None}


def _surface(
    mask: np.ndarray, start_kji: list[int], directions: np.ndarray, origin: np.ndarray
) -> trimesh.Trimesh | None:
    """Máscara (k, j, i) recortada → malha fechada e suave em LPS (mm).

    Desfoque gaussiano da máscara e marching cubes no nível em que o volume da
    malha bate com o dos voxels. Tudo em unidades de índice (1 voxel = 1), e só
    no fim a malha vai para mm.
    """
    from scipy.ndimage import gaussian_filter

    voxels = float(mask.sum())
    if voxels == 0:
        return None
    # Borda de zeros maior que o alcance do desfoque: a estrutura que encosta
    # na borda do volume também fecha.
    padded = np.pad(mask, SMOOTH_PAD).astype(np.float32)
    mesh = None
    for n, sigma in enumerate(SMOOTH_SIGMAS_VOXELS):
        field = gaussian_filter(padded, sigma)
        peak = float(field.max())
        level = min(0.5, 0.5 * peak)
        mesh = _marching(field, level)
        if mesh is None:
            continue
        if abs(mesh.volume) < THIN_FRACTION * voxels and n + 1 < len(SMOOTH_SIGMAS_VOXELS):
            continue  # o desfoque comeu a estrutura: fina demais para este σ
        for _ in range(LEVEL_STEPS - 1):
            volume = abs(mesh.volume)
            if abs(volume - voxels) <= LEVEL_TOLERANCE * voxels or mesh.area == 0:
                break
            # Passo de Newton: falta (sobra) uma casca de espessura ΔV/A
            # voxels; na borda desfocada o campo cai 1/(σ√2π) por voxel.
            gap = (voxels - volume) / mesh.area
            level = level - gap / (sigma * np.sqrt(2 * np.pi))
            level = float(np.clip(level, 0.05 * peak, min(0.7, 0.9 * peak)))
            mesh = _marching(field, level) or mesh
        break
    if mesh is None or len(mesh.faces) == 0:
        return None

    ijk = (mesh.vertices + np.asarray(start_kji, dtype=float) - SMOOTH_PAD)[:, ::-1]
    world = origin + ijk @ directions  # linhas de `directions` = vetores de i, j, k
    mesh = trimesh.Trimesh(vertices=world, faces=mesh.faces, process=True)
    # A troca (k, j, i) → (i, j, k) e uma geometria com determinante negativo
    # viram a malha do avesso; o sinal do volume diz qual é o caso.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _marching(field: np.ndarray, level: float) -> trimesh.Trimesh | None:
    from skimage.measure import marching_cubes

    try:
        verts_kji, faces, _, _ = marching_cubes(field, level=level, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    if len(faces) == 0:
        return None
    return trimesh.Trimesh(vertices=verts_kji, faces=faces, process=True)
