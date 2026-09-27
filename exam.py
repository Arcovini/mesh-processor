"""Exame de imagem (série DICOM ou NRRD) → NRRD canônico. Módulo puro.

bytes in → bytes out + stats. Sem I/O, sem env, sem rede — como `processor.py`.
`main.py` entrega os arquivos um de cada vez (gerador), e daqui sai um único
NRRD — uma série — que vai para o R2 em `cases/{uid}.exam-{n}.nrrd` ao lado do
GLB. Um caso pode ter até 4 séries; cada uma chega num request próprio.

O formato canônico é pensado para o visualizador, não para arquivamento:

    NRRD, type int16 | uint16 | float, dimension 3,
    space: left-posterior-superior (sempre), sizes, space directions (mm),
    kinds: domain domain domain, endian: little, encoding: gzip, space origin.

Nada além disso. É assim que a anonimização acontece *por construção*: o
cabeçalho DICOM inteiro (nome, ID, datas, instituição…) fica para trás, e só a
geometria e os pixels atravessam. A única exceção, por decisão do usuário
(2026-09-26), é o nome da série (SeriesDescription): vira o `label` das stats e
aparece no seletor de série do visualizador, só com a limpeza de `clean_label`.
O que continua existindo é o conteúdo da
imagem — um rosto reconstruível numa TC de crânio, por exemplo — e isso só se
resolve com acesso controlado (Sprint 4), não aqui.

Convenção de eixos, usada em todo o módulo e no header de saída:
  i = colunas da imagem (varia mais rápido), j = linhas, k = fatias.
  O array numpy fica em ordem C com shape (nk, nj, ni) — índice [k, j, i] —, e
  `space directions` lista os vetores de i, j e k nessa ordem (a ordem de
  `sizes`). Trocar isso transpõe o volume sem nenhum erro aparente; o teste de
  transposição em `test_exam.py` existe por isso.

Erro de entrada → `ValueError` com mensagem em pt-BR (vai direto para a tela
do clínico), a mesma regra de `processor.py`.
"""
from __future__ import annotations

import io
import os
import re
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

# ---- Limites (valores discutidos com o advisor; ver CLAUDE.md) --------------

# Teto de voxels do volume que vai para o navegador. 16 M voxels int16 ≈ 32 MB
# de memória no celular (≈ 12–18 MB gzip no R2). Acima disso o volume é
# reamostrado com um "piso de espaçamento" (_plan_resample): acha o menor
# espaçamento s que cabe no teto e leva a s só os eixos mais finos que ele, por
# média de área. Eixos empatados (o plano da imagem, quase sempre) encolhem
# juntos — o pixel continua quadrado —, e os eixos mais grossos não perdem nada.
MAX_VOXELS = 16_000_000

# NRRD de entrada: o pynrrd descomprime o volume inteiro na memória antes de
# qualquer redução. Checado pelo cabeçalho, antes de ler os dados — um .nrrd
# gzip de 200 MB pode descomprimir em mais de 1 GB.
MAX_NRRD_INPUT_BYTES = 600 * 1024 * 1024

# Zip da série: teto da soma dos membros descomprimidos (os datasets ficam na
# memória até a série ser escolhida). A série DICOM em si é decodificada fatia
# por fatia, já reduzida ao tamanho final — ver _stack_series.
MAX_UNZIPPED_BYTES = 600 * 1024 * 1024

# Séries com menos imagens que isto são localizer/scout/relatório de dose e são
# descartadas antes de exigir que sobre uma série só.
MIN_SLICES_PER_SERIES = 8

# Tolerâncias geométricas. O PACS grava IOP com ~6 casas decimais.
IOP_TOL = 1e-4
SPACING_TOL_MM = 0.01
SPACING_TOL_REL = 0.01

GZIP_LEVEL = 4  # nível 9 dobra o tempo por ~3 % de ganho

CANONICAL_SPACE = "left-posterior-superior"

# Rótulo da série no visualizador: texto livre digitado no tomógrafo.
MAX_LABEL_CHARS = 64

# ---- Tipos -------------------------------------------------------------------


@dataclass(frozen=True)
class ExamStats:
    source: str                     # "dicom" | "nrrd"
    shape: tuple[int, int, int]     # (ni, nj, nk) — a ordem de `sizes`
    spacing: tuple[float, float, float]  # mm, na mesma ordem
    dtype: str                      # "int16" | "uint16" | "float32"
    downsample: tuple[float, float, float]  # fator por eixo (1 = intacto; não precisa ser inteiro)
    input_files: int                # arquivos recebidos (membros do zip contam)
    ignored_files: int              # não-DICOM, sem pixels, sem geometria, séries descartadas
    modality: str | None            # só para log/resposta; nunca vai no NRRD
    nrrd_bytes: int
    label: str | None = None        # SeriesDescription (ou nome do .nrrd) limpa; None se vazia


@dataclass
class _Volume:
    data: np.ndarray        # (nk, nj, ni), ordem C
    directions: np.ndarray  # 3×3, linhas = vetores de i, j, k em LPS (mm)
    origin: np.ndarray      # centro do voxel [0,0,0] em LPS (mm)


# ---- Entrada: arquivos soltos ou zip ------------------------------------------


def _is_nrrd(name: str, data: bytes) -> bool:
    return data[:7] == b"NRRD000" or _ext(name) in (".nrrd", ".nhdr")


def _ext(name: str) -> str:
    return os.path.splitext(name.lower())[1]


_ZIP_READ_ERROR = (
    "Não foi possível abrir o .zip do exame (protegido por senha, compactação não "
    "suportada ou arquivo incompleto). Compacte a pasta de novo, sem senha, e envie."
)


def _expand(items: Iterable[tuple[str, bytes]]) -> Iterator[tuple[str, bytes]]:
    """Abre zips em membros; o resto passa direto. Um item de cada vez."""
    for name, data in items:
        if _ext(name) == ".zip" or data[:4] == b"PK\x03\x04":
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                raise ValueError(f"Arquivo .zip inválido ou corrompido: {name}.")
            with zf:
                total = 0
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    base = os.path.basename(info.filename)
                    # Lixo de sistema operacional dentro do zip.
                    if base.startswith(".") or info.filename.startswith("__MACOSX/"):
                        continue
                    total += info.file_size
                    if total > MAX_UNZIPPED_BYTES:
                        raise ValueError(
                            "O .zip do exame descompactado passa de "
                            f"{MAX_UNZIPPED_BYTES // (1024 * 1024)} MB. Envie só a série "
                            "usada na segmentação."
                        )
                    try:
                        member = zf.read(info)
                    except (RuntimeError, NotImplementedError, zipfile.BadZipFile,
                            zlib.error, EOFError, OSError):
                        raise ValueError(_ZIP_READ_ERROR)
                    yield info.filename, member
        else:
            yield name, data


# ---- Ponto de entrada ---------------------------------------------------------


def normalize_exam(items: Iterable[tuple[str, bytes]]) -> tuple[bytes, ExamStats]:
    """Série DICOM (arquivos soltos ou .zip) ou um .nrrd → NRRD canônico gzip.

    `items` pode ser um gerador: cada arquivo é lido, reduzido ao que importa
    (posição + pixels) e descartado antes do próximo, então o pico de memória
    fica perto do volume em si, não da soma dos arquivos.
    """
    it = _expand(items)
    first = next(it, None)
    if first is None:
        raise ValueError("Nenhum arquivo de exame recebido.")

    if _is_nrrd(*first):
        extra = next(it, None)
        if extra is not None:
            raise ValueError(
                "Envie o exame como um único arquivo .nrrd ou como uma série DICOM, "
                "não os dois juntos."
            )
        vol, dtype = _read_nrrd(first[1])
        source, n_in, ignored, modality = "nrrd", 1, 0, None
        label = clean_label(os.path.splitext(os.path.basename(first[0]))[0])
        vol, factors = _downsample(vol, MAX_VOXELS)
    else:
        # A série DICOM já sai reduzida: decodificar o volume inteiro antes de
        # reduzir custaria memória do tamanho da entrada.
        vol, dtype, factors, n_in, ignored, modality, label = _read_dicom(_chain(first, it))
        source = "dicom"

    out = _write_nrrd(vol, dtype)
    nk, nj, ni = vol.data.shape
    spacing = tuple(float(round(v, 6)) for v in np.linalg.norm(vol.directions, axis=1))
    return out, ExamStats(
        source=source,
        shape=(ni, nj, nk),
        spacing=spacing,  # type: ignore[arg-type]
        dtype=dtype,
        downsample=factors,
        input_files=n_in,
        ignored_files=ignored,
        modality=modality,
        nrrd_bytes=len(out),
        label=label,
    )


_WS = re.compile(r"\s+")


def clean_label(text: str) -> str | None:
    """Nome da série como veio, só sem o que quebra a tela: caracteres de
    controle (incluindo o `\\0` de preenchimento do DICOM), espaços repetidos e
    excesso de tamanho. Vazio → None (quem chama decide o rótulo reserva)."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in text)
    text = _WS.sub(" ", text).strip()
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - 1].rstrip() + "…"
    return text or None


def _chain(first, rest):
    yield first
    yield from rest


# ---- Tipo de saída -------------------------------------------------------------


def _output_dtype(vmin: float, vmax: float, integral: bool) -> str:
    """Menor tipo que guarda os valores sem perda."""
    if integral:
        if vmin >= -32768 and vmax <= 32767:
            return "int16"
        if vmin >= 0 and vmax <= 65535:
            return "uint16"
    return "float32"


# ---- NRRD de entrada -----------------------------------------------------------

# Cada letra do espaço de entrada diz o sentido positivo do eixo; LPS é o
# nosso. R, A e I apontam ao contrário de L, P e S → aquele eixo troca de sinal.
_SPACES = {
    "left-posterior-superior": "LPS",
    "right-anterior-superior": "RAS",
    "left-anterior-superior": "LAS",
    "lps": "LPS",
    "ras": "RAS",
    "las": "LAS",
}


def _to_lps_sign(space_letters: str) -> np.ndarray:
    return np.array([1.0 if c in "LPS" else -1.0 for c in space_letters])


def _read_nrrd(data: bytes) -> tuple[_Volume, str]:
    import nrrd  # import tardio: só o caminho do exame paga o import

    fh = io.BytesIO(data)
    try:
        header = nrrd.read_header(fh)
    except Exception as e:
        raise ValueError(f"Arquivo NRRD inválido ou corrompido: {e}")

    if "data file" in header or "datafile" in header:
        raise ValueError(
            "Este NRRD tem os dados num arquivo separado (.nhdr + .raw). "
            "Exporte como um único arquivo .nrrd."
        )
    if int(header.get("dimension", 0)) != 3:
        raise ValueError(
            "O NRRD precisa ser um volume 3D em tons de cinza "
            f"(este tem dimensão {header.get('dimension')})."
        )

    try:
        itemsize = np.dtype(nrrd.reader._determine_datatype(header)).itemsize
    except Exception:
        itemsize = 8
    n_bytes = int(np.prod([int(v) for v in header.get("sizes", [0])])) * itemsize
    if n_bytes > MAX_NRRD_INPUT_BYTES:
        raise ValueError(
            f"O NRRD descompactado teria {n_bytes / 1024 / 1024:.0f} MB, acima do limite de "
            f"{MAX_NRRD_INPUT_BYTES // (1024 * 1024)} MB. Exporte o volume com menos "
            "resolução (3D Slicer: Crop Volume com espaçamento maior)."
        )

    try:
        arr = nrrd.read_data(header, fh, index_order="C")
    except Exception as e:
        raise ValueError(f"Não foi possível ler os dados do NRRD: {e}")
    del data, fh

    sizes = [int(s) for s in header["sizes"]]  # (ni, nj, nk)
    if arr.shape != (sizes[2], sizes[1], sizes[0]):
        raise ValueError("NRRD inconsistente: o tamanho dos dados não bate com o cabeçalho.")

    dirs = header.get("space directions")
    if dirs is None:
        spacings = header.get("spacings")
        if spacings is None:
            spacings = [1.0, 1.0, 1.0]
        dirs = np.diag([float(s) for s in spacings])
    dirs = np.asarray(dirs, dtype=float)
    if dirs.shape != (3, 3) or not np.all(np.isfinite(dirs)):
        raise ValueError("O NRRD precisa ter geometria 3D (space directions) nos três eixos.")
    origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)

    space = str(header.get("space", "left-posterior-superior")).strip().lower()
    letters = _SPACES.get(space)
    if letters is None:
        raise ValueError(
            f"Espaço de coordenadas do NRRD não suportado: '{space}'. "
            "Exporte em LPS ou RAS (padrão do 3D Slicer)."
        )
    sign = _to_lps_sign(letters)
    dirs = dirs * sign  # multiplica a componente x, y, z de cada vetor
    origin = origin * sign

    integral = np.issubdtype(arr.dtype, np.integer)
    vmin, vmax = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 0.0)
    dtype = _output_dtype(vmin, vmax, integral)
    arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
    return _Volume(arr, dirs, origin), dtype


# ---- DICOM ---------------------------------------------------------------------


@dataclass
class _Slice:
    ipp: np.ndarray
    ds: object | None  # pydicom Dataset ainda sem decodificar; None depois de usado
    slope: float
    intercept: float


@dataclass
class _Series:
    description: str
    rows: int
    cols: int
    iop: np.ndarray
    pixel_spacing: tuple[float, float]
    monochrome1: bool
    modality: str | None
    label: str | None
    slices: list[_Slice]


def _ensure_transfer_syntax(ds) -> None:
    """Arquivo sem preâmbulo nem meta de arquivo (saída crua de alguns PACS):
    o pydicom lê com force=True, mas só decodifica os pixels sabendo a Transfer
    Syntax. Ela é a codificação com que o conjunto de dados foi lido."""
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian

    meta = getattr(ds, "file_meta", None)
    if meta is not None and "TransferSyntaxUID" in meta:
        return
    implicit, little = ds.original_encoding
    if meta is None:
        ds.file_meta = meta = FileMetaDataset()
    meta.TransferSyntaxUID = (
        ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian if little else ExplicitVRBigEndian
    )


def _looks_like_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def _read_dicom(items: Iterable[tuple[str, bytes]]):
    import pydicom
    from pydicom.errors import InvalidDicomError

    series: dict[str, _Series] = {}
    n_in = 0
    ignored = 0
    multiframe = 0

    for name, data in items:
        n_in += 1
        if _is_nrrd(name, data):
            raise ValueError(
                "Envie o exame como um único arquivo .nrrd ou como uma série DICOM, "
                "não os dois juntos."
            )
        # Sniff pelo preâmbulo "DICM"; sem ele, só tenta se a extensão pede.
        # Arquivos de PACS muitas vezes não têm extensão (IM000001).
        # Nomes de PACS também vêm como UID ("1.2.840.113619.2.55.3"): a
        # "extensão" numérica não diz nada.
        force = not _looks_like_dicom(data)
        ext = _ext(name)
        if force and ext not in ("", ".dcm", ".dicom", ".ima") and not ext[1:].isdigit():
            ignored += 1
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=force)
        except (InvalidDicomError, Exception):
            ignored += 1
            continue
        del data

        if "PixelData" not in ds:  # DICOMDIR, SR, relatório de dose…
            ignored += 1
            continue
        _ensure_transfer_syntax(ds)
        if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
            multiframe += 1
            continue
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            ignored += 1  # captura colorida (tela, foto) não vira volume
            continue
        # Reconstruções (MPR, MIP) de alguns aparelhos levam, na própria série,
        # uma imagem de referência dos planos em outra orientação, marcada
        # LOCALIZER no ImageType (visto numa TC Siemens do TCIA). Ela não é
        # fatia do volume: fica de fora, como o scout.
        if "LOCALIZER" in (str(v).upper() for v in (getattr(ds, "ImageType", None) or [])):
            ignored += 1
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ps = getattr(ds, "PixelSpacing", None)
        if ipp is None or iop is None or ps is None:
            ignored += 1  # sem geometria não dá para empilhar
            continue

        uid = str(getattr(ds, "SeriesInstanceUID", "") or "sem-uid")
        s = series.get(uid)
        if s is None:
            s = _Series(
                description=str(getattr(ds, "SeriesDescription", "") or "sem descrição"),
                rows=int(ds.Rows),
                cols=int(ds.Columns),
                iop=np.array([float(v) for v in iop]),
                pixel_spacing=(float(ps[0]), float(ps[1])),
                monochrome1=photometric == "MONOCHROME1",
                modality=str(getattr(ds, "Modality", "") or "") or None,
                label=clean_label(str(getattr(ds, "SeriesDescription", "") or "")),
                slices=[],
            )
            series[uid] = s
        else:
            if (int(ds.Rows), int(ds.Columns)) != (s.rows, s.cols):
                raise ValueError(
                    f"A série '{s.description}' mistura imagens de tamanhos diferentes."
                )
            if np.max(np.abs(np.array([float(v) for v in iop]) - s.iop)) > IOP_TOL:
                raise ValueError(
                    f"A série '{s.description}' mistura orientações diferentes "
                    "(por exemplo axial e coronal). Envie só a série usada na segmentação."
                )
        s.slices.append(
            _Slice(
                ipp=np.array([float(v) for v in ipp]),
                ds=ds,
                slope=float(getattr(ds, "RescaleSlope", 1) or 1),
                intercept=float(getattr(ds, "RescaleIntercept", 0) or 0),
            )
        )
        del ds

    small = {u: s for u, s in series.items() if len(s.slices) < MIN_SLICES_PER_SERIES}
    big = {u: s for u, s in series.items() if len(s.slices) >= MIN_SLICES_PER_SERIES}
    ignored += sum(len(s.slices) for s in small.values())

    if not big:
        if multiframe:
            print(f"[exam] DICOM multi-frame recebido ({multiframe} arquivo(s)) — não suportado")
            raise ValueError(
                "Este exame está no formato DICOM multi-frame (todas as imagens num "
                "arquivo só), que ainda não é aceito. Exporte a série como imagens "
                "separadas ou como .nrrd (3D Slicer)."
            )
        if small:
            n = max(len(s.slices) for s in small.values())
            raise ValueError(
                f"A série DICOM tem só {n} imagem(ns); o mínimo para montar o volume "
                f"é {MIN_SLICES_PER_SERIES}."
            )
        raise ValueError(
            "Nenhuma imagem DICOM em tons de cinza com posição no paciente foi "
            "encontrada nos arquivos enviados."
        )
    if len(big) > 1:
        lista = "; ".join(
            f"'{s.description}' ({len(s.slices)} imagens)"
            for s in sorted(big.values(), key=lambda s: -len(s.slices))
        )
        raise ValueError(
            f"Os arquivos têm mais de uma série de imagens: {lista}. "
            "Envie só a série usada na segmentação."
        )

    s = next(iter(big.values()))
    series.clear()  # solta as séries descartadas antes de decodificar a escolhida
    vol, dtype, factors = _stack_series(s)
    return vol, dtype, factors, n_in, ignored, s.modality, s.label


def _stack_series(s: _Series) -> tuple[_Volume, str]:
    row = s.iop[:3]
    col = s.iop[3:]
    normal = np.cross(row, col)

    # Ordem pela posição ao longo da normal — nunca por InstanceNumber, que o
    # PACS às vezes numera ao contrário ou reinicia.
    s.slices.sort(key=lambda sl: float(np.dot(sl.ipp, normal)))
    pos = np.array([float(np.dot(sl.ipp, normal)) for sl in s.slices])
    gaps = np.diff(pos)
    median = float(np.median(gaps))
    if median <= SPACING_TOL_MM or np.any(np.abs(gaps) <= SPACING_TOL_MM):
        raise ValueError(
            f"A série '{s.description}' tem imagens repetidas na mesma posição "
            "(duas séries misturadas ou arquivos duplicados)."
        )
    tol = max(SPACING_TOL_MM, SPACING_TOL_REL * median)
    if np.any(np.abs(gaps - median) > tol):
        raise ValueError(
            f"A série '{s.description}' tem espaçamento irregular entre as imagens — "
            "provavelmente faltam imagens. Envie a série completa."
        )

    n = len(s.slices)
    ps_row, ps_col = s.pixel_spacing  # (entre linhas, entre colunas)
    d_k0 = (s.slices[-1].ipp - s.slices[0].ipp) / (n - 1)

    # Redução planejada só pela geometria (a mesma de _downsample) e aplicada
    # enquanto decodifica, fatia a fatia: o pico de memória é o volume de SAÍDA.
    sizes = (s.cols, s.rows, n)
    rs = _Resampler(sizes, _plan_resample(sizes, (ps_col, ps_row, float(np.linalg.norm(d_k0))), MAX_VOXELS))
    integral = all(sl.slope.is_integer() and sl.intercept.is_integer() for sl in s.slices)

    for k, sl in enumerate(s.slices):
        ds, sl.ds = sl.ds, None
        try:
            px = ds.pixel_array
        except Exception as e:
            raise ValueError(
                "Não foi possível decodificar as imagens DICOM "
                f"(compressão {ds.file_meta.get('TransferSyntaxUID', '?')}): {e}"
            )
        del ds
        rs.add(k, px.astype(np.float64) * sl.slope + sl.intercept)

    acc = rs.acc
    vmin, vmax = (float(acc.min()), float(acc.max())) if acc.size else (0.0, 0.0)
    if s.monochrome1:
        # MONOCHROME1: valor alto = preto. Invertemos para que "claro" signifique
        # o mesmo em qualquer série.
        acc = (vmin + vmax) - acc
    dtype = _output_dtype(vmin, vmax, integral)
    out = np.rint(acc).astype(dtype) if dtype != "float32" else acc.astype(np.float32)
    del acc

    d = np.stack([row * ps_col, col * ps_row, d_k0])
    origin, directions = rs.geometry(s.slices[0].ipp, d)
    return _Volume(out, directions, origin), dtype, rs.factors


# ---- Redução -------------------------------------------------------------------


def _plan_resample(sizes, spacing, max_voxels: int) -> tuple[int, int, int]:
    """Tamanhos de saída (ni, nj, nk) com o "piso de espaçamento".

    Acha o menor espaçamento s tal que, levando a s cada eixo mais fino que ele
    (os mais grossos ficam como estão), o volume cabe em `max_voxels`. O eixo a
    fica com floor(extensão_a / s) amostras. Eixos de mesmo espaçamento
    encolhem juntos: o pixel do plano continua quadrado, em vez de um eixo só
    perder a metade (a regra antiga deixava o corte axial com 1,48 × 0,74 mm).
    """
    n = np.asarray(sizes, dtype=np.int64)
    sp = np.asarray(spacing, dtype=float)
    if int(np.prod(n)) <= max_voxels:
        return int(n[0]), int(n[1]), int(n[2])
    extent = n * sp

    def out(s: float) -> np.ndarray:
        return np.clip(np.floor(extent / s + 1e-9).astype(np.int64), 1, n)

    lo, hi = float(sp.min()), float(extent.max())
    for _ in range(64):  # busca binária: out(s) só diminui com s
        mid = (lo + hi) / 2
        if int(np.prod(out(mid))) <= max_voxels:
            hi = mid
        else:
            lo = mid
    o = out(hi)
    return int(o[0]), int(o[1]), int(o[2])


def _box_weights(n_in: int, n_out: int) -> np.ndarray:
    """Matriz (n_out × n_in) de média de área: a amostra nova b cobre as
    antigas [b·f, (b+1)·f), f = n_in / n_out (não precisa ser inteiro), cada
    uma pesando o quanto se sobrepõe. Linhas somam 1; f inteiro = média de
    bloco."""
    f = n_in / n_out
    lo = np.arange(n_out)[:, None] * f
    i = np.arange(n_in)[None, :]
    return np.clip(np.minimum(lo + f, i + 1) - np.maximum(lo, i), 0.0, None) / f


class _Resampler:
    """Reamostra um volume (ni, nj, nk) → `out_sizes` recebendo uma fatia k
    por vez: no plano, Wj · fatia · Wiᵀ; ao longo de k, cada fatia soma nas
    amostras novas que ela cobre. Memória: o volume de saída (float32) e as
    matrizes de peso. Serve à série DICOM (enquanto decodifica) e ao NRRD."""

    def __init__(self, sizes, out_sizes):
        ni, nj, nk = sizes
        ni2, nj2, nk2 = out_sizes
        self.sizes, self.out_sizes = (ni, nj, nk), (ni2, nj2, nk2)
        self.wi = None if ni2 == ni else _box_weights(ni, ni2)
        self.wj = None if nj2 == nj else _box_weights(nj, nj2)
        self.wk = None if nk2 == nk else _box_weights(nk, nk2)
        self.acc = np.zeros((nk2, nj2, ni2), dtype=np.float32)

    def add(self, k: int, px: np.ndarray) -> None:
        """px: fatia k em (nj, ni), já na escala final (slope/intercept)."""
        if self.wj is not None:
            px = self.wj @ px
        if self.wi is not None:
            px = px @ self.wi.T
        if self.wk is None:
            self.acc[k] = px
            return
        col = self.wk[:, k]
        for b in np.flatnonzero(col):
            self.acc[b] += (col[b] * px).astype(np.float32)

    @property
    def factors(self) -> tuple[float, float, float]:
        return tuple(round(a / b, 4) for a, b in zip(self.sizes, self.out_sizes))  # type: ignore[return-value]

    def geometry(self, origin: np.ndarray, directions: np.ndarray):
        """Origem e direções do volume novo. O centro da amostra b fica em
        b·f + (f−1)/2 no índice antigo: a origem anda (f−1)/2."""
        f = np.array([a / b for a, b in zip(self.sizes, self.out_sizes)], dtype=float)
        return origin + ((f - 1) / 2) @ directions, directions * f[:, None]


def _downsample(vol: _Volume, max_voxels: int) -> tuple[_Volume, tuple[float, float, float]]:
    """Reduz um volume já montado (entrada .nrrd) até caber em `max_voxels`,
    com o mesmo plano e o mesmo _Resampler da série DICOM."""
    nk, nj, ni = vol.data.shape
    sizes = (ni, nj, nk)
    out_sizes = _plan_resample(sizes, np.linalg.norm(vol.directions, axis=1), max_voxels)
    if out_sizes == sizes:
        return vol, (1.0, 1.0, 1.0)
    rs = _Resampler(sizes, out_sizes)
    for k in range(nk):  # uma fatia por vez: memória de poucas fatias além da saída
        rs.add(k, vol.data[k].astype(np.float64))
    dtype = vol.data.dtype
    out = np.rint(rs.acc).astype(dtype) if np.issubdtype(dtype, np.integer) else rs.acc.astype(dtype)
    origin, directions = rs.geometry(vol.origin, vol.directions)
    return _Volume(out, directions, origin), rs.factors


# ---- Saída ---------------------------------------------------------------------


def _write_nrrd(vol: _Volume, dtype: str) -> bytes:
    import nrrd

    data = np.ascontiguousarray(vol.data.astype(np.dtype(dtype).newbyteorder("<"), copy=False))
    header = {
        "space": CANONICAL_SPACE,
        "space directions": vol.directions,
        "kinds": ["domain", "domain", "domain"],
        "endian": "little",
        "encoding": "gzip",
        "space origin": vol.origin,
    }
    buf = io.BytesIO()
    # index_order="C": o array é (nk, nj, ni) e o pynrrd grava sizes = ni nj nk,
    # a mesma ordem dos vetores em `space directions`.
    nrrd.write(buf, data, header, index_order="C", compression_level=GZIP_LEVEL)
    return buf.getvalue()
