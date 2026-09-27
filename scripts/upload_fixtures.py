"""Gera as fixtures DICOM de várias séries da página de upload. Ferramenta de dev.

    .venv/bin/python scripts/upload_fixtures.py ../medCaseViewer/tests/upload/fixtures

Grava em <pasta>/multi/ um "CD" com cinco séries sintéticas 16×16, sem
extensão como sai de um PACS, mais o lixo de sempre:

    arterial/     ARTERIAL 1.0      10 imagens  FrameOfReference A
    nefro/        NEFROGRAFICA 1.0  12 imagens  FrameOfReference A
    torax/        ANGIO TORAX 1.0    9 imagens  FrameOfReference B (outro exame)
    portal/       PORTAL 1.0         9 imagens  A, falta a 5ª (espaçamento irregular)
    localizador/  LOCALIZADOR        2 imagens  (fica de fora: < 8)
    VIEWER.EXE, index.html

e, ao lado: multi.zip (a pasta inteira, deflate), crua/ (8 imagens sem o
preâmbulo DICM, VR implícito, como alguns PACS exportam), mpr/ (reconstrução
coronal de 10 fatias com 1 imagem LOCALIZER axial na mesma série, como a
Siemens grava — a referência fica de fora), misto.zip (um STL e uma imagem
DICOM juntos: a página pede para separar) e stl.zip (STL compactado: a página
pede para descompactar).
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import zipfile

import numpy as np
from pydicom.uid import generate_uid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_exam import make_slice  # noqa: E402

FOR_A = "1.2.826.0.1.3680043.8.498.1"
FOR_B = "1.2.826.0.1.3680043.8.498.2"

SERIES = [
    ("arterial", "ARTERIAL 1.0", 10, FOR_A, None),
    ("nefro", "NEFROGRAFICA 1.0", 12, FOR_A, None),
    ("torax", "ANGIO TORAX 1.0", 9, FOR_B, None),
    ("portal", "PORTAL 1.0", 10, FOR_A, 4),  # sem a 5ª imagem
    ("localizador", "LOCALIZADOR", 2, FOR_A, None),
]


def _pixels(k: int) -> np.ndarray:
    return (np.arange(256, dtype=np.int16).reshape(16, 16) + k * 10).astype(np.int16)


def _raw_slice(k: int, series_uid: str) -> bytes:
    """Imagem sem preâmbulo nem meta de arquivo: VR implícito cru."""
    from pydicom import dcmread

    ds = dcmread(io.BytesIO(make_slice(k, series_uid=series_uid, description="CRUA", pixels=_pixels(k))))
    del ds.file_meta
    ds.preamble = None
    buf = io.BytesIO()
    ds.save_as(buf, implicit_vr=True, little_endian=True, enforce_file_format=False)
    return buf.getvalue()


def main(out: str) -> None:
    multi = os.path.join(out, "multi")
    shutil.rmtree(multi, ignore_errors=True)
    for folder, desc, n, frame, skip in SERIES:
        d = os.path.join(multi, folder)
        os.makedirs(d)
        uid = generate_uid()
        for k in range(n):
            if k == skip:
                continue
            data = make_slice(k, series_uid=uid, description=desc, frame_of_reference=frame, pixels=_pixels(k))
            with open(os.path.join(d, f"IM{k + 1:04d}"), "wb") as f:
                f.write(data)
    with open(os.path.join(multi, "VIEWER.EXE"), "wb") as f:
        f.write(b"MZ" + b"\0" * 64)
    with open(os.path.join(multi, "index.html"), "w") as f:
        f.write("<html>visualizador do CD</html>")

    with zipfile.ZipFile(os.path.join(out, "multi.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(multi):
            for name in sorted(files):
                p = os.path.join(root, name)
                zf.write(p, os.path.relpath(p, out))

    crua = os.path.join(out, "crua")
    shutil.rmtree(crua, ignore_errors=True)
    os.makedirs(crua)
    uid = generate_uid()
    for k in range(8):
        with open(os.path.join(crua, f"{k + 1:04d}"), "wb") as f:
            f.write(_raw_slice(k, uid))

    mpr = os.path.join(out, "mpr")
    shutil.rmtree(mpr, ignore_errors=True)
    os.makedirs(mpr)
    uid = generate_uid()
    coronal = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    for k in range(10):
        data = make_slice(k, series_uid=uid, description="CORONAL MPR", iop=coronal,
                          frame_of_reference=FOR_A, image_type=["DERIVED", "PRIMARY", "AXIAL"], pixels=_pixels(k))
        with open(os.path.join(mpr, f"IM{k + 1:04d}"), "wb") as f:
            f.write(data)
    data = make_slice(0, series_uid=uid, description="CORONAL MPR", frame_of_reference=FOR_A,
                      image_type=["DERIVED", "PRIMARY", "LOCALIZER"], pixels=_pixels(0))
    with open(os.path.join(mpr, "IM0099"), "wb") as f:
        f.write(data)

    rim = os.path.join(out, "Rim.stl")
    one = os.path.join(multi, "arterial", "IM0001")
    with zipfile.ZipFile(os.path.join(out, "misto.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rim, "Rim.stl")
        zf.write(one, "IM0001")
    with zipfile.ZipFile(os.path.join(out, "stl.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rim, "Rim.stl")
    print(f"{out}: multi/ (5 séries), multi.zip, crua/, mpr/, misto.zip, stl.zip")


if __name__ == "__main__":
    main(sys.argv[1])
