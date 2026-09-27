"""Testes do exam.py — série DICOM / NRRD → NRRD canônico.

Fixtures sintéticas geradas aqui mesmo (pydicom + pynrrd), sem arquivo externo;
o teste de integração com o exame real só roda se ele estiver em ~/Downloads.

O valor de cada voxel sintético codifica a própria posição (100·k + 10·j + i),
então qualquer troca de ordem entre fatias ou transposição de eixos aparece
como dado errado, não só como geometria errada.
"""
from __future__ import annotations

import glob
import io
import os
import random
import zipfile

import numpy as np
import pytest
import nrrd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import exam
from exam import MAX_LABEL_CHARS, clean_label, normalize_exam

ROWS, COLS = 6, 5  # j, i — diferentes de propósito
SPACING = (0.8, 0.7)  # (entre linhas, entre colunas)
STEP = 1.5


def _pattern(k: int) -> np.ndarray:
    j, i = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
    return (100 * k + 10 * j + i).astype(np.int16)


def _oblique_iop(deg: float = 10.0) -> tuple[float, ...]:
    a = np.deg2rad(deg)
    return (1.0, 0.0, 0.0, 0.0, float(np.cos(a)), float(-np.sin(a)))


def make_slice(
    k: int,
    *,
    series_uid: str,
    iop=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    origin=(-10.0, 20.0, 30.0),
    description: str = "Axial",
    pixels: np.ndarray | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric: str = "MONOCHROME2",
    position_k: float | None = None,
    patient_name: str = "Fulana de Tal",
    frame_of_reference: str | None = None,
    image_type: list[str] | None = None,
) -> bytes:
    """Uma fatia DICOM com geometria real, serializada com preâmbulo DICM."""
    row = np.array(iop[:3])
    col = np.array(iop[3:])
    normal = np.cross(row, col)
    pk = k if position_k is None else position_k
    ipp = np.array(origin) + normal * STEP * pk

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = patient_name
    ds.PatientID = "123456"
    ds.Modality = "MR"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = description
    if frame_of_reference:
        ds.FrameOfReferenceUID = frame_of_reference
    if image_type:
        ds.ImageType = image_type
    ds.InstanceNumber = k + 1
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.ImageOrientationPatient = [float(v) for v in iop]
    ds.PixelSpacing = list(SPACING)
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    px = _pattern(k) if pixels is None else pixels
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if px.dtype == np.int16 else 0
    ds.PixelData = px.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def series(n: int = 10, **kw) -> list[tuple[str, bytes]]:
    uid = kw.pop("series_uid", generate_uid())
    return [(f"IM{k:04d}.dcm", make_slice(k, series_uid=uid, **kw)) for k in range(n)]


def read_out(nrrd_bytes: bytes):
    fh = io.BytesIO(nrrd_bytes)
    header = nrrd.read_header(fh)
    data = nrrd.read_data(header, fh, index_order="C")
    return header, data


# ---- DICOM ---------------------------------------------------------------------


def test_dicom_shuffled_oblique_series_is_sorted_by_position_with_right_geometry():
    iop = _oblique_iop(10)
    items = series(10, iop=iop)
    random.Random(7).shuffle(items)
    out, stats = normalize_exam(items)
    header, data = read_out(out)

    assert stats.source == "dicom"
    assert stats.shape == (COLS, ROWS, 10)
    assert list(header["sizes"]) == [COLS, ROWS, 10]
    assert data.shape == (10, ROWS, COLS)
    for k in range(10):
        assert np.array_equal(data[k], _pattern(k)), f"fatia {k} fora de ordem ou transposta"

    row, col = np.array(iop[:3]), np.array(iop[3:])
    normal = np.cross(row, col)
    d = header["space directions"]
    assert np.allclose(d[0], row * SPACING[1])  # i anda nas colunas
    assert np.allclose(d[1], col * SPACING[0])  # j anda nas linhas
    assert np.allclose(d[2], normal * STEP)
    assert np.allclose(header["space origin"], [-10.0, 20.0, 30.0])
    assert header["space"] == "left-posterior-superior"


def test_output_header_has_only_geometric_fields_and_no_phi():
    out, _ = normalize_exam(series(10, patient_name="Fulana de Tal"))
    header, _ = read_out(out)
    assert set(header) == {
        "type", "dimension", "space", "sizes", "space directions",
        "kinds", "endian", "encoding", "space origin",
    }
    assert header["encoding"] == "gzip"
    assert b"Fulana" not in out and b"123456" not in out


def test_zip_of_the_series_gives_the_same_volume():
    items = series(10)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in items:
            zf.writestr(f"pasta/{name}", data)
        zf.writestr("__MACOSX/pasta/._IM0000.dcm", b"lixo")
        zf.writestr("pasta/.DS_Store", b"lixo")
    loose, _ = normalize_exam(items)
    zipped, stats = normalize_exam([("serie.zip", buf.getvalue())])
    assert read_out(loose)[1].tobytes() == read_out(zipped)[1].tobytes()
    assert stats.input_files == 10


def test_localizer_and_junk_are_ignored():
    items = series(10) + series(2, description="Localizer")
    items.append(("LEIAME.txt", b"texto qualquer"))
    items.append(("DICOMDIR", b"\0" * 200))
    out, stats = normalize_exam(items)
    assert stats.shape[2] == 10
    assert stats.ignored_files == 4


def test_two_real_series_are_refused_listing_both():
    items = series(10, description="T2 axial") + series(9, description="T1 axial")
    with pytest.raises(ValueError) as e:
        normalize_exam(items)
    assert "T2 axial" in str(e.value) and "T1 axial" in str(e.value)


def test_missing_slice_is_refused():
    items = series(10)
    del items[4]
    with pytest.raises(ValueError, match="espaçamento irregular"):
        normalize_exam(items)


def test_duplicate_position_is_refused():
    uid = generate_uid()
    items = series(10, series_uid=uid)
    items.append(("dup.dcm", make_slice(3, series_uid=uid)))
    with pytest.raises(ValueError, match="repetidas"):
        normalize_exam(items)


def test_mixed_orientation_within_a_series_is_refused():
    uid = generate_uid()
    items = series(10, series_uid=uid)
    items.append(("cor.dcm", make_slice(11, series_uid=uid, iop=_oblique_iop(30))))
    with pytest.raises(ValueError, match="orientações"):
        normalize_exam(items)


def test_multiframe_is_refused_with_clear_message():
    data = make_slice(0, series_uid=generate_uid())
    ds = pydicom.dcmread(io.BytesIO(data))
    ds.NumberOfFrames = 2
    ds.PixelData = np.concatenate([_pattern(0), _pattern(1)]).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    with pytest.raises(ValueError, match="multi-frame"):
        normalize_exam([("mf.dcm", buf.getvalue())])


def test_too_few_slices_is_refused():
    with pytest.raises(ValueError, match="mínimo"):
        normalize_exam(series(3))


def test_monochrome1_is_inverted():
    out, _ = normalize_exam(series(10, photometric="MONOCHROME1"))
    _, data = read_out(out)
    vmin, vmax = 0, 100 * 9 + 10 * (ROWS - 1) + (COLS - 1)
    assert data[0, 0, 0] == vmax  # era o menor valor
    assert data[9, ROWS - 1, COLS - 1] == vmin


def test_integer_rescale_stays_int16_and_is_applied():
    out, stats = normalize_exam(series(10, intercept=-1024))
    _, data = read_out(out)
    assert stats.dtype == "int16"
    assert data[2, 1, 1] == 100 * 2 + 10 + 1 - 1024


def test_fractional_rescale_becomes_float32():
    out, stats = normalize_exam(series(10, slope=0.5))
    header, data = read_out(out)
    assert stats.dtype == "float32" and header["type"] == "float"
    assert data[2, 1, 1] == pytest.approx((100 * 2 + 10 + 1) * 0.5)


def test_unsigned_values_above_int16_stay_uint16():
    px = np.full((ROWS, COLS), 60000, dtype=np.uint16)
    out, stats = normalize_exam(series(10, pixels=px))
    assert stats.dtype == "uint16"
    assert read_out(out)[1].max() == 60000


def test_jpeg2000_series_is_decoded():
    from pydicom.uid import JPEG2000Lossless

    # O OpenJPEG recusa imagens minúsculas (poucos níveis de decomposição);
    # 64×64 é o menor tamanho realista.
    def big(k: int) -> np.ndarray:
        j, i = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        return (1000 * k + 10 * j + i).astype(np.int16)

    uid = generate_uid()
    items = []
    for k in range(10):
        ds = pydicom.dcmread(io.BytesIO(make_slice(k, series_uid=uid, pixels=big(k))))
        try:
            ds.compress(JPEG2000Lossless, encoding_plugin="pylibjpeg")
        except Exception as e:  # encoder ausente nesta máquina
            pytest.skip(f"sem encoder JPEG2000: {e}")
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        items.append((f"j2k{k}.dcm", buf.getvalue()))
    out, _ = normalize_exam(items)
    _, data = read_out(out)
    assert np.array_equal(data[5], big(5))


# ---- NRRD ----------------------------------------------------------------------


def _nrrd_bytes(data_kji: np.ndarray, **header) -> bytes:
    h = {"kinds": ["domain"] * 3, "encoding": "raw", **header}
    buf = io.BytesIO()
    nrrd.write(buf, data_kji, h, index_order="C")
    return buf.getvalue()


def test_nrrd_axis_order_is_preserved_on_an_asymmetric_volume():
    ni, nj, nk = 4, 6, 8
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    data = (100 * k + 10 * j + i).astype(np.int16)
    dirs = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    src = _nrrd_bytes(data, space="left-posterior-superior",
                      **{"space directions": dirs, "space origin": np.array([1.0, 2, 3])})
    out, stats = normalize_exam([("vol.nrrd", src)])
    header, back = read_out(out)
    assert list(header["sizes"]) == [4, 6, 8]
    assert stats.shape == (4, 6, 8)
    assert np.array_equal(back, data)
    assert np.allclose(header["space directions"], dirs)


def test_nrrd_in_ras_is_converted_to_lps():
    data = np.zeros((8, 6, 4), dtype=np.int16)
    dirs = np.array([[0.9, 0.1, 0], [0.2, 1.1, 0.3], [0, 0.1, 2.0]])
    origin = np.array([10.0, -20.0, 30.0])
    src = _nrrd_bytes(data, space="right-anterior-superior",
                      **{"space directions": dirs, "space origin": origin})
    header, _ = read_out(normalize_exam([("ras.nrrd", src)])[0])
    flip = np.array([-1.0, -1.0, 1.0])
    assert header["space"] == "left-posterior-superior"
    assert np.allclose(header["space directions"], dirs * flip)
    assert np.allclose(header["space origin"], origin * flip)


def test_nrrd_custom_fields_are_dropped():
    data = np.zeros((8, 6, 4), dtype=np.uint16)
    buf = io.BytesIO()
    nrrd.write(buf, data, {
        "space": "left-posterior-superior",
        "space directions": np.eye(3),
        "space origin": np.zeros(3),
        "kinds": ["domain"] * 3,
        "PatientName": "Fulana de Tal",
    }, index_order="C", custom_field_map={"PatientName": "string"})
    out, _ = normalize_exam([("com-phi.nrrd", buf.getvalue())])
    assert b"Fulana" not in out
    assert "PatientName" not in read_out(out)[0]


def test_nrrd_4d_is_refused():
    data = np.zeros((2, 8, 6, 4), dtype=np.int16)
    buf = io.BytesIO()
    nrrd.write(buf, data, {"encoding": "raw"}, index_order="C")
    with pytest.raises(ValueError, match="3D"):
        normalize_exam([("4d.nrrd", buf.getvalue())])


def test_nrrd_and_dicom_together_are_refused():
    src = _nrrd_bytes(np.zeros((8, 6, 4), dtype=np.int16), space="left-posterior-superior",
                      **{"space directions": np.eye(3), "space origin": np.zeros(3)})
    with pytest.raises(ValueError, match="não os dois"):
        normalize_exam([("a.nrrd", src)] + series(10))
    with pytest.raises(ValueError, match="não os dois"):
        normalize_exam(series(10) + [("a.nrrd", src)])


def test_empty_exam_is_refused():
    with pytest.raises(ValueError):
        normalize_exam([])


# ---- Redução -------------------------------------------------------------------


def test_downsample_halves_the_smallest_spacing_axis_first(monkeypatch):
    ni, nj, nk = 8, 8, 20
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    data = (10 * k).astype(np.int16)  # só varia em k
    dirs = np.diag([1.0, 1.0, 0.5])
    origin = np.array([0.0, 0.0, 0.0])
    src = _nrrd_bytes(data, space="left-posterior-superior",
                      **{"space directions": dirs, "space origin": origin})
    monkeypatch.setattr(exam, "MAX_VOXELS", 700)
    out, stats = normalize_exam([("big.nrrd", src)])
    header, back = read_out(out)
    assert stats.downsample == (1, 1, 2)
    assert stats.shape == (8, 8, 10)
    assert np.allclose(header["space directions"], np.diag([1.0, 1.0, 1.0]))
    # Centro do voxel novo = média dos centros dos dois antigos.
    assert np.allclose(header["space origin"], [0.0, 0.0, 0.25])
    # Média de 10·(2t) e 10·(2t+1) = 20t + 5.
    assert back[3, 0, 0] == 65


def test_downsample_goes_in_plane_when_that_is_finer(monkeypatch):
    data = np.ones((10, 16, 16), dtype=np.int16)
    src = _nrrd_bytes(data, space="left-posterior-superior",
                      **{"space directions": np.diag([0.5, 0.5, 3.0]), "space origin": np.zeros(3)})
    monkeypatch.setattr(exam, "MAX_VOXELS", 16 * 16 * 10 // 4)
    _, stats = normalize_exam([("ct.nrrd", src)])
    assert stats.downsample == (2, 2, 1)


# ---- Exame real (só na máquina de desenvolvimento) -------------------------------

REAL_DICOM = os.path.expanduser("~/Downloads/ScalarVolume_9")
REAL_NRRD = os.path.expanduser("~/Downloads/11.nrrd")


@pytest.mark.skipif(
    not (os.path.isdir(REAL_DICOM) and os.path.isfile(REAL_NRRD)),
    reason="exame de exemplo ausente",
)
def test_real_dicom_series_and_its_nrrd_export_normalize_to_the_same_volume():
    def items(paths):
        for p in paths:
            with open(p, "rb") as f:
                yield os.path.basename(p), f.read()

    d_out, d_stats = normalize_exam(items(sorted(glob.glob(os.path.join(REAL_DICOM, "*.dcm")))))
    n_out, n_stats = normalize_exam(items([REAL_NRRD]))
    dh, dd = read_out(d_out)
    nh, nd = read_out(n_out)
    assert d_stats.shape == n_stats.shape == (256, 256, 144)
    assert np.allclose(dh["space directions"], nh["space directions"], atol=0.01)
    assert np.allclose(dh["space origin"], nh["space origin"], atol=0.01)
    assert np.array_equal(dd.astype(np.int32), nd.astype(np.int32))


# ---- Revisão: memória, zip e nomes de arquivo --------------------------------------


def test_dicom_series_downsampled_while_decoding_matches_the_nrrd_path(monkeypatch):
    # A série DICOM é reduzida fatia a fatia (sem montar o volume inteiro); o
    # resultado tem de ser o mesmo que reduzir o volume pronto.
    monkeypatch.setattr(exam, "MAX_VOXELS", COLS * ROWS * 10 // 3)
    items = series(10, iop=_oblique_iop(10))
    d_out, d_stats = normalize_exam(items)

    monkeypatch.setattr(exam, "MAX_VOXELS", 10**9)
    ref_out, _ = normalize_exam(items)
    ref_header, ref = read_out(ref_out)
    vol = exam._Volume(ref, np.asarray(ref_header["space directions"]), np.asarray(ref_header["space origin"]))
    expected, factors = exam._downsample(vol, COLS * ROWS * 10 // 3)

    header, data = read_out(d_out)
    assert d_stats.downsample == factors
    assert np.array_equal(data, expected.data)
    assert np.allclose(header["space directions"], expected.directions)
    assert np.allclose(header["space origin"], expected.origin)


def test_undecodable_pixels_in_a_discarded_series_do_not_break_the_upload():
    items = series(10)
    bad = pydicom.dcmread(io.BytesIO(make_slice(0, series_uid=generate_uid(), description="Scout")))
    bad.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.90"  # diz JPEG2000…
    from pydicom.encaps import encapsulate
    bad.PixelData = encapsulate([b"\x00\x01 isto nao e JPEG2000"])  # …mas não é
    bad["PixelData"].VR = "OB"
    bad["PixelData"].is_undefined_length = True
    buf = io.BytesIO()
    bad.save_as(buf, enforce_file_format=True)
    out, stats = normalize_exam(items + [("scout.dcm", buf.getvalue())])
    assert stats.shape[2] == 10
    assert stats.ignored_files == 1


def test_uid_named_files_without_preamble_are_read():
    uid = generate_uid()
    items = []
    for k in range(10):
        # Sem os 128 bytes de preâmbulo + "DICM", nome no formato de UID.
        raw = make_slice(k, series_uid=uid)[132:]
        items.append((f"1.2.840.113619.2.55.{k}", raw))
    _, stats = normalize_exam(items)
    assert stats.shape == (COLS, ROWS, 10)


def test_corrupted_zip_member_is_a_clear_error():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in series(10):
            zf.writestr(name, data)
    raw = bytearray(buf.getvalue())
    raw[200:260] = b"\xff" * 60  # estraga os dados comprimidos do 1º membro
    with pytest.raises(ValueError, match="zip"):
        normalize_exam([("serie.zip", bytes(raw))])


def test_zip_that_inflates_too_much_is_refused(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in series(10):
            zf.writestr(name, data)
    monkeypatch.setattr(exam, "MAX_UNZIPPED_BYTES", 2000)
    with pytest.raises(ValueError, match="descompactado"):
        normalize_exam([("serie.zip", buf.getvalue())])


def test_huge_nrrd_is_refused_by_the_header(monkeypatch):
    src = _nrrd_bytes(np.zeros((8, 6, 4), dtype=np.int16), space="left-posterior-superior",
                      **{"space directions": np.eye(3), "space origin": np.zeros(3)})
    monkeypatch.setattr(exam, "MAX_NRRD_INPUT_BYTES", 100)
    with pytest.raises(ValueError, match="limite"):
        normalize_exam([("grande.nrrd", src)])


# ---- Rótulo da série (SeriesDescription) ------------------------------------------


def test_series_label_is_the_dicom_series_description():
    _, stats = normalize_exam(series(10, description="ARTERIAL 1.0 B30f"))
    assert stats.label == "ARTERIAL 1.0 B30f"


def test_series_without_description_has_no_label():
    _, stats = normalize_exam(series(10, description=""))
    assert stats.label is None


def test_nrrd_label_is_the_file_name_without_extension():
    src = _nrrd_bytes(np.zeros((4, 5, 6), np.int16))
    _, stats = normalize_exam([("pasta/Fase venosa.nrrd", src)])
    assert stats.label == "Fase venosa"


def test_label_keeps_the_text_and_drops_only_control_characters():
    assert clean_label("  ARTERIAL\x00\x00 ") == "ARTERIAL"
    assert clean_label("T1\tpós\n contraste") == "T1 pós contraste"
    assert clean_label("\x00\x1f  ") is None
    assert clean_label("") is None


def test_long_label_is_cut_with_an_ellipsis():
    out = clean_label("A" * 200)
    assert len(out) == MAX_LABEL_CHARS
    assert out.endswith("…")


def test_raw_dicom_without_preamble_or_file_meta_is_decoded():
    # Alguns PACS exportam o conjunto de dados cru: sem "DICM", sem meta de
    # arquivo, VR implícito. Sem Transfer Syntax o pydicom não decodifica.
    from pydicom import dcmread

    uid = generate_uid()
    items = []
    for k in range(10):
        ds = dcmread(io.BytesIO(make_slice(k, series_uid=uid)))
        del ds.file_meta
        ds.preamble = None
        buf = io.BytesIO()
        ds.save_as(buf, implicit_vr=True, little_endian=True, enforce_file_format=False)
        items.append((f"{k:04d}", buf.getvalue()))
    assert items[0][1][128:132] != b"DICM"
    _, stats = normalize_exam(items)
    assert stats.shape == (5, 6, 10)


def test_localizer_image_inside_a_reformat_series_is_ignored():
    # MPR coronal da Siemens: 130 fatias coronais + 1 imagem LOCALIZER axial
    # (a referência dos planos) com o MESMO SeriesInstanceUID.
    uid = generate_uid()
    coronal = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    items = [(f"IM{k:04d}", make_slice(k, series_uid=uid, iop=coronal,
                                       image_type=["DERIVED", "PRIMARY", "AXIAL"])) for k in range(10)]
    items.append(("IM9999", make_slice(0, series_uid=uid, image_type=["DERIVED", "PRIMARY", "LOCALIZER"])))
    _, stats = normalize_exam(items)
    assert stats.shape[2] == 10
    assert stats.ignored_files == 1


# ---- Piso de espaçamento (reamostragem por área) ----------------------------------


def test_plan_keeps_the_axial_pixel_square_on_the_real_ct():
    # Fase de 2 mm da TC do TCIA: a regra antiga dava 256 × 512 (pixel 1,48 × 0,74).
    ni, nj, nk = exam._plan_resample((512, 512, 117), (0.74, 0.74, 2.0), exam.MAX_VOXELS)
    assert (ni, nj, nk) == (369, 369, 117)
    assert ni * nj * nk <= exam.MAX_VOXELS


def test_plan_on_a_thin_ct_tends_to_isotropic():
    ni, nj, nk = exam._plan_resample((512, 512, 800), (0.7, 0.7, 0.625), exam.MAX_VOXELS)
    sp = (512 * 0.7 / ni, 512 * 0.7 / nj, 800 * 0.625 / nk)
    assert ni == nj
    assert max(sp) / min(sp) < 1.02  # voxel ~cúbico
    assert ni * nj * nk <= exam.MAX_VOXELS


def test_plan_never_touches_coarser_axes():
    ni, nj, nk = exam._plan_resample((512, 512, 40), (0.7, 0.7, 5.0), 4_000_000)
    assert nk == 40 and ni == nj < 512


def test_non_integer_resample_keeps_square_pixels_mean_and_geometry(monkeypatch):
    ni, nj, nk = 20, 20, 10
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    data = (100 + 10 * i).astype(np.int16)  # rampa em i
    dirs = np.diag([1.0, 1.0, 2.0])
    origin = np.array([5.0, -3.0, 7.0])
    src = _nrrd_bytes(data, space="left-posterior-superior",
                      **{"space directions": dirs, "space origin": origin})
    monkeypatch.setattr(exam, "MAX_VOXELS", 2000)
    out, stats = normalize_exam([("ramp.nrrd", src)])
    header, back = read_out(out)
    f = 20 / 14
    assert stats.shape == (14, 14, 10)
    assert stats.downsample == (round(f, 4), round(f, 4), 1.0)
    assert np.allclose(header["space directions"], np.diag([f, f, 2.0]))
    assert np.allclose(header["space origin"], origin + [(f - 1) / 2, (f - 1) / 2, 0.0])
    # Média de área de uma rampa = a rampa no centro da amostra nova.
    centers = np.arange(14) * f + (f - 1) / 2
    assert np.allclose(back[0, 0, :], np.rint(100 + 10 * centers), atol=1)
    # A média do volume se conserva.
    assert abs(back.mean() - data.mean()) < 0.5
