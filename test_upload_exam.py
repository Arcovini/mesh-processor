"""Testes do campo `exam` no POST /upload, ponta a ponta em DRY_RUN.

DRY_RUN_GLB_DIR aponta para uma pasta temporária: o que o endpoint "sobe" para
o R2 aparece em <tmp>/cases/{uid}.glb e .exam-{n}.nrrd/.json, exatamente como
no dev local. Também cobre POST /cases/{uid}/exam (séries 1..3).
Também cobre a normalização de STL marcado como RAS no processor.
Run with: .venv/bin/python -m pytest test_upload_exam.py -q
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import trimesh

os.environ.setdefault("DRY_RUN", "true")  # main.py exige credenciais no import sem isso

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from processor import _load_and_decimate  # noqa: E402
from test_exam import series  # noqa: E402

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def dev_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN_GLB_DIR", str(tmp_path))
    return tmp_path


def _stl(radius=10.0, header=b"3D Slicer output. SPACE=LPS") -> bytes:
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.apply_translation((30.0, -20.0, 10.0))
    data = bytearray(m.export(file_type="stl"))
    data[:80] = header.ljust(80, b" ")
    return bytes(data)


def _dicom_files(n=10, **kw):
    return [("exam", (name, data, "application/dicom")) for name, data in series(n, **kw)]


def _meta(dev_dir, uid, n):
    return json.loads((dev_dir / "cases" / f"{uid}.exam-{n}.json").read_text("utf-8"))


def test_exam_only_case_stores_the_nrrd(dev_dir):
    r = client.post("/upload", files=_dicom_files())
    assert r.status_code == 200, r.text
    body = r.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["uid"])
    assert body["processing"] is False
    assert body["stats"] is None
    assert body["exam"]["stored"] is True
    assert body["exam"]["shape"] == [5, 6, 10]
    assert body["exam"]["source"] == "dicom"
    assert body["exam"]["index"] == 0
    assert body["exam"]["label"] == "Axial"
    assert re.fullmatch(r"[0-9a-f]{64}", body["write_token"])
    uid = body["uid"]
    assert (dev_dir / "cases" / f"{uid}.exam-0.nrrd").read_bytes()[:7] == b"NRRD000"
    assert _meta(dev_dir, uid, 0) == {
        "version": 1,
        "label": "Axial",
        "images": 10,
        "shape": [5, 6, 10],
        "spacing": body["exam"]["spacing"],
        "bytes": (dev_dir / "cases" / f"{uid}.exam-0.nrrd").stat().st_size,
    }
    assert not (dev_dir / "cases" / f"{uid}.glb").exists()


def test_series_without_description_gets_a_fallback_label(dev_dir):
    r = client.post("/upload", files=_dicom_files(description=""))
    assert r.status_code == 200, r.text
    assert r.json()["exam"]["label"] == "Série 1"
    assert _meta(dev_dir, r.json()["uid"], 0)["label"] == "Série 1"


def test_model_and_exam_share_the_same_uid(dev_dir):
    files = [("files", ("Rim.stl", _stl(), "model/stl"))] + _dicom_files()
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    uid = body["uid"]
    assert body["processing"] is False
    assert body["stats"]["meshes"][0]["name"] == "Rim"
    assert (dev_dir / "cases" / f"{uid}.glb").exists()
    assert (dev_dir / "cases" / f"{uid}.exam-0.nrrd").exists()
    assert (dev_dir / "cases" / f"{uid}.exam-0.json").exists()


def test_model_only_case_goes_to_r2_under_our_own_uid(dev_dir):
    r = client.post("/upload", files=[("files", ("Rim.stl", _stl(), "model/stl"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["uid"])
    assert (dev_dir / "cases" / f"{body['uid']}.glb").read_bytes()[:4] == b"glTF"
    assert body["viewer_url"].endswith(f"?id={body['uid']}")
    assert body["exam"] is None
    assert body["write_token"] is None
    # Nada a esperar depois da resposta (a página de antes do 3c lê o campo).
    assert body["processing"] is False
    assert "sketchfab_name" not in body


def test_status_endpoint_is_gone():
    # Só servia para esperar o processamento do Sketchfab.
    assert client.get("/status/272a33d42c0a49949a21b6e79169606e").status_code == 404


def test_bad_exam_fails_before_any_upload(dev_dir, monkeypatch):
    called = []
    monkeypatch.setattr(main, "upload_glb", lambda *a, **k: called.append(1))
    items = series(10)
    del items[4]  # falta uma imagem
    files = [("files", ("Rim.stl", _stl(), "model/stl"))] + [
        ("exam", (n, d, "application/dicom")) for n, d in items
    ]
    r = client.post("/upload", files=files)
    assert r.status_code == 400
    assert "faltam imagens" in r.json()["detail"]
    assert called == []
    assert not (dev_dir / "cases").exists()


def test_too_many_loose_files_asks_for_a_zip(monkeypatch):
    monkeypatch.setattr(main, "MAX_EXAM_FILES", 5)
    r = client.post("/upload", files=_dicom_files(10))
    assert r.status_code == 400
    assert ".zip" in r.json()["detail"]


def test_exam_over_the_size_cap_is_refused(monkeypatch):
    monkeypatch.setattr(main, "MAX_EXAM_BYTES", 1000)
    r = client.post("/upload", files=_dicom_files(10))
    assert r.status_code == 413


def test_nothing_sent_is_refused():
    r = client.post("/upload", data={"boolean_ops": ""})
    assert r.status_code == 400


def _r2_down(*a, **k):
    raise main.R2Error("sem rede")


def test_r2_failure_on_exam_with_model_is_best_effort(monkeypatch):
    monkeypatch.setattr(main, "upload_exam", _r2_down)
    files = [("files", ("Rim.stl", _stl(), "model/stl"))] + _dicom_files()
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    assert r.json()["exam"]["stored"] is False
    assert r.json()["exam"]["error"]
    # Sem a série 0 gravada não há o que completar.
    assert r.json()["write_token"] is None


def test_r2_failure_on_the_model_fails_the_request(dev_dir, monkeypatch):
    # O R2 é o único lugar do modelo desde o Sprint 3c: sem o GLB gravado o
    # link abriria "caso não encontrado". A série 0 nem chega a ser gravada.
    monkeypatch.setattr(main, "upload_glb", _r2_down)
    files = [("files", ("Rim.stl", _stl(), "model/stl"))] + _dicom_files()
    r = client.post("/upload", files=files)
    assert r.status_code == 502
    assert "modelo" in r.json()["detail"]
    assert not (dev_dir / "cases").exists()


def test_r2_failure_on_exam_only_case_fails_the_request(monkeypatch):
    # Sem modelo, o exame é o caso: devolver um link que abre "caso não
    # encontrado" seria pior que pedir para enviar de novo.
    monkeypatch.setattr(main, "upload_exam", _r2_down)
    r = client.post("/upload", files=_dicom_files())
    assert r.status_code == 502


def test_bad_model_is_refused_before_normalizing_the_exam(monkeypatch):
    called = []
    monkeypatch.setattr(main, "normalize_exam", lambda items: called.append(1))
    files = [("files", ("Vazio.stl", b"", "model/stl"))] + _dicom_files()
    r = client.post("/upload", files=files)
    assert r.status_code == 400
    assert called == []


def test_json_is_written_only_after_the_nrrd(monkeypatch):
    order = []
    monkeypatch.setattr(main, "upload_exam", lambda *a, **k: order.append("nrrd"))
    monkeypatch.setattr(main, "upload_exam_meta", lambda *a, **k: order.append("json"))
    r = client.post("/upload", files=_dicom_files())
    assert r.status_code == 200, r.text
    assert order == ["nrrd", "json"]


def test_nrrd_without_json_does_not_count_as_stored(dev_dir, monkeypatch):
    # O JSON é o que faz a série existir: falhar nele é falhar a série.
    monkeypatch.setattr(main, "upload_exam_meta", _r2_down)
    r = client.post("/upload", files=_dicom_files())
    assert r.status_code == 502


# ---- POST /cases/{uid}/exam (séries 1..3) --------------------------------------


def _new_case(dev_dir, **kw):
    r = client.post("/upload", files=_dicom_files(**kw))
    assert r.status_code == 200, r.text
    return r.json()["uid"], r.json()["write_token"]


def _add(uid, token, index, files):
    return client.post(
        f"/cases/{uid}/exam",
        data={"write_token": token, "index": str(index)},
        files=files,
    )


def test_extra_series_is_stored_next_to_the_first(dev_dir):
    uid, token = _new_case(dev_dir, description="ARTERIAL 1.0 B30f")
    r = _add(uid, token, 1, _dicom_files(12, description="NEFROGRAFICA 1.0 B30f"))
    assert r.status_code == 200, r.text
    assert r.json()["exam"]["index"] == 1
    assert r.json()["exam"]["label"] == "NEFROGRAFICA 1.0 B30f"
    assert _meta(dev_dir, uid, 0)["label"] == "ARTERIAL 1.0 B30f"
    assert _meta(dev_dir, uid, 1)["label"] == "NEFROGRAFICA 1.0 B30f"
    assert _meta(dev_dir, uid, 1)["images"] == 12
    assert (dev_dir / "cases" / f"{uid}.exam-1.nrrd").read_bytes()[:7] == b"NRRD000"


def test_retrying_the_same_index_overwrites(dev_dir):
    uid, token = _new_case(dev_dir)
    assert _add(uid, token, 2, _dicom_files(10, description="primeira")).status_code == 200
    assert _add(uid, token, 2, _dicom_files(10, description="segunda")).status_code == 200
    assert _meta(dev_dir, uid, 2)["label"] == "segunda"


def test_wrong_token_is_refused(dev_dir):
    uid, token = _new_case(dev_dir)
    other = "a" * 32
    assert _add(uid, "0" * 64, 1, _dicom_files()).status_code == 403
    # O token de um caso não serve para outro.
    assert _add(other, token, 1, _dicom_files()).status_code == 403
    assert not (dev_dir / "cases" / f"{uid}.exam-1.json").exists()


def test_non_ascii_token_is_403_not_500(dev_dir):
    uid, _ = _new_case(dev_dir)
    assert _add(uid, "é" * 64, 1, _dicom_files()).status_code == 403


def test_malformed_uid_is_refused_even_with_its_own_token():
    uid = "../fora"
    assert _add(uid, main._write_token(uid), 1, _dicom_files()).status_code in (403, 404)


@pytest.mark.parametrize("index", [0, 4, -1])
def test_index_outside_1_to_3_is_refused(dev_dir, index):
    uid, token = _new_case(dev_dir)
    r = _add(uid, token, index, _dicom_files())
    assert r.status_code == 400
    assert "4 séries" in r.json()["detail"]


def test_invalid_extra_series_gets_the_same_message_as_upload(dev_dir):
    uid, token = _new_case(dev_dir)
    items = series(10)
    del items[4]
    r = _add(uid, token, 1, [("exam", (n, d, "application/dicom")) for n, d in items])
    assert r.status_code == 400
    assert "faltam imagens" in r.json()["detail"]


def test_r2_failure_on_extra_series_is_502(dev_dir, monkeypatch):
    uid, token = _new_case(dev_dir)
    monkeypatch.setattr(main, "upload_exam", _r2_down)
    r = _add(uid, token, 1, _dicom_files())
    assert r.status_code == 502
    assert "Tente enviar de novo" in r.json()["detail"]


def test_production_boot_requires_the_write_secret(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "false",
        "R2_ACCOUNT_ID": "x",
        "R2_ACCESS_KEY_ID": "x",
        "R2_SECRET_ACCESS_KEY": "x",
        "EXAM_WRITE_SECRET": "",
    }
    r = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "EXAM_WRITE_SECRET" in r.stderr


def test_production_boots_without_a_sketchfab_token():
    env = {
        **os.environ,
        "DRY_RUN": "false",
        "R2_ACCOUNT_ID": "x",
        "R2_ACCESS_KEY_ID": "x",
        "R2_SECRET_ACCESS_KEY": "x",
        "EXAM_WRITE_SECRET": "x",
    }
    env.pop("SKETCHFAB_TOKEN", None)
    r = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


# ---- STL marcado como RAS ----------------------------------------------------------


def test_stl_marked_ras_is_brought_to_lps():
    lps, _, _ = _load_and_decimate(_stl(header=b"3D Slicer output. SPACE=LPS"), 300_000)
    ras, _, _ = _load_and_decimate(_stl(header=b"3D Slicer output. SPACE=RAS"), 300_000)
    # Mesma malha, x e y com sinal trocado.
    assert np.allclose(ras.centroid, lps.centroid * np.array([-1, -1, 1]), atol=1e-6)
    assert ras.volume == pytest.approx(lps.volume)  # rotação, não espelhamento


def test_stl_without_space_mark_is_taken_as_lps():
    plain, _, _ = _load_and_decimate(_stl(header=b"exported by another tool"), 300_000)
    assert np.allclose(plain.centroid, [30.0, -20.0, 10.0], atol=0.5)
