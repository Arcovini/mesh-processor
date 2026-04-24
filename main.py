"""FastAPI app — orchestrates multi-STL upload, mesh processing and Sketchfab publishing.

Reads env, validates on startup, exposes 3 endpoints, handles CORS.
All mesh work is delegated to processor.py; all Sketchfab work to sketchfab.py.
This module is pure orchestration.
"""
from __future__ import annotations

import os
import re
import secrets
import unicodedata
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from processor import DEFAULT_TARGET_TRIANGLES, process_stls
from sketchfab import SketchfabError, get_status, upload_model


def _auto_sketchfab_name() -> str:
    """YYYY-MM-DD-<12 random digits>. Placeholder id until the DB layer lands."""
    return f"{date.today().isoformat()}-{secrets.randbelow(10**12):012d}"

MAX_TOTAL_BYTES = 60 * 1024 * 1024  # 60 MB across all files in one request

# Catches "_(timestamp)" tails (with or without closing paren) as a fallback
# when common-prefix/suffix stripping can't catch per-file unique timestamps.
_TIMESTAMP_TAIL = re.compile(r"_\(.*$")


def _longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _longest_common_suffix(strings: list[str]) -> str:
    reversed_prefix = _longest_common_prefix([s[::-1] for s in strings])
    return reversed_prefix[::-1]


def _transliterate(s: str) -> str:
    """ASCII-ize — Sketchfab/glTF node names with accents (ã, ç) render as mojibake."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def clean_mesh_names(filenames: list[str | None]) -> list[str]:
    """Clean mesh names by stripping what repeats across the batch + timestamp tails."""
    bases = [(f or "mesh").rsplit(".", 1)[0] for f in filenames]

    if len(bases) >= 2:
        prefix = _longest_common_prefix(bases)
        suffix = _longest_common_suffix(bases)
        end = (lambda b: len(b) - len(suffix)) if suffix else (lambda b: len(b))
        stripped = [b[len(prefix): end(b)] for b in bases]
        # Only apply if no name gets emptied out — otherwise revert the batch
        if all(s.strip(" _") for s in stripped):
            bases = stripped

    cleaned = []
    for b in bases:
        b = _TIMESTAMP_TAIL.sub("", b)
        if b.startswith("STL"):  # single-file fallback; multi-file prefix would already catch this
            b = b[3:]
        b = _transliterate(b).strip(" _")
        cleaned.append(b or "mesh")
    return cleaned

DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes")
SKETCHFAB_TOKEN = os.getenv("SKETCHFAB_TOKEN", "")
VIEWER_BASE = os.getenv("VIEWER_BASE", "https://biodesignlab.com.br/case/")

if not DRY_RUN and not SKETCHFAB_TOKEN:
    raise RuntimeError(
        "SKETCHFAB_TOKEN é obrigatório quando DRY_RUN não está ativo. "
        "Exporte o token ou rode com DRY_RUN=true para desenvolvimento."
    )

app = FastAPI(title="mesh-processor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://biodesignlab.com.br",
        # VSCode "Live Server" (default port 5500), plus common variants.
        # 127.0.0.1 and localhost are distinct origins to the browser.
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "dry_run": DRY_RUN}


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    target_triangles: int = Form(default=DEFAULT_TARGET_TRIANGLES),
) -> dict:
    if not files:
        raise HTTPException(400, "Nenhum arquivo enviado.")

    payloads: list[bytes] = []
    total_size = 0
    for f in files:
        contents = await f.read()
        if len(contents) == 0:
            raise HTTPException(400, f"Arquivo vazio: {f.filename or '(sem nome)'}.")
        total_size += len(contents)
        if total_size > MAX_TOTAL_BYTES:
            raise HTTPException(
                413,
                f"Soma dos arquivos ultrapassa {MAX_TOTAL_BYTES // (1024 * 1024)}MB.",
            )
        payloads.append(contents)

    mesh_names = clean_mesh_names([f.filename for f in files])
    stls = list(zip(mesh_names, payloads))

    try:
        glb_bytes, stats = process_stls(stls, target_triangles_per_mesh=target_triangles)
    except ValueError as e:
        raise HTTPException(400, str(e))

    sketchfab_name = _auto_sketchfab_name()

    try:
        uid = upload_model(glb_bytes, name=sketchfab_name, token=SKETCHFAB_TOKEN)
    except SketchfabError as e:
        raise HTTPException(502, str(e))

    return {
        "uid": uid,
        "viewer_url": f"{VIEWER_BASE}?id={uid}",
        "sketchfab_name": sketchfab_name,
        "stats": {
            "total_input_triangles": stats.total_input_triangles,
            "total_output_triangles": stats.total_output_triangles,
            "glb_size_mb": round(stats.glb_size_bytes / (1024 * 1024), 2),
            "meshes": [
                {
                    "name": m.name,
                    "input_triangles": m.input_triangles,
                    "output_triangles": m.output_triangles,
                    "decimated": m.decimated,
                    "color": m.color,
                }
                for m in stats.meshes
            ],
        },
        "processing": True,
    }


@app.get("/status/{uid}")
def status(uid: str) -> dict:
    try:
        return get_status(uid, token=SKETCHFAB_TOKEN)
    except SketchfabError as e:
        raise HTTPException(502, str(e))
