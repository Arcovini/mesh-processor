"""FastAPI app — orchestrates multi-STL upload, mesh processing and Sketchfab publishing.

Reads env, validates on startup, exposes 3 endpoints, handles CORS.
All mesh work is delegated to processor.py; all Sketchfab work to sketchfab.py.
This module is pure orchestration.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from processor import DEFAULT_TARGET_TRIANGLES, process_stls
from sketchfab import SketchfabError, get_status, upload_model

MAX_TOTAL_BYTES = 60 * 1024 * 1024  # 60 MB across all files in one request

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
    name: str = Form(default=""),
    target_triangles: int = Form(default=DEFAULT_TARGET_TRIANGLES),
) -> dict:
    if not files:
        raise HTTPException(400, "Nenhum arquivo enviado.")

    stls: list[tuple[str, bytes]] = []
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
        mesh_name = (f.filename or "mesh").rsplit(".", 1)[0]
        stls.append((mesh_name, contents))

    try:
        glb_bytes, stats = process_stls(stls, target_triangles_per_mesh=target_triangles)
    except ValueError as e:
        raise HTTPException(400, str(e))

    model_name = name.strip() or f"Caso ({len(files)} estruturas)"

    try:
        uid = upload_model(glb_bytes, name=model_name, token=SKETCHFAB_TOKEN)
    except SketchfabError as e:
        raise HTTPException(502, str(e))

    return {
        "uid": uid,
        "viewer_url": f"{VIEWER_BASE}?id={uid}",
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
