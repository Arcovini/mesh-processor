"""Thin Sketchfab Data API v3 client.

Knows nothing about meshes; only HTTP. Takes bytes + metadata, returns uid / status.

DRY_RUN=true short-circuits the real API. Essential on the Basic (free) plan,
which caps uploads at 10/month — we don't want to burn slots on dev/CI traffic.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import requests

API_BASE = "https://api.sketchfab.com/v3"
UPLOAD_TIMEOUT_S = 60
STATUS_TIMEOUT_S = 10

# Known-working UID from medCaseViewer/case/main.js fallback — used as the
# default fake uid in DRY_RUN so the resulting viewer_url actually loads.
DEFAULT_DRY_RUN_UID = "272a33d42c0a49949a21b6e79169606e"


class SketchfabError(RuntimeError):
    pass


def _dry_run() -> bool:
    return os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes")


def _dry_run_uid() -> str:
    # Explicit override wins; otherwise the known-good fallback; random only if
    # the caller set DRY_RUN_UID=random (useful to verify "no viewer" paths).
    override = os.getenv("DRY_RUN_UID")
    if override == "random":
        return uuid.uuid4().hex
    return override or DEFAULT_DRY_RUN_UID


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Token {token}"}


def upload_model(glb_bytes: bytes, name: str, token: str) -> str:
    """POST the GLB to Sketchfab. Returns the uid of the created model."""
    if _dry_run():
        uid = _dry_run_uid()
        print(f"[sketchfab DRY_RUN] upload '{name}' ({len(glb_bytes)} bytes) -> uid={uid}")
        return uid

    resp = requests.post(
        f"{API_BASE}/models",
        headers=_auth(token),
        files={"modelFile": (f"{name}.glb", glb_bytes, "model/gltf-binary")},
        data={
            "name": name,
            "source": "biodesignlab",
            "isPublished": "true",
            "private": "false",
            # Downloadable models don't count against the Basic plan's 10/month cap.
            # See https://sketchfab.com/plans — "A model that is downloadable doesn't count against this limit."
            "isDownloadable": "true",
        },
        timeout=UPLOAD_TIMEOUT_S,
    )
    if resp.status_code != 201:
        raise SketchfabError(f"Upload falhou ({resp.status_code}): {resp.text[:500]}")
    return resp.json()["uid"]


def get_status(uid: str, token: str) -> dict[str, Any]:
    """Return {'ready': bool, 'error': str | None} for the given model uid."""
    if _dry_run():
        return {"ready": True, "error": None}

    resp = requests.get(
        f"{API_BASE}/models/{uid}",
        headers=_auth(token),
        timeout=STATUS_TIMEOUT_S,
    )
    if resp.status_code == 404:
        return {"ready": False, "error": "Modelo não encontrado no Sketchfab"}
    if resp.status_code != 200:
        raise SketchfabError(f"Status falhou ({resp.status_code}): {resp.text[:500]}")

    processing = resp.json().get("status", {}).get("processing", "")
    if processing == "SUCCEEDED":
        return {"ready": True, "error": None}
    if processing == "FAILED":
        return {"ready": False, "error": "Falha no processamento no Sketchfab"}
    return {"ready": False, "error": None}
