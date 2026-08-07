"""FastAPI app — orchestrates multi-STL upload, mesh processing and Sketchfab publishing.

Reads env, validates on startup, exposes 3 endpoints, handles CORS.
All mesh work is delegated to processor.py; all Sketchfab work to sketchfab.py.
This module is pure orchestration.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import unicodedata
import zipfile
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from processor import DEFAULT_TARGET_TRIANGLES, process_obj_bundle, process_stls
from r2 import R2Error, upload_glb
from sketchfab import SketchfabError, get_status, upload_model

# Accepted texture extensions inside an OBJ bundle. Most photogrammetry exports
# use JPG (smaller); PNG is here because Three.js MTLLoader-style tools sometimes
# emit it. Anything else is treated as junk (e.g. a stray .DS_Store inside a zip).
_TEXTURE_EXTS = {".jpg", ".jpeg", ".png"}


def _auto_sketchfab_name() -> str:
    """YYYY-MM-DD-<12 random digits>. Placeholder id until the DB layer lands."""
    return f"{date.today().isoformat()}-{secrets.randbelow(10**12):012d}"

MAX_TOTAL_BYTES = 60 * 1024 * 1024  # 60 MB across all files in one request

# Teto defensivo de divisões (dentro/fora) por caso — um caso clínico real tem
# poucas; dezenas indicam configuração errada (e a operação tem custo de CPU).
MAX_BOOLEAN_OPS = 20


def _parse_boolean_ops(raw: str, filenames: list[str]) -> list[tuple[int, int]]:
    """Valida o form field `boolean_ops` → lista de (idx referência, idx a dividir).

    O campo é um JSON `[{"principal": "<filename>", "secondary": "<filename>"}]`
    com os filenames originais do mesmo request (nomes de campo são contrato de
    API; na UI eles aparecem como "referência" e "a dividir"); devolvemos
    índices em `filenames` para o caller mapear aos nomes limpos. Campo vazio →
    sem divisões. Erros são HTTPException 400 com mensagem pt-BR (aparecem
    direto na tela do clínico).
    """
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            400, "Configuração de divisão de estruturas inválida (JSON malformado)."
        )
    if not isinstance(data, list):
        raise HTTPException(
            400, "Configuração de divisão de estruturas inválida (esperada uma lista)."
        )
    if len(data) > MAX_BOOLEAN_OPS:
        raise HTTPException(
            400, f"No máximo {MAX_BOOLEAN_OPS} divisões de estruturas por caso."
        )

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in data:
        if not isinstance(item, dict):
            raise HTTPException(
                400,
                "Cada divisão precisa dos campos 'principal' e 'secondary'.",
            )
        indices = []
        for key in ("principal", "secondary"):
            value = item.get(key)
            if value not in filenames:
                raise HTTPException(
                    400,
                    f"A divisão referencia um arquivo não enviado: {value!r}.",
                )
            indices.append(filenames.index(value))
        principal_idx, secondary_idx = indices
        if principal_idx == secondary_idx:
            raise HTTPException(
                400, "Uma divisão precisa de duas estruturas diferentes."
            )
        if (principal_idx, secondary_idx) in seen:
            raise HTTPException(400, "Divisão repetida: remova a repetição.")
        seen.add((principal_idx, secondary_idx))
        pairs.append((principal_idx, secondary_idx))
    return pairs

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

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "clinical-3d")

if not DRY_RUN:
    _required = {
        "SKETCHFAB_TOKEN": SKETCHFAB_TOKEN,
        "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
    }
    _missing = [name for name, value in _required.items() if not value]
    if _missing:
        raise RuntimeError(
            f"Variáveis obrigatórias ausentes em produção: {', '.join(_missing)}. "
            "Configure-as ou rode com DRY_RUN=true para desenvolvimento."
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


def _ext(name: str | None) -> str:
    return os.path.splitext((name or "").lower())[1]


def _extract_obj_bundle(
    file_pairs: list[tuple[str, bytes]],
) -> tuple[bytes, bytes | None, dict[str, bytes], str]:
    """Pull (obj, mtl_or_None, textures, obj_name) out of a list of uploaded files.

    Accepts either a single .zip containing the bundle or the loose files
    themselves. Requires exactly one OBJ; the MTL and texture images are
    optional (an OBJ without a material falls through to keyword colouring in
    processor.process_obj_bundle). Raises HTTPException with pt-BR messages on
    any mismatch.
    """
    # If a zip was uploaded, expand it in-memory and recurse with the contents.
    # We unwrap at most one level — a zip-of-zips is weird and rejected.
    zip_pairs = [(n, b) for n, b in file_pairs if _ext(n) == ".zip"]
    if zip_pairs:
        if len(zip_pairs) > 1 or len(file_pairs) > 1:
            raise HTTPException(
                400,
                "Envie um único arquivo .zip ou os arquivos do modelo soltos — não os dois.",
            )
        try:
            with zipfile.ZipFile(io.BytesIO(zip_pairs[0][1])) as zf:
                file_pairs = [
                    (os.path.basename(zi.filename), zf.read(zi))
                    for zi in zf.infolist()
                    if not zi.is_dir() and not zi.filename.startswith("__MACOSX/")
                    and os.path.basename(zi.filename)
                ]
        except zipfile.BadZipFile:
            raise HTTPException(400, "Arquivo .zip inválido ou corrompido.")

    objs = [(n, b) for n, b in file_pairs if _ext(n) == ".obj"]
    mtls = [(n, b) for n, b in file_pairs if _ext(n) == ".mtl"]
    textures = {n: b for n, b in file_pairs if _ext(n) in _TEXTURE_EXTS}

    if len(objs) != 1:
        raise HTTPException(
            400,
            f"Bundle OBJ deve conter exatamente um arquivo .obj (encontrados: {len(objs)}).",
        )
    if len(mtls) > 1:
        raise HTTPException(
            400,
            f"Bundle OBJ deve conter no máximo um arquivo .mtl (encontrados: {len(mtls)}).",
        )

    mtl_bytes = mtls[0][1] if mtls else None
    return objs[0][1], mtl_bytes, textures, objs[0][0]


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    target_triangles: int = Form(default=DEFAULT_TARGET_TRIANGLES),
    boolean_ops: str = Form(default=""),
) -> dict:
    if not files:
        raise HTTPException(400, "Nenhum arquivo enviado.")

    file_pairs: list[tuple[str, bytes]] = []
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
        file_pairs.append((f.filename or "", contents))

    # Sniff input shape: all STL vs OBJ bundle (zip or loose). Reject mixed —
    # the colour-by-keyword path (STL) and the preserve-texture path (OBJ) are
    # fundamentally different and combining them produces a confusing result.
    exts = {_ext(n) for n, _ in file_pairs}
    is_obj_bundle = ".obj" in exts or ".zip" in exts
    is_stl_only = exts == {".stl"}

    if is_obj_bundle and ".stl" in exts:
        raise HTTPException(
            400,
            "Envie apenas arquivos STL ou um único bundle OBJ — não misturados.",
        )

    ops_idx = _parse_boolean_ops(boolean_ops, [n for n, _ in file_pairs])
    if ops_idx and not is_stl_only:
        raise HTTPException(
            400,
            "A divisão de estruturas está disponível apenas para envios de arquivos STL.",
        )

    if is_obj_bundle:
        obj_bytes, mtl_bytes, textures, obj_filename = _extract_obj_bundle(file_pairs)
        mesh_name = clean_mesh_names([obj_filename])[0]
        try:
            glb_bytes, stats = process_obj_bundle(
                obj_bytes, mtl_bytes, textures, mesh_name
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
    elif is_stl_only:
        mesh_names = clean_mesh_names([n for n, _ in file_pairs])
        stls = list(zip(mesh_names, [b for _, b in file_pairs]))
        # As ops chegam por filename original; process_stls fala nomes limpos.
        ops_names = [(mesh_names[p], mesh_names[s]) for p, s in ops_idx]
        try:
            glb_bytes, stats = process_stls(
                stls,
                target_triangles_per_mesh=target_triangles,
                boolean_ops=ops_names,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
    else:
        raise HTTPException(
            400,
            "Tipo de arquivo não reconhecido. Envie arquivos .stl ou um bundle OBJ "
            "(.obj + .mtl + imagem, soltos ou em .zip).",
        )

    sketchfab_name = _auto_sketchfab_name()

    try:
        uid = upload_model(glb_bytes, name=sketchfab_name, token=SKETCHFAB_TOKEN)
    except SketchfabError as e:
        raise HTTPException(502, str(e))

    # Best-effort parallel push to R2. Sketchfab is the source of truth in
    # Sprint 2; R2 is a backup that Sprint 3 will migrate the viewer to.
    # Failures here log loudly but do not break the request — the clinician
    # already has a working viewer link via Sketchfab.
    try:
        upload_glb(
            glb_bytes,
            uid=uid,
            bucket=R2_BUCKET,
            account_id=R2_ACCOUNT_ID,
            access_key=R2_ACCESS_KEY_ID,
            secret_key=R2_SECRET_ACCESS_KEY,
        )
        print(f"[r2] uploaded cases/{uid}.glb")
    except R2Error as e:
        print(f"[r2 ERROR] failed cases/{uid}.glb: {e}")

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
                    "color": m.color or None,
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
