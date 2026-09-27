"""FastAPI app — orchestrates multi-STL upload, mesh processing and storage.

Reads env, validates on startup, exposes 3 endpoints, handles CORS.
All mesh work is delegated to processor.py; the image exam (DICOM/NRRD) to
exam.py; storage to r2.py (the only destination since Sprint 3c).
This module is pure orchestration.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import secrets
import unicodedata
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from exam import ExamStats, normalize_exam
from processor import (
    DEFAULT_TARGET_TRIANGLES,
    ProcessStats,
    process_obj_bundle,
    process_stls,
)
from r2 import R2Error, upload_exam, upload_exam_meta, upload_glb

# Accepted texture extensions inside an OBJ bundle. Most photogrammetry exports
# use JPG (smaller); PNG is here because Three.js MTLLoader-style tools sometimes
# emit it. Anything else is treated as junk (e.g. a stray .DS_Store inside a zip).
_TEXTURE_EXTS = {".jpg", ".jpeg", ".png"}


MAX_TOTAL_BYTES = 60 * 1024 * 1024  # 60 MB across all mesh files in one request

# Exame de imagem (campo `exam`): teto separado do dos modelos. 200 MB cabem nos
# 5 min que o Railway dá para o corpo do request chegar numa conexão de ~8 Mbps.
MAX_EXAM_BYTES = 200 * 1024 * 1024
# O parser multipart do Starlette recusa mais de 1000 arquivos com uma mensagem
# genérica em inglês; abaixo disso pedimos o .zip em português.
MAX_EXAM_FILES = 900
# Séries por caso (sem contraste, arterial, portal, tardia). Cada uma chega num
# request próprio: 4 fases de TC fina (~1 GB) não cabem num corpo só nos 5 min
# do Railway. A série 0 vem no /upload; as outras em POST /cases/{uid}/exam.
MAX_EXAM_SERIES = 4

_UID_RE = re.compile(r"[0-9a-f]{32}")

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
    """ASCII-ize — glTF node names with accents (ã, ç) rendered as mojibake on Sketchfab."""
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
VIEWER_BASE = os.getenv("VIEWER_BASE", "https://biodesignlab.com.br/case/")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "clinical-3d")

# Assina o write_token das séries extras do exame (HMAC do uid). Sem estado e
# sem banco: quem recebeu o token no /upload pode acrescentar séries àquele
# caso; quem só tem o link do visualizador, não. Trocar o segredo invalida os
# tokens em uso — só afeta uploads em andamento, porque a página não os guarda.
EXAM_WRITE_SECRET = os.getenv("EXAM_WRITE_SECRET", "") or (
    "dev-only-exam-write-secret" if DRY_RUN else ""
)

if not DRY_RUN:
    _required = {
        "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "EXAM_WRITE_SECRET": EXAM_WRITE_SECRET,
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
        # Porta do servidor estático que o Playwright do medCaseViewer levanta
        # (playwright.config.js), para os specs de upload rodarem contra um
        # backend local em DRY_RUN.
        "http://127.0.0.1:5505",
        "http://localhost:5505",
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


def _read_mesh_files(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Lê os arquivos do modelo e aplica as checagens baratas (vazio, 60 MB).

    Vem antes do exame: um STL vazio ou grande demais é recusado na hora, e não
    depois de dezenas de segundos normalizando uma série DICOM.
    """
    file_pairs: list[tuple[str, bytes]] = []
    total_size = 0
    for f in files:
        contents = f.file.read()
        if len(contents) == 0:
            raise HTTPException(400, f"Arquivo vazio: {f.filename or '(sem nome)'}.")
        total_size += len(contents)
        if total_size > MAX_TOTAL_BYTES:
            raise HTTPException(
                413,
                f"Soma dos arquivos ultrapassa {MAX_TOTAL_BYTES // (1024 * 1024)}MB.",
            )
        file_pairs.append((f.filename or "", contents))
    return file_pairs


def _process_meshes(
    file_pairs: list[tuple[str, bytes]], target_triangles: int, boolean_ops: str
) -> tuple[bytes, ProcessStats]:
    """STLs ou bundle OBJ → GLB. Erros viram HTTPException 400 em pt-BR."""
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

    try:
        if is_obj_bundle:
            obj_bytes, mtl_bytes, textures, obj_filename = _extract_obj_bundle(file_pairs)
            mesh_name = clean_mesh_names([obj_filename])[0]
            return process_obj_bundle(obj_bytes, mtl_bytes, textures, mesh_name)
        if is_stl_only:
            mesh_names = clean_mesh_names([n for n, _ in file_pairs])
            stls = list(zip(mesh_names, [b for _, b in file_pairs]))
            # As ops chegam por filename original; process_stls fala nomes limpos.
            ops_names = [(mesh_names[p], mesh_names[s]) for p, s in ops_idx]
            return process_stls(
                stls,
                target_triangles_per_mesh=target_triangles,
                boolean_ops=ops_names,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))

    raise HTTPException(
        400,
        "Tipo de arquivo não reconhecido. Envie arquivos .stl ou um bundle OBJ "
        "(.obj + .mtl + imagem, soltos ou em .zip).",
    )


def _normalize_exam(exam: list[UploadFile]) -> tuple[bytes, ExamStats]:
    """Campo `exam` → NRRD canônico (exam.py). Erro de entrada → 400/413.

    Roda ANTES de qualquer upload: um exame recusado não deixa GLB órfão no R2.
    """
    if len(exam) > MAX_EXAM_FILES:
        raise HTTPException(
            400,
            f"O exame tem {len(exam)} arquivos; o limite para arquivos soltos é "
            f"{MAX_EXAM_FILES}. Envie a série compactada em um único .zip.",
        )
    total = sum(f.size or 0 for f in exam)
    if total > MAX_EXAM_BYTES:
        raise HTTPException(
            413,
            f"O exame soma {total / 1024 / 1024:.0f} MB e ultrapassa o limite de "
            f"{MAX_EXAM_BYTES // (1024 * 1024)} MB.",
        )
    # Gerador: exam.py lê um arquivo de cada vez (o FastAPI já guardou cada
    # upload num arquivo temporário), então o pico de memória é o volume, não
    # a soma dos arquivos.
    items = ((f.filename or "", f.file.read()) for f in exam)
    try:
        return normalize_exam(items)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _write_token(uid: str) -> str:
    return hmac.new(EXAM_WRITE_SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()


def _exam_info(stats: ExamStats, n: int) -> dict:
    """O que a resposta conta sobre uma série (o JSON no R2 é um recorte disto)."""
    return {
        "stored": True,
        "index": n,
        "label": stats.label or f"Série {n + 1}",
        "source": stats.source,
        "shape": list(stats.shape),
        "spacing": list(stats.spacing),
        "downsample": list(stats.downsample),
        "size_mb": round(stats.nrrd_bytes / (1024 * 1024), 2),
        "ignored_files": stats.ignored_files,
        "modality": stats.modality,
    }


def _store_exam_series(nrrd: bytes, info: dict, uid: str, r2: dict) -> None:
    """NRRD e depois o JSON da série. O JSON é o que faz a série existir para o
    visualizador: um NRRD órfão (falha entre os dois puts) fica invisível e é
    sobrescrito na nova tentativa. Levanta R2Error."""
    n = info["index"]
    # Público no R2: do cabeçalho DICOM, só o nome da série (decisão do
    # usuário). Nem a modalidade vai — o visualizador não precisa dela.
    meta = {
        "version": 1,
        "label": info["label"],
        "images": info["shape"][2],
        "shape": info["shape"],
        "spacing": info["spacing"],
        "bytes": len(nrrd),
    }
    upload_exam(nrrd, uid=uid, n=n, **r2)
    upload_exam_meta(json.dumps(meta, ensure_ascii=False).encode("utf-8"), uid=uid, n=n, **r2)
    print(f"[r2] uploaded cases/{uid}.exam-{n}.nrrd + .json")


def _r2() -> dict:
    return dict(
        bucket=R2_BUCKET,
        account_id=R2_ACCOUNT_ID,
        access_key=R2_ACCESS_KEY_ID,
        secret_key=R2_SECRET_ACCESS_KEY,
    )


def _log_exam(stats: ExamStats) -> None:
    print(
        f"[exam] {stats.source} {stats.input_files} arquivo(s) "
        f"(ignorados {stats.ignored_files}) -> {stats.shape} "
        f"{stats.dtype} downsample={stats.downsample} "
        f"{stats.nrrd_bytes / 1024 / 1024:.1f} MB modality={stats.modality} "
        f"label={stats.label!r}"
    )


# `def` (e não `async def`): o FastAPI roda a função num thread pool. O
# processamento é CPU pura (decimação, divisões, decodificar e comprimir o
# exame — dezenas de segundos numa série grande); num `async def` isso travaria
# o event loop e o /health pararia de responder durante o upload.
@app.post("/upload")
def upload(
    files: list[UploadFile] = File(default=[]),
    exam: list[UploadFile] = File(default=[]),
    target_triangles: int = Form(default=DEFAULT_TARGET_TRIANGLES),
    boolean_ops: str = Form(default=""),
) -> dict:
    if not files and not exam:
        raise HTTPException(400, "Nenhum arquivo enviado.")

    # 0) Checagens baratas do modelo antes do trabalho pesado do exame.
    file_pairs = _read_mesh_files(files) if files else []
    if file_pairs:
        _parse_boolean_ops(boolean_ops, [n for n, _ in file_pairs])

    # 1) Exame primeiro: é o que mais pode ser recusado (série misturada,
    #    imagens faltando) e o mais caro de descobrir depois.
    exam_bytes: bytes | None = None
    exam_stats: ExamStats | None = None
    if exam:
        exam_bytes, exam_stats = _normalize_exam(exam)
        _log_exam(exam_stats)

    # 2) Modelo 3D, se houver.
    glb_bytes: bytes | None = None
    stats: ProcessStats | None = None
    if file_pairs:
        glb_bytes, stats = _process_meshes(file_pairs, target_triangles, boolean_ops)

    r2 = _r2()

    # 3) Destino. Desde o Sprint 3c o R2 é o único lugar do modelo (o Sketchfab
    #    só serve, como leitura, os casos antigos): o uid é nosso, e sem o GLB
    #    gravado o link abriria "caso não encontrado" — então o request falha.
    #    O visualizador acha o exame pelo cases/{uid}.exam-0.json.
    uid = secrets.token_hex(16)
    if glb_bytes is not None:
        try:
            upload_glb(glb_bytes, uid=uid, **r2)
            print(f"[r2] uploaded cases/{uid}.glb")
        except R2Error as e:
            print(f"[r2 ERROR] failed cases/{uid}.glb: {e}")
            raise HTTPException(
                502, "Não foi possível guardar o modelo agora. Tente enviar de novo."
            )

    # A série do /upload é sempre a 0: a que as estruturas usaram (a página
    # manda primeiro a que o clínico marcou como "usada na segmentação").
    exam_info: dict | None = None
    write_token: str | None = None
    if exam_bytes is not None and exam_stats is not None:
        exam_info = _exam_info(exam_stats, 0)
        # Falha de infraestrutura no exame. Com modelo: best-effort —
        # o link do caso continua valendo e a resposta diz que o exame não foi
        # guardado. Sem modelo o exame É o caso: sem ele o link abriria
        # "caso não encontrado", então o request falha.
        try:
            _store_exam_series(exam_bytes, exam_info, uid, r2)
            # Só com a série 0 gravada faz sentido acrescentar outras.
            write_token = _write_token(uid)
        except R2Error as e:
            print(f"[exam ERROR] failed cases/{uid}.exam-0: {e}")
            if glb_bytes is None:
                raise HTTPException(
                    502, "Não foi possível guardar o exame agora. Tente enviar de novo."
                )
            exam_info["stored"] = False
            exam_info["error"] = "O exame não pôde ser guardado. Tente enviar de novo."

    return {
        "uid": uid,
        "viewer_url": f"{VIEWER_BASE}?id={uid}",
        "stats": (
            {
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
            }
            if stats is not None
            else None
        ),
        "exam": exam_info,
        # Autoriza POST /cases/{uid}/exam (séries 1..3). A página guarda só na
        # memória; None quando o caso não tem exame gravado.
        "write_token": write_token,
        # Nada processa depois da resposta desde o Sprint 3c. Fica por
        # compatibilidade: a página de antes do 3c consultava o /status (que
        # não existe mais) a menos que viesse `processing: false`.
        "processing": False,
    }


@app.post("/cases/{uid}/exam")
def add_exam_series(
    uid: str,
    write_token: str = Form(...),
    index: int = Form(...),
    exam: list[UploadFile] = File(...),
) -> dict:
    """Série extra (1..3) do exame de um caso criado pelo /upload.

    Idempotente: o mesmo `index` sobrescreve — é assim que a página tenta de
    novo uma série que falhou. O servidor não confere se a série alinha com as
    estruturas (FrameOfReferenceUID): a página faz isso antes de enviar, e
    clientes de API são confiáveis.
    """
    # Bytes: compare_digest com str não-ASCII (token forjado) lança TypeError.
    token_ok = hmac.compare_digest(write_token.encode(), _write_token(uid).encode())
    if not _UID_RE.fullmatch(uid) or not token_ok:
        raise HTTPException(403, "Este envio não tem permissão para alterar o caso.")
    if not 1 <= index < MAX_EXAM_SERIES:
        raise HTTPException(
            400, f"Um caso tem no máximo {MAX_EXAM_SERIES} séries de exame."
        )
    nrrd, stats = _normalize_exam(exam)
    _log_exam(stats)
    info = _exam_info(stats, index)
    try:
        _store_exam_series(nrrd, info, uid, _r2())
    except R2Error as e:
        print(f"[exam ERROR] failed cases/{uid}.exam-{index}: {e}")
        raise HTTPException(
            502, "Não foi possível guardar esta série agora. Tente enviar de novo."
        )
    return {"uid": uid, "exam": info}

