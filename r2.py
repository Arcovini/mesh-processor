"""Thin Cloudflare R2 client (S3-compatible API).

Knows nothing about meshes or exams; only object storage. Takes bytes + uid,
writes to cases/{uid}.glb (model) and, per image-exam series n (0 = the one the
structures were segmented on), cases/{uid}.exam-{n}.nrrd + cases/{uid}.exam-{n}.json.
Sprint 2 destination — runs in parallel with sketchfab.upload_model so we
start owning the GLBs ourselves.

DRY_RUN=true short-circuits the real API — useful for local dev so we
don't spend ops budget on iterations, and lets the service boot without
R2 credentials configured.
"""
from __future__ import annotations

import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 60
GLB_CONTENT_TYPE = "model/gltf-binary"
EXAM_CONTENT_TYPE = "application/octet-stream"
EXAM_META_CONTENT_TYPE = "application/json"


class R2Error(RuntimeError):
    pass


def _dry_run() -> bool:
    return os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes")


def _client(account_id: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(connect_timeout=CONNECT_TIMEOUT_S, read_timeout=READ_TIMEOUT_S),
    )


def _put(
    data: bytes,
    *,
    key: str,
    content_type: str,
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Grava `data` em `key`. Único lugar que fala com o R2 (e com o DRY_RUN)."""
    if _dry_run():
        print(f"[r2 DRY_RUN] upload '{key}' ({len(data)} bytes) -> bucket={bucket}")
        # Dev: com DRY_RUN_GLB_DIR apontando para a raiz do medCaseViewer servida
        # localmente, o objeto é gravado em <dir>/cases/{uid}.glb (ou .exam-n.*) e o
        # viewer local abre o caso de verdade pelo link devolvido (ver loader.js,
        # que tenta o caminho local antes do R2 quando roda em localhost). Sem a
        # variável, nada é escrito e o comportamento é o de antes. O nome da
        # variável ficou do tempo em que só existia o GLB.
        dev_dir = os.getenv("DRY_RUN_GLB_DIR", "").strip()
        if dev_dir:
            path = os.path.join(dev_dir, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            print(f"[r2 DRY_RUN] gravado em {path}")
        return

    try:
        client = _client(account_id, access_key, secret_key)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    except (BotoCoreError, ClientError) as e:
        raise R2Error(f"Upload R2 falhou ({key}): {e}") from e


def upload_glb(
    glb_bytes: bytes,
    *,
    uid: str,
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
) -> None:
    """GLB em cases/{uid}.glb. Levanta R2Error em falha."""
    _put(
        glb_bytes, key=f"cases/{uid}.glb", content_type=GLB_CONTENT_TYPE,
        bucket=bucket, account_id=account_id, access_key=access_key, secret_key=secret_key,
    )


def exam_key(uid: str, n: int, ext: str) -> str:
    """Chave de uma série do exame: cases/{uid}.exam-{n}.{nrrd|json}."""
    return f"cases/{uid}.exam-{n}.{ext}"


def upload_exam(
    nrrd_bytes: bytes,
    *,
    uid: str,
    n: int,
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Série n do exame (exam.py) em cases/{uid}.exam-{n}.nrrd. Levanta R2Error.

    O gzip está DENTRO do NRRD (`encoding: gzip`), não na transferência: por
    isso octet-stream e nenhum Content-Encoding. Se o R2 anunciasse gzip, o
    navegador inflaria os bytes e o cabeçalho do NRRD passaria a mentir.
    """
    _put(
        nrrd_bytes, key=exam_key(uid, n, "nrrd"), content_type=EXAM_CONTENT_TYPE,
        bucket=bucket, account_id=account_id, access_key=access_key, secret_key=secret_key,
    )


def upload_exam_meta(
    json_bytes: bytes,
    *,
    uid: str,
    n: int,
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Metadados da série n em cases/{uid}.exam-{n}.json. Levanta R2Error.

    Gravado DEPOIS do NRRD: é a existência deste JSON que faz a série aparecer
    no visualizador, então um NRRD sem JSON (falha no meio) fica invisível.
    """
    _put(
        json_bytes, key=exam_key(uid, n, "json"), content_type=EXAM_META_CONTENT_TYPE,
        bucket=bucket, account_id=account_id, access_key=access_key, secret_key=secret_key,
    )
