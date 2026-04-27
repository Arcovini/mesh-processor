"""Thin Cloudflare R2 client (S3-compatible API).

Knows nothing about meshes; only object storage. Takes bytes + uid,
writes to cases/{uid}.glb. Sprint 2 destination — runs in parallel
with sketchfab.upload_model so we start owning the GLBs ourselves.

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


def upload_glb(
    glb_bytes: bytes,
    *,
    uid: str,
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Upload GLB bytes to R2 at cases/{uid}.glb. Raises R2Error on failure."""
    key = f"cases/{uid}.glb"

    if _dry_run():
        print(f"[r2 DRY_RUN] upload '{key}' ({len(glb_bytes)} bytes) -> bucket={bucket}")
        return

    try:
        client = _client(account_id, access_key, secret_key)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=glb_bytes,
            ContentType=GLB_CONTENT_TYPE,
        )
    except (BotoCoreError, ClientError) as e:
        raise R2Error(f"Upload R2 falhou: {e}") from e
