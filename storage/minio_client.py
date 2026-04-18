"""MinIO（S3 兼容）上传；未配置 endpoint 时全部操作为空操作。"""

from __future__ import annotations

import io
from datetime import timedelta
from functools import lru_cache

from config.settings import get_settings


@lru_cache(maxsize=1)
def _client():
    s = get_settings()
    if not s.minio_configured():
        return None
    from minio import Minio

    ep = (s.minio_endpoint or "").strip().replace("https://", "").replace("http://", "")
    return Minio(
        ep,
        access_key=s.minio_access_key.strip(),
        secret_key=s.minio_secret_key.strip(),
        secure=bool(s.minio_secure),
    )


def ensure_bucket_exists() -> tuple[bool, str]:
    s = get_settings()
    c = _client()
    if c is None:
        return False, "MinIO 未配置"
    b = (s.minio_bucket or "education-knowledge").strip()
    try:
        if not c.bucket_exists(b):
            c.make_bucket(b)
        return True, b
    except Exception as e:
        return False, str(e)


def put_document_image(
    doc_id: str,
    filename: str,
    data: bytes,
    content_type: str,
) -> tuple[bool, str, str]:
    """
    写入 ``documents/{doc_id}/images/{filename}``。
    返回 ``(ok, object_key, error)``。
    """
    s = get_settings()
    c = _client()
    if c is None:
        return False, "", "MinIO 未配置"
    ok, bmsg = ensure_bucket_exists()
    if not ok:
        return False, "", bmsg
    bucket = (s.minio_bucket or "education-knowledge").strip()
    key = f"documents/{doc_id}/images/{filename}"
    try:
        c.put_object(
            bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )
        return True, key, ""
    except Exception as e:
        return False, key, str(e)


def public_url_for_key(object_key: str) -> str | None:
    base = (get_settings().minio_public_base_url or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/{object_key.lstrip('/')}"


def presigned_get_url(
    object_key: str,
    *,
    expires_seconds: int = 3600,
) -> tuple[bool, str, str]:
    """
    生成 GET 预签名 URL。
    返回 ``(ok, url_or_empty, error)``；未配置 MinIO 时 ``ok=False``。
    """
    s = get_settings()
    c = _client()
    if c is None:
        return False, "", "MinIO 未配置"
    ok, bmsg = ensure_bucket_exists()
    if not ok:
        return False, "", bmsg
    bucket = (s.minio_bucket or "education-knowledge").strip()
    key = object_key.lstrip("/")
    if not key or ".." in key:
        return False, "", "非法 object key"
    try:
        sec = max(60, min(int(expires_seconds), 86400 * 7))
    except (TypeError, ValueError):
        sec = 3600
    try:
        url = c.presigned_get_object(
            bucket,
            key,
            expires=timedelta(seconds=sec),
        )
        return True, str(url), ""
    except Exception as e:
        return False, "", str(e)


def minio_health_check() -> tuple[bool, str]:
    ok, msg = ensure_bucket_exists()
    if ok:
        return True, ""
    return False, msg
