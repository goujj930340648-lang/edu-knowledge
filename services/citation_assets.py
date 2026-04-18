"""为引用补充 MinIO 预签名图链（按 chunk 指纹关联文档）。"""

from __future__ import annotations

from typing import Any

from pymongo.database import Database

from storage.minio_client import presigned_get_url


def _presign_keys(keys: list[str], *, max_urls: int = 5) -> list[str]:
    out: list[str] = []
    for k in keys[:max_urls]:
        k = (k or "").strip()
        if not k:
            continue
        ok, url, _ = presigned_get_url(k, expires_seconds=3600)
        if ok and url:
            out.append(url)
    return out


def enrich_citations_with_images(
    db: Database | None,
    citations: list[dict[str, Any]],
    *,
    max_urls_per_citation: int = 5,
) -> list[dict[str, Any]]:
    if not db or not citations:
        return citations
    out: list[dict[str, Any]] = []
    for c in citations:
        row = dict(c)
        chash = str(row.get("chunk_id") or "").strip()
        if not chash:
            out.append(row)
            continue
        chunk = db.knowledge_chunk.find_one(
            {"content_hash": chash},
            {"_id": 0, "doc_id": 1},
        )
        if not chunk:
            out.append(row)
            continue
        doc_id = str(chunk.get("doc_id") or "").strip()
        if not doc_id:
            out.append(row)
            continue
        doc = db.knowledge_document.find_one(
            {"doc_id": doc_id},
            {"_id": 0, "image_object_keys": 1},
        )
        keys = list((doc or {}).get("image_object_keys") or [])
        urls = _presign_keys(keys, max_urls=max_urls_per_citation)
        if urls:
            row["image_urls"] = urls
        out.append(row)
    return out
