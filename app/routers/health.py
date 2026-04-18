from __future__ import annotations

import os

from fastapi import APIRouter

from config.settings import get_settings
from storage.mongo_db import mongo_health_check
from storage.minio_client import minio_health_check
from utils.client import maybe_strip_proxy_for_milvus

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    mongo_ok, mongo_err = mongo_health_check(s.resolved_mongo_uri())

    milvus_ok = True
    milvus_err: str | None = None
    milvus_configured = False
    u = (os.environ.get("MILVUS_URI") or os.environ.get("MILVUS_URL") or "").strip()
    if u:
        milvus_configured = True
        try:
            maybe_strip_proxy_for_milvus()
            from pymilvus import connections

            alias = "health_ping"
            if connections.has_connection(alias):
                connections.disconnect(alias)
            connections.connect(alias, uri=u)
            connections.disconnect(alias)
            milvus_ok = True
        except Exception as e:
            milvus_ok = False
            milvus_err = str(e)
    else:
        milvus_err = "MILVUS_URI 未设置（向量检索不可用）"

    minio_info: dict = {"configured": False, "ok": None, "error": None}
    if s.minio_configured():
        m_ok, m_err = minio_health_check()
        minio_info = {
            "configured": True,
            "ok": m_ok,
            "error": None if m_ok else m_err,
        }

    core_ok = mongo_ok
    vector_ok = (not milvus_configured) or milvus_ok
    status = "ok" if core_ok and vector_ok else "degraded"

    return {
        "status": status,
        "mongodb": {"ok": mongo_ok, "error": mongo_err or None},
        "milvus": {
            "configured": milvus_configured,
            "ok": milvus_ok if milvus_configured else None,
            "error": milvus_err,
        },
        "minio": minio_info,
    }
