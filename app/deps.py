from __future__ import annotations

from fastapi import HTTPException, Request
from pymongo.database import Database

from config.settings import get_settings
from storage.mongo_db import get_mongo_db


def verify_api_key(request: Request) -> None:
    """
    若配置了 ``API_KEYS``（逗号分隔），则校验：

    - 请求头 ``X-API-Key``（常规 fetch）
    - 或查询参数 ``x_api_key``（``EventSource`` 无法带头时的权宜之计，仅建议本机联调）
    """
    keys = get_settings().api_key_list()
    if not keys:
        return
    h = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    q = request.query_params.get("x_api_key")
    val = (h or q or "").strip()
    if val not in keys:
        raise HTTPException(
            status_code=401,
            detail="无效或缺失的 X-API-Key（流式可尝试 URL ?x_api_key=）",
        )


def require_mongo() -> Database:
    s = get_settings()
    uri = s.resolved_mongo_uri()
    if not uri:
        raise HTTPException(
            status_code=503,
            detail="MongoDB 未配置：请在 .env 中设置 MONGODB_URI 或 MONGO_URL",
        )
    return get_mongo_db(uri, s.mongodb_database)
