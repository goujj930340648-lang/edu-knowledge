"""同步 PyMongo 连接与索引（供 FastAPI BackgroundTasks / 依赖注入）。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=8)
def _mongo_client(uri: str) -> MongoClient:
    return MongoClient(
        uri, 
        serverSelectionTimeoutMS=5000,
        directConnection=True,
        connectTimeoutMS=2000
    )


def close_mongo_client() -> None:
    _mongo_client.cache_clear()


@lru_cache(maxsize=8)
def get_mongo_db(uri: str, db_name: str) -> Database:
    client = _mongo_client(uri)
    db = client[db_name]
    _ensure_indexes(db)
    return db


def _create_index_safe(collection: Collection, keys: Any, **kwargs: Any) -> None:
    """创建索引；若因已有数据违反唯一约束导致建索引失败，则记录告警并跳过（避免拖垮所有 API）。"""
    try:
        collection.create_index(keys, **kwargs)
    except DuplicateKeyError:
        logger.warning(
            "跳过索引 %s keys=%r %r：集合中存在重复键，无法建立唯一索引，请清理数据后重试",
            collection.full_name,
            keys,
            kwargs,
        )


def _ensure_indexes(db: Database) -> None:
    _create_index_safe(db.course_series, "series_code", unique=True)
    _create_index_safe(db.course_module, "module_code", unique=True)
    _create_index_safe(db.course_module, "series_code")
    _create_index_safe(db.question_bank, "bank_code", unique=True)
    _create_index_safe(db.question_item, "question_code", unique=True)
    _create_index_safe(db.question_item, "bank_code")
    _create_index_safe(db.knowledge_document, "doc_id", unique=True)
    _create_index_safe(db.knowledge_chunk, "chunk_id", unique=True)
    _create_index_safe(db.knowledge_chunk, "doc_id")
    _create_index_safe(db.knowledge_chunk, "content_hash")
    _create_index_safe(db.ingest_task, "task_id", unique=True)
    _create_index_safe(db.chat_history, [("session_id", 1), ("created_at", 1)])
    _create_index_safe(db.source_mapping, [("source_file", 1), ("doc_id", 1)])
    _create_index_safe(db.source_mapping, "doc_id")
    _create_index_safe(db.asset_object, "object_key", unique=True)
    _create_index_safe(db.asset_object, "source_doc_id")


def mongo_health_check(uri: str) -> tuple[bool, str]:
    if not uri.strip():
        return False, "MONGODB_URI / MONGO_URL 未配置"
    try:
        c = MongoClient(uri, serverSelectionTimeoutMS=3000)
        try:
            c.admin.command("ping")
        finally:
            c.close()
        return True, ""
    except PyMongoError as e:
        return False, str(e)


def append_progress(db: Database, task_id: str, message: str) -> None:
    db.ingest_task.update_one(
        {"task_id": task_id},
        {
            "$push": {
                "progress_logs": {"ts": _utcnow().isoformat(), "message": message}
            },
            "$set": {"updated_at": _utcnow()},
        },
    )
