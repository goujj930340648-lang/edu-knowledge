"""同步 PyMongo 连接与索引（供 FastAPI BackgroundTasks / 依赖注入）。"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=8)
def _mongo_client(uri: str) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def close_mongo_client() -> None:
    _mongo_client.cache_clear()


@lru_cache(maxsize=8)
def get_mongo_db(uri: str, db_name: str) -> Database:
    client = _mongo_client(uri)
    db = client[db_name]
    _ensure_indexes(db)
    return db


def _ensure_indexes(db: Database) -> None:
    db.course_series.create_index("series_code", unique=True)
    db.course_module.create_index("module_code", unique=True)
    db.course_module.create_index("series_code")
    db.question_bank.create_index("bank_code", unique=True)
    db.question_item.create_index("question_code", unique=True)
    db.question_item.create_index("bank_code")
    db.knowledge_document.create_index("doc_id", unique=True)
    db.knowledge_chunk.create_index("chunk_id", unique=True)
    db.knowledge_chunk.create_index("doc_id")
    db.knowledge_chunk.create_index("content_hash")
    db.ingest_task.create_index("task_id", unique=True)
    db.chat_history.create_index([("session_id", 1), ("created_at", 1)])
    db.source_mapping.create_index([("source_file", 1), ("doc_id", 1)])
    db.source_mapping.create_index("doc_id")
    db.asset_object.create_index("object_key", unique=True)
    db.asset_object.create_index("source_doc_id")


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
