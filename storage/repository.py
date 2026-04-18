"""MongoDB 写入与查询（与 PLAN collection 命名对齐）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ReplaceOne, UpdateOne
from pymongo.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def upsert_course_catalog(
    db: Database,
    series_list: list[dict[str, Any]],
    modules: list[dict[str, Any]],
) -> dict[str, Any]:
    s_ops = []
    for s in series_list:
        code = (s.get("series_code") or "").strip()
        if not code:
            continue
        doc = {**s, "updated_at": _utcnow()}
        s_ops.append(
            ReplaceOne(
                {"series_code": code},
                {**doc, "created_at": doc.get("created_at") or _utcnow()},
                upsert=True,
            )
        )
    m_ops = []
    for m in modules:
        mc = (m.get("module_code") or "").strip()
        if not mc:
            continue
        doc = {**m, "updated_at": _utcnow()}
        m_ops.append(
            ReplaceOne(
                {"module_code": mc},
                {**doc, "created_at": doc.get("created_at") or _utcnow()},
                upsert=True,
            )
        )
    s_res = db.course_series.bulk_write(s_ops, ordered=False) if s_ops else None
    m_res = db.course_module.bulk_write(m_ops, ordered=False) if m_ops else None
    return {
        "series_upserted": getattr(s_res, "upserted_count", 0) + getattr(s_res, "modified_count", 0)
        if s_res
        else 0,
        "modules_upserted": getattr(m_res, "upserted_count", 0) + getattr(m_res, "modified_count", 0)
        if m_res
        else 0,
    }


def upsert_question_data(
    db: Database,
    banks: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    b_ops = []
    for b in banks:
        bc = (b.get("bank_code") or "").strip()
        if not bc:
            continue
        doc = {**b, "updated_at": _utcnow()}
        b_ops.append(
            ReplaceOne(
                {"bank_code": bc},
                {**doc, "created_at": doc.get("created_at") or _utcnow()},
                upsert=True,
            )
        )
    i_ops = []
    for it in items:
        qc = (it.get("question_code") or "").strip()
        if not qc:
            continue
        doc = {**it, "updated_at": _utcnow()}
        i_ops.append(
            ReplaceOne(
                {"question_code": qc},
                {**doc, "created_at": doc.get("created_at") or _utcnow()},
                upsert=True,
            )
        )
    b_res = db.question_bank.bulk_write(b_ops, ordered=False) if b_ops else None
    i_res = db.question_item.bulk_write(i_ops, ordered=False) if i_ops else None
    return {
        "banks_upserted": getattr(b_res, "upserted_count", 0) + getattr(b_res, "modified_count", 0)
        if b_res
        else 0,
        "items_upserted": getattr(i_res, "upserted_count", 0) + getattr(i_res, "modified_count", 0)
        if i_res
        else 0,
    }


def create_ingest_task(
    db: Database,
    task_id: str,
    task_type: str,
    *,
    sub_tasks: list[dict[str, Any]] | None = None,
) -> None:
    db.ingest_task.insert_one(
        {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "sub_tasks": sub_tasks or [],
            "progress_logs": [],
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
        }
    )


def update_ingest_task(
    db: Database,
    task_id: str,
    patch: dict[str, Any],
    *,
    push_log: str | None = None,
) -> None:
    upd: dict[str, Any] = {"$set": {**patch, "updated_at": _utcnow()}}
    if push_log:
        upd.setdefault("$push", {})["progress_logs"] = {
            "ts": _utcnow().isoformat(),
            "message": push_log,
        }
    db.ingest_task.update_one({"task_id": task_id}, upd)


def get_ingest_task(db: Database, task_id: str) -> dict[str, Any] | None:
    return db.ingest_task.find_one({"task_id": task_id}, {"_id": 0})


def persist_document_import(
    db: Database,
    *,
    doc_id: str,
    doc_type: str,
    source_path: str,
    source_file: str,
    title: str,
    ingest_task_id: str,
    graph_state: dict[str, Any],
    image_object_keys: list[str] | None = None,
    asset_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """根据导入图终态写入 ``knowledge_document`` / ``knowledge_chunk``。"""
    from processor.vector_indexer.utils import content_fingerprint
    from schema.edu_content import EduContent

    chunks_raw = graph_state.get("chunks") or []
    indexed = graph_state.get("indexed_records") or []
    vec_ids = graph_state.get("vector_ids") or []
    hash_to_milvus: dict[str, str] = {}
    for row, pk in zip(indexed, vec_ids):
        if isinstance(row, dict) and row.get("content_hash"):
            hash_to_milvus[str(row["content_hash"])] = str(pk)

    chunk_docs: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks_raw):
        if not isinstance(ch, dict):
            continue
        raw_ec = ch.get("edu_content")
        if not raw_ec:
            continue
        try:
            edu = EduContent.model_validate(raw_ec)
        except Exception:
            continue
        cid = str(ch.get("chunk_id") or "").strip()
        chash = cid if cid else content_fingerprint(edu)
        heading = ch.get("heading_path") or []
        section_path = [str(x) for x in heading] if isinstance(heading, list) else []
        chunk_docs.append(
            {
                "chunk_id": chash,
                "doc_id": doc_id,
                "section_path": section_path,
                "chunk_text": edu.content,
                "chunk_kind": "text",
                "chunk_index": i,
                "content_hash": chash,
                "milvus_pk": hash_to_milvus.get(chash),
                "image_refs": [],
                "code_refs": [],
                "table_refs": [],
                "updated_at": _utcnow(),
                "created_at": _utcnow(),
            }
        )

    if chunk_docs:
        db.knowledge_chunk.delete_many({"doc_id": doc_id})
        db.knowledge_chunk.insert_many(chunk_docs)

    img_keys = list(image_object_keys or [])
    if asset_rows:
        db.asset_object.delete_many({"source_doc_id": doc_id})
        now = _utcnow()
        for a in asset_rows:
            a.setdefault("created_at", now)
            a.setdefault("updated_at", now)
        db.asset_object.insert_many(asset_rows)

    db.knowledge_document.replace_one(
        {"doc_id": doc_id},
        {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_path": source_path,
            "source_file": source_file,
            "title": title,
            "domain_tags": [],
            "chunk_count": len(chunk_docs),
            "image_count": len(img_keys),
            "image_object_keys": img_keys,
            "ingest_task_id": ingest_task_id,
            "updated_at": _utcnow(),
            "created_at": _utcnow(),
        },
        upsert=True,
    )
    return {"chunk_count": len(chunk_docs), "image_count": len(img_keys)}


def list_modules_for_series(
    db: Database, series_codes: list[str]
) -> list[dict[str, Any]]:
    if not series_codes:
        return []
    return list(
        db.course_module.find({"series_code": {"$in": series_codes}}, {"_id": 0})
    )


def search_courses(db: Database, keyword: str, limit: int = 50) -> list[dict[str, Any]]:
    if not keyword.strip():
        cur = db.course_series.find({}, {"_id": 0}).limit(limit)
        return list(cur)
    rx = {"$regex": keyword, "$options": "i"}
    q = {"$or": [{"title": rx}, {"series_code": rx}, {"category_path": rx}, {"description": rx}]}
    return list(db.course_series.find(q, {"_id": 0}).limit(limit))


def search_questions(
    db: Database,
    *,
    keyword: str = "",
    bank_code: str = "",
    question_type: str = "",
    limit: int = 50,
    quality_flags_only: bool = False,
) -> list[dict[str, Any]]:
    q: dict[str, Any] = {}
    if bank_code.strip():
        q["bank_code"] = bank_code.strip()
    if question_type.strip():
        q["question_type"] = {"$regex": question_type, "$options": "i"}
    if quality_flags_only:
        q["quality_flags.0"] = {"$exists": True}
    if keyword.strip():
        rx = {"$regex": keyword, "$options": "i"}
        q["$or"] = [{"stem": rx}, {"question_code": rx}]
    cur = db.question_item.find(q, {"_id": 0}).limit(limit)
    return list(cur)


def get_chunks_by_content_hashes(
    db: Database, hashes: list[str]
) -> dict[str, dict[str, Any]]:
    if not hashes:
        return {}
    cur = db.knowledge_chunk.find({"content_hash": {"$in": hashes}}, {"_id": 0})
    return {str(d.get("content_hash") or ""): d for d in cur if d.get("content_hash")}


def search_document_chunks_by_text(
    db: Database, keyword: str, limit: int = 20
) -> list[dict[str, Any]]:
    if not keyword.strip():
        return []
    rx = {"$regex": keyword, "$options": "i"}
    return list(db.knowledge_chunk.find({"chunk_text": rx}, {"_id": 0}).limit(limit))


def insert_chat_messages(
    db: Database,
    session_id: str,
    task_id: str,
    user_text: str,
    assistant_text: str,
    *,
    citations: list[Any] | None = None,
    intent: str = "",
) -> None:
    base = _utcnow()
    docs = [
        {
            "session_id": session_id,
            "task_id": task_id,
            "role": "user",
            "content": user_text,
            "citations": [],
            "intent": intent,
            "created_at": base,
        },
        {
            "session_id": session_id,
            "task_id": task_id,
            "role": "assistant",
            "content": assistant_text,
            "citations": citations or [],
            "intent": intent,
            "created_at": base,
        },
    ]
    db.chat_history.insert_many(docs)


def list_chat_history(db: Database, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    cur = (
        db.chat_history.find({"session_id": session_id}, {"_id": 0})
        .sort("created_at", 1)
        .limit(limit)
    )
    return list(cur)


def upsert_source_mapping(
    db: Database,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    ops = []
    for r in rows:
        key = {
            "source_file": r.get("source_file", ""),
            "doc_id": r.get("doc_id") or "",
        }
        ops.append(
            UpdateOne(
                key,
                {"$set": {**r, "updated_at": _utcnow()}, "$setOnInsert": {"created_at": _utcnow()}},
                upsert=True,
            )
        )
    res = db.source_mapping.bulk_write(ops, ordered=False)
    return res.upserted_count + res.modified_count


def list_source_mappings(
    db: Database,
    *,
    doc_id: str | None = None,
    source_file: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    q: dict[str, Any] = {}
    if doc_id and str(doc_id).strip():
        q["doc_id"] = str(doc_id).strip()
    if source_file and str(source_file).strip():
        q["source_file"] = str(source_file).strip()
    cur = db.source_mapping.find(q, {"_id": 0}).limit(max(1, min(limit, 500)))
    return list(cur)
