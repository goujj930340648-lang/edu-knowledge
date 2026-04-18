from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import require_mongo, verify_api_key
from pymongo.database import Database
from services.vector_doc_search import (
    search_documents_with_hydrate,
    search_documents_with_hydrate_meta,
)
from storage.repository import list_modules_for_series, search_courses, search_questions

router = APIRouter(tags=["search"], dependencies=[Depends(verify_api_key)])


@router.get("/courses")
def search_courses_api(
    keyword: str = "",
    limit: int = Query(50, ge=1, le=200),
    include_modules: bool = False,
    db: Database = Depends(require_mongo),
) -> dict:
    rows = search_courses(db, keyword, limit=limit)
    out: dict = {"series": rows}
    if include_modules and rows:
        codes = [str(r.get("series_code") or "") for r in rows if r.get("series_code")]
        out["modules"] = list_modules_for_series(db, codes)
    return out


@router.get("/questions")
def search_questions_api(
    keyword: str = "",
    bank_code: str = "",
    question_type: str = "",
    limit: int = Query(50, ge=1, le=200),
    quality_flags_only: bool = Query(
        False,
        description="仅返回 quality_flags 非空的题目（质量看板）",
    ),
    db: Database = Depends(require_mongo),
) -> dict:
    items = search_questions(
        db,
        keyword=keyword,
        bank_code=bank_code,
        question_type=question_type,
        limit=limit,
        quality_flags_only=quality_flags_only,
    )
    return {"items": items}


@router.get("/documents")
def search_documents_api(
    q: str = Query("", description="检索查询"),
    doc_type: str = Query("", description="course_doc / project_doc，需 v2 入库带 doc_type"),
    limit: int = Query(10, ge=1, le=50),
    hydrate: bool = Query(True, description="是否用 Mongo knowledge_chunk 回表"),
    hyde: bool = Query(
        False,
        description="HyDE：LLM 生成假设文档再检索（v2）；亦可设环境变量 SEARCH_DOCUMENT_HYDE=1",
    ),
    rerank: bool = Query(
        False,
        description="BGE 精排 + 悬崖截断；亦可设 SEARCH_DOCUMENT_RERANK=1",
    ),
    retrieval_meta: bool = Query(
        False,
        description="为 True 时返回 retrieval_meta（hyde_used / rerank_used）",
    ),
    db: Database = Depends(require_mongo),
) -> dict:
    if not (q or "").strip():
        return {"hits": [], "count": 0}
    dt = doc_type.strip() or None
    if retrieval_meta:
        pack = search_documents_with_hydrate_meta(
            db if hydrate else None,
            q,
            doc_type=dt,
            limit=limit,
            use_hyde=hyde,
            use_rerank=rerank,
        )
        hits = pack["hits"]
        return {
            "hits": hits,
            "count": len(hits),
            "retrieval_meta": pack.get("retrieval_meta") or {},
        }
    hits = search_documents_with_hydrate(
        db if hydrate else None,
        q,
        doc_type=dt,
        limit=limit,
        use_hyde=hyde,
        use_rerank=rerank,
    )
    return {"hits": hits, "count": len(hits)}
