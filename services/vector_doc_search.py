"""Milvus 文档向量检索 + Mongo 回表（可选）；可选 HyDE + RRF、可选 BGE 精排 + 悬崖截断。"""

from __future__ import annotations

import os
from typing import Any

from pymongo.database import Database

from processor.query_process.config import get_query_config
from processor.query_process.embed_util import embed_query_for_chunks
from storage.repository import get_chunks_by_content_hashes
from utils.milvus_search_edu import (
    dense_search_legacy_chunks,
    hybrid_search_chunks_v2,
    rag_mode,
)
from utils.retrieval_rrf import rrf_merge_hits


def _env_bool(key: str, default: bool = False) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _doc_type_expr(doc_type: str | None) -> str | None:
    if not (doc_type or "").strip():
        return None
    dt = doc_type.strip().replace("\\", "\\\\").replace('"', '\\"')
    return f'doc_type == "{dt}"'


def _milvus_hybrid_hits(
    q: str,
    *,
    doc_type: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    q = (q or "").strip()
    if not q or limit <= 0:
        return []
    mode = rag_mode()
    expr: str | None = _doc_type_expr(doc_type) if mode == "v2" else None
    try:
        dense, sparse = embed_query_for_chunks(q)
        if mode == "v2":
            if sparse is None:
                return []
            return hybrid_search_chunks_v2(dense, sparse, limit=limit, expr=expr)
        return dense_search_legacy_chunks(dense, limit=limit, expr=expr)
    except Exception:
        return []


def _try_hyde_document(user_query: str, catalog_hint: str = "（未限定）") -> str | None:
    if len((user_query or "").strip()) < 4:
        return None
    try:
        from prompts.query_prompt import HYDE_SYSTEM_TEMPLATE, HYDE_USER_TEMPLATE
        from utils.client import get_llm_client

        llm = get_llm_client()
        user_p = HYDE_USER_TEMPLATE.format(
            catalog_hint=catalog_hint,
            rewritten_query=user_query.strip(),
        )
        prompt = f"{HYDE_SYSTEM_TEMPLATE}\n\n{user_p}"
        hy = llm.chat(prompt=prompt, json_mode=False).strip()
    except Exception:
        return None
    if not hy or len(hy) < 20:
        return None
    return hy


def _effective_hyde(use_hyde_param: bool) -> bool:
    return use_hyde_param or _env_bool("SEARCH_DOCUMENT_HYDE", False)


def _effective_rerank(use_rerank_param: bool) -> bool:
    return use_rerank_param or _env_bool("SEARCH_DOCUMENT_RERANK", False)


def _maybe_rerank_and_cliff(
    query: str,
    hits: list[dict[str, Any]],
    *,
    final_limit: int,
) -> list[dict[str, Any]]:
    if not hits or not (query or "").strip():
        return hits[:final_limit] if final_limit > 0 else []
    cfg = get_query_config()
    try:
        from utils.local_bge_reranker import cliff_truncation_count, get_local_bge_reranker

        reranker = get_local_bge_reranker()
    except Exception:
        return hits[:final_limit] if final_limit > 0 else []

    docs: list[dict[str, Any]] = []
    for h in hits:
        ent = h.get("entity") if isinstance(h, dict) else None
        if not isinstance(ent, dict):
            continue
        c = str(ent.get("content") or "").strip()
        if not c:
            continue
        docs.append({"hit": h, "content": c})
    if not docs:
        return []
    pairs = [(query, d["content"]) for d in docs]
    try:
        try:
            raw_scores = reranker.compute_score(pairs, normalize=True)
        except TypeError:
            raw_scores = reranker.compute_score(pairs)
    except Exception:
        return hits[:final_limit] if final_limit > 0 else []

    try:
        import numpy as np

        arr = np.asarray(raw_scores).ravel()
        scores = [float(x) for x in arr]
    except Exception:
        scores = [float(x) for x in raw_scores] if isinstance(raw_scores, (list, tuple)) else []

    if len(scores) != len(docs):
        return hits[:final_limit] if final_limit > 0 else []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits_sorted = [docs[i]["hit"] for i in order]
    scores_sorted = [scores[i] for i in order]

    max_k = min(cfg.rerank_max_top_k, len(scores_sorted))
    min_k = min(cfg.rerank_min_top_k, max_k)
    n_keep = cliff_truncation_count(
        scores_sorted,
        max_top_k=max_k,
        min_top_k=min_k,
        gap_ratio=cfg.rerank_gap_ratio,
        gap_abs=cfg.rerank_gap_abs,
    )
    trimmed = hits_sorted[:n_keep]
    if final_limit > 0:
        trimmed = trimmed[:final_limit]
    return trimmed


def hybrid_document_search(
    query: str,
    *,
    doc_type: str | None = None,
    limit: int = 10,
    use_hyde: bool = False,
    catalog_hint: str = "（未限定）",
    use_rerank: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    返回 (hits, meta)。HyDE：原查询 + 假设文档两路 RRF（仅 v2 且 HyDE 成功时）；legacy 仅原查询。
    """
    meta: dict[str, Any] = {"hyde_used": False, "rerank_used": False, "hyde_error": None}
    q = (query or "").strip()
    if not q:
        return [], meta

    branch_limit = max(
        limit,
        int(os.environ.get("SEARCH_HYDE_BRANCH_LIMIT", str(max(limit * 2, 16)))),
    )
    rrf_k = int(os.environ.get("RRF_K", "60"))
    rrf_max = max(
        limit,
        int(os.environ.get("RRF_MAX_RESULTS", str(max(limit * 2, 16)))),
    )

    want_hyde = _effective_hyde(use_hyde)
    want_rerank = _effective_rerank(use_rerank)

    if want_hyde and rag_mode() == "v2":
        hy_text: str | None = None
        try:
            hy_text = _try_hyde_document(q, catalog_hint=catalog_hint)
        except Exception as e:  # pragma: no cover
            meta["hyde_error"] = str(e)
        if hy_text:
            meta["hyde_used"] = True
            h0 = _milvus_hybrid_hits(q, doc_type=doc_type, limit=branch_limit)
            h1 = _milvus_hybrid_hits(hy_text, doc_type=doc_type, limit=branch_limit)
            hits = rrf_merge_hits(
                [(h0, 1.0), (h1, 1.0)],
                rrf_k=rrf_k,
                max_results=rrf_max,
            )
        else:
            hits = _milvus_hybrid_hits(q, doc_type=doc_type, limit=branch_limit)
    else:
        hits = _milvus_hybrid_hits(q, doc_type=doc_type, limit=branch_limit)

    if want_rerank and hits:
        meta["rerank_used"] = True
        hits = _maybe_rerank_and_cliff(q, hits, final_limit=limit)
    elif limit > 0:
        hits = hits[:limit]

    return hits, meta


def _hydrate_milvus_hits(db: Database, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hashes: list[str] = []
    for h in hits:
        ent = h.get("entity") if isinstance(h, dict) else None
        if isinstance(ent, dict) and ent.get("content_hash"):
            hashes.append(str(ent["content_hash"]))
    by_hash = get_chunks_by_content_hashes(db, hashes)
    out: list[dict[str, Any]] = []
    for h in hits:
        ent = (h.get("entity") or {}) if isinstance(h, dict) else {}
        chash = str(ent.get("content_hash") or "")
        row = by_hash.get(chash, {})
        out.append(
            {
                "score": h.get("distance"),
                "milvus": ent,
                "chunk": row or None,
            }
        )
    return out


def search_documents_with_hydrate(
    db: Database | None,
    query: str,
    *,
    doc_type: str | None = None,
    limit: int = 10,
    use_hyde: bool = False,
    catalog_hint: str = "（未限定）",
    use_rerank: bool = False,
) -> list[dict[str, Any]]:
    hits, _meta = hybrid_document_search(
        query,
        doc_type=doc_type,
        limit=limit,
        use_hyde=use_hyde,
        catalog_hint=catalog_hint,
        use_rerank=use_rerank,
    )
    if not db or not hits:
        return hits
    return _hydrate_milvus_hits(db, hits)


def search_documents_with_hydrate_meta(
    db: Database | None,
    query: str,
    *,
    doc_type: str | None = None,
    limit: int = 10,
    use_hyde: bool = False,
    catalog_hint: str = "（未限定）",
    use_rerank: bool = False,
) -> dict[str, Any]:
    """单次检索 + ``retrieval_meta``（HyDE / 精排是否生效）。"""
    hits, meta = hybrid_document_search(
        query,
        doc_type=doc_type,
        limit=limit,
        use_hyde=use_hyde,
        catalog_hint=catalog_hint,
        use_rerank=use_rerank,
    )
    if not db:
        return {"hits": hits, "retrieval_meta": meta}
    return {"hits": _hydrate_milvus_hits(db, hits), "retrieval_meta": meta}
