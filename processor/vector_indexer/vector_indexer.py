"""
Vector_Indexer：将 ``state["chunks"]`` 中汇聚的 ``EduContent`` 向量化并写入 Milvus。

- **legacy**（默认）：单集合 ``MILVUS_COLLECTION``，单字段 ``vector``（OpenAI 兼容 Embedding）。
- **v2**：双表（默认 ``edu_knowledge_item_names_v1`` + ``edu_knowledge_chunks_v1``），稠密+稀疏（本地 BGE-M3）。

环境变量见 ``vector_indexer_node`` 文档字符串。
"""

from __future__ import annotations

from typing import Any

from schema.edu_content import EduContent

from processor.import_state import ImportGraphState
from processor.vector_indexer.config import VectorIndexerConfig
from processor.vector_indexer.embedding_service import (
    EmbeddingError,
    EmbeddingResult,
    get_embedding_service,
)
from processor.vector_indexer.utils import (
    MILVUS_VARCHAR_CONTENT_MAX,
    batch_query_field,
    catalog_display_name,
    content_fingerprint,
    extract_catalog_items,
    item_fingerprint,
    merge_upstream_lists,
    sanitize_milvus_row,
    truncate_content_field,
)
from utils.client import get_milvus_client


def _vector_indexer_legacy(
    documents: list[EduContent],
    flat_rows: list[dict[str, Any]],
    *,
    collection_name: str,
    skip_dedup: bool,
) -> dict[str, Any]:
    hashes = [r["content_hash"] for r in flat_rows]
    skipped_hashes: set[str] = set()
    if not skip_dedup:
        milvus_pre = get_milvus_client()
        try:
            skipped_hashes = batch_query_field(
                milvus_pre, collection_name, "content_hash", hashes
            )
        except Exception as e:
            return {"errors": [f"Milvus 去重查询失败: {e}"], "is_success": False}

    to_embed_docs: list[EduContent] = []
    to_embed_rows: list[dict[str, Any]] = []
    for doc, row in zip(documents, flat_rows):
        if row["content_hash"] in skipped_hashes:
            continue
        to_embed_docs.append(doc)
        to_embed_rows.append(row)

    if not to_embed_rows:
        return {
            "warnings": [
                f"全部切片均已存在（content_hash 命中 {len(skipped_hashes)} 条），跳过写入"
            ],
            "indexed_records": [],
            "vector_ids": [],
            "is_success": True,
        }

    texts = [d.content for d in to_embed_docs]
    try:
        service = get_embedding_service()
        result = service.embed_documents(texts)

        if isinstance(result, EmbeddingError):
            return {
                "errors": [f"Embedding 失败: {result.message}"],
                "is_success": False,
            }

        vectors = result.embeddings
    except Exception as e:
        return {"errors": [f"Embedding 失败: {e}"], "is_success": False}

    if len(vectors) != len(to_embed_rows):
        return {
            "errors": [
                f"Embedding 返回条数与文档不一致: {len(vectors)} != {len(to_embed_rows)}"
            ],
            "is_success": False,
        }

    trunc_notes: list[str] = []
    records = []
    for row, vec in zip(to_embed_rows, vectors):
        payload = dict(row)
        payload["vector"] = vec
        payload, did_trunc = truncate_content_field(payload)
        if did_trunc:
            trunc_notes.append(
                f"content 已截断至 {MILVUS_VARCHAR_CONTENT_MAX} 字符（Milvus VarChar 上限）"
            )
        records.append(sanitize_milvus_row(payload))

    milvus = get_milvus_client()
    try:
        res = milvus.insert(collection_name=collection_name, data=records)
    except Exception as e:
        return {"errors": [f"Milvus 入库失败: {e}"], "is_success": False}

    pks = getattr(res, "primary_keys", None)
    vector_ids = [str(x) for x in list(pks)] if pks is not None else []

    warnings: list[str] = []
    if skipped_hashes:
        warnings.append(f"已跳过 {len(skipped_hashes)} 条重复切片（content_hash 已存在）")
    if trunc_notes:
        warnings.append(
            f"{len(trunc_notes)} 条切片：{trunc_notes[0]}"
            + ("（其余同上）" if len(trunc_notes) > 1 else "")
        )

    out: dict[str, Any] = {
        "indexed_records": records,
        "vector_ids": vector_ids,
        "is_success": True,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _vector_indexer_v2(
    documents: list[EduContent],
    flat_rows: list[dict[str, Any]],
    *,
    names_collection: str,
    chunks_collection: str,
    skip_dedup: bool,
) -> dict[str, Any]:
    service = get_embedding_service()

    # Check if service supports dense+sparse (BGE-M3)
    # We'll try to use it and handle errors if it doesn't support sparse
    milvus = get_milvus_client()
    warnings: list[str] = []

    # --- 名称表 ---
    catalog_rows = extract_catalog_items(documents)
    catalog_inserts: list[dict[str, Any]] = []
    catalog_ids: list[str] = []
    if catalog_rows:
        fp_list = [r["item_fingerprint"] for r in catalog_rows]
        existing_fp: set[str] = set()
        if not skip_dedup:
            try:
                existing_fp = batch_query_field(
                    milvus, names_collection, "item_fingerprint", fp_list
                )
            except Exception as e:
                return {"errors": [f"名称表去重查询失败: {e}"], "is_success": False}

        new_items = [r for r in catalog_rows if r["item_fingerprint"] not in existing_fp]
        if existing_fp:
            warnings.append(f"已跳过 {len(existing_fp)} 条重复名称项（item_fingerprint）")

        if new_items:
            texts = [r["item_name"] for r in new_items]
            try:
                result = service.embed_dense_only(texts)

                if isinstance(result, EmbeddingError):
                    return {"errors": [f"名称表 Embedding 失败: {result.message}"], "is_success": False}

                name_vecs = result.embeddings
            except Exception as e:
                return {"errors": [f"名称表 Embedding 失败: {e}"], "is_success": False}
            if len(name_vecs) != len(new_items):
                return {
                    "errors": ["名称表向量条数与记录不一致"],
                    "is_success": False,
                }
            for r, vec in zip(new_items, name_vecs):
                catalog_inserts.append(
                    {
                        "item_name": r["item_name"],
                        "item_type": r["item_type"],
                        "item_fingerprint": r["item_fingerprint"],
                        "vector": vec,
                    }
                )
            try:
                res_n = milvus.insert(collection_name=names_collection, data=catalog_inserts)
                pkn = getattr(res_n, "primary_keys", None)
                catalog_ids = [str(x) for x in list(pkn)] if pkn is not None else []
            except Exception as e:
                return {"errors": [f"名称表 Milvus 写入失败: {e}"], "is_success": False}

    # --- 切片表（稠密 + 稀疏）---
    hashes = [r["content_hash"] for r in flat_rows]
    skipped_hashes: set[str] = set()
    if not skip_dedup:
        try:
            skipped_hashes = batch_query_field(
                milvus, chunks_collection, "content_hash", hashes
            )
        except Exception as e:
            return {"errors": [f"切片表去重查询失败: {e}"], "is_success": False}

    to_embed_docs: list[EduContent] = []
    to_embed_rows: list[dict[str, Any]] = []
    for doc, row in zip(documents, flat_rows):
        if row["content_hash"] in skipped_hashes:
            continue
        to_embed_docs.append(doc)
        to_embed_rows.append(row)

    if not to_embed_rows:
        wmsg = f"全部切片均已存在（content_hash 命中 {len(skipped_hashes)} 条），跳过切片写入"
        warnings.append(wmsg)
        out: dict[str, Any] = {
            "indexed_records": [],
            "vector_ids": [],
            "indexed_catalog_records": catalog_inserts,
            "catalog_vector_ids": catalog_ids,
            "is_success": True,
        }
        if warnings:
            out["warnings"] = warnings
        return out

    texts = [d.content for d in to_embed_docs]
    try:
        dense_result, sparse_result = service.embed_dense_sparse(texts)

        if isinstance(dense_result, EmbeddingError):
            return {"errors": [f"稠密向量生成失败: {dense_result.message}"], "is_success": False}

        if isinstance(sparse_result, EmbeddingError):
            return {"errors": [f"稀疏向量生成失败: {sparse_result.message}"], "is_success": False}

        dense_vecs = dense_result.embeddings
        sparse_vecs = sparse_result.embeddings
    except Exception as e:
        return {"errors": [f"双向量生成失败: {e}"], "is_success": False}

    if len(dense_vecs) != len(to_embed_rows) or len(sparse_vecs) != len(to_embed_rows):
        return {
            "errors": ["稠密/稀疏向量条数与切片不一致"],
            "is_success": False,
        }

    chunk_records: list[dict[str, Any]] = []
    content_trunc_count = 0
    fixed_scalar = {
        "content",
        "vector_dense",
        "vector_sparse",
        "course_name",
        "content_type",
        "content_hash",
        "source_file",
        "document_class",
        "question_type",
    }
    for doc, row, dv, sv in zip(to_embed_docs, to_embed_rows, dense_vecs, sparse_vecs):
        base: dict[str, Any] = {
            "content": row.get("content") or doc.content,
            "vector_dense": dv,
            "vector_sparse": sv,
            "course_name": row.get("course_name") or "",
            "content_type": row.get("content_type") or "",
            "content_hash": row["content_hash"],
            "source_file": row.get("source_file") or "",
            "document_class": row.get("document_class") or "",
            "question_type": row.get("question_type") or "",
        }
        for k, v in row.items():
            if k in fixed_scalar:
                continue
            base[k] = v
        base, did_trunc = truncate_content_field(base)
        if did_trunc:
            content_trunc_count += 1
        chunk_records.append(sanitize_milvus_row(base))

    try:
        res_c = milvus.insert(collection_name=chunks_collection, data=chunk_records)
        pkc = getattr(res_c, "primary_keys", None)
        chunk_ids = [str(x) for x in list(pkc)] if pkc is not None else []
    except Exception as e:
        return {"errors": [f"切片表 Milvus 写入失败: {e}"], "is_success": False}

    if skipped_hashes:
        warnings.append(f"已跳过 {len(skipped_hashes)} 条重复切片（content_hash 已存在）")
    if content_trunc_count:
        warnings.append(
            f"{content_trunc_count} 条切片的 content 超过 Milvus VarChar 上限（{MILVUS_VARCHAR_CONTENT_MAX}），"
            "已截断后写入；长文 syllabus 建议后续改为按章节多切片。"
        )

    # Always include catalog fields, even if empty
    out = {
        "indexed_records": chunk_records,
        "vector_ids": chunk_ids,
        "indexed_catalog_records": catalog_inserts,
        "catalog_vector_ids": catalog_ids,
        "is_success": True,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _merge_upstream_lists(state: ImportGraphState, patch: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph 对 TypedDict 状态按 **键整体替换**；本节点若只返回新的 ``errors`` / ``warnings``，
    会覆盖前面节点（file_router、分类器、抽取器）已写入的列表。此处把上游列表 **前置拼接**。
    """
    out = dict(patch)
    prior_e = list(state.get("errors") or [])
    prior_w = list(state.get("warnings") or [])
    new_e = list(out.get("errors") or [])
    if prior_e or new_e:
        out["errors"] = prior_e + new_e
    new_w = list(out.get("warnings") or [])
    if prior_w or new_w:
        out["warnings"] = prior_w + new_w
    return out


def vector_indexer_node(state: ImportGraphState) -> dict[str, Any]:
    """
    向量入库节点。

    **legacy**

    - ``MILVUS_COLLECTION``：默认 ``edu_knowledge_vectors_v1``，字段 ``vector``。
    - ``EMBEDDING_BACKEND``：默认 OpenAI 兼容 API。

    **v2（双表 + 混合向量）**

    - ``MILVUS_RAG_MODE=v2``
    - ``MILVUS_NAMES_COLLECTION`` / ``MILVUS_CHUNKS_COLLECTION``：默认 ``edu_knowledge_item_names_v1`` / ``edu_knowledge_chunks_v1``
    - 须 **本地 BGE-M3**：``BGE_M3_PATH`` 指向已存在目录，或 ``EMBEDDING_BACKEND=local_bge_m3`` / ``local``（见 ``utils.local_bge_client``）
    - ``MILVUS_SKIP_DEDUP``：``1`` 时跳过去重查询。
    """
    chunks = state.get("chunks") or []
    if not chunks:
        return merge_upstream_lists(
            state,
            {"warnings": ["没有检测到有效切片，跳过入库"], "is_success": True},
        )

    documents: list[EduContent] = []
    for ch in chunks:
        raw = ch.get("edu_content") if isinstance(ch, dict) else None
        if not raw:
            continue
        documents.append(EduContent.model_validate(raw))

    if not documents:
        return merge_upstream_lists(
            state,
            {"errors": ["提取出的 EduContent 对象为空"], "is_success": False},
        )

    # Load configuration from environment
    config = VectorIndexerConfig.from_env()

    flat_rows: list[dict[str, Any]] = []
    for doc in documents:
        flat = doc.to_flat_dict()
        flat["content_hash"] = content_fingerprint(doc)
        flat_rows.append(flat)

    api_doc_type = (state.get("api_doc_type") or "").strip()
    if api_doc_type:
        for fr in flat_rows:
            fr["doc_type"] = api_doc_type

    if config.rag_mode == "v2":
        return merge_upstream_lists(
            state,
            _vector_indexer_v2(
                documents,
                flat_rows,
                names_collection=config.v2_names_collection,
                chunks_collection=config.v2_chunks_collection,
                skip_dedup=config.skip_dedup,
            ),
        )

    return merge_upstream_lists(
        state,
        _vector_indexer_legacy(
            documents,
            flat_rows,
            collection_name=config.legacy_collection,
            skip_dedup=config.skip_dedup,
        ),
    )


__all__ = ["vector_indexer_node"]
