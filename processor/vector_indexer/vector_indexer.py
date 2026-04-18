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
from processor.vector_indexer.indexer import (
    IndexerResult,
    create_indexer,
)
from processor.vector_indexer.utils import (
    content_fingerprint,
    merge_upstream_lists,
)


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

    # Use strategy pattern indexer
    indexer = create_indexer(config)
    result = indexer.index(documents, flat_rows)

    # Convert IndexerResult to LangGraph state format
    output: dict[str, Any] = {
        "indexed_records": result.indexed_records,
        "vector_ids": result.vector_ids,
        "is_success": result.is_success,
    }

    # Add v2-specific fields if present
    if result.indexed_catalog_records or result.catalog_vector_ids:
        output["indexed_catalog_records"] = result.indexed_catalog_records
        output["catalog_vector_ids"] = result.catalog_vector_ids

    # Add warnings and errors if present
    if result.warnings:
        output["warnings"] = result.warnings
    if result.errors:
        output["errors"] = result.errors

    # Merge with upstream state
    return merge_upstream_lists(state, output)


__all__ = ["vector_indexer_node"]
