"""
Vector indexer LangGraph node.

This module provides a minimal glue layer between LangGraph and the vector indexing system.
It follows the 5-step pattern:
1. Extract chunks from state
2. Create VectorIndexerConfig
3. Create indexer via factory
4. Execute indexing
5. Update and return state

The node properly handles errors/warnings accumulation from upstream nodes.
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
    """LangGraph node for vector indexing.

    This node indexes document chunks into Milvus vector database.
    Supports two modes:
    - **legacy** (default): Single collection, dense vectors (OpenAI-compatible)
    - **v2**: Dual collections (names + chunks), hybrid vectors (BGE-M3)

    Environment Variables (legacy mode):
        MILVUS_COLLECTION: Collection name (default: edu_knowledge_vectors_v1)
        EMBEDDING_BACKEND: Embedding service (default: openai compatible)

    Environment Variables (v2 mode):
        MILVUS_RAG_MODE: Set to "v2" for dual collection mode
        MILVUS_NAMES_COLLECTION: Names collection (default: edu_knowledge_item_names_v1)
        MILVUS_CHUNKS_COLLECTION: Chunks collection (default: edu_knowledge_chunks_v1)
        BGE_M3_PATH: Path to local BGE-M3 model (required for v2)
        EMBEDDING_BACKEND: Set to "local_bge_m3" or "local"
        MILVUS_SKIP_DEDUP: Set to "1" to skip deduplication

    Args:
        state: LangGraph state containing:
            - chunks: List of chunk dictionaries with edu_content field
            - errors: Existing errors from upstream nodes (preserved)
            - warnings: Existing warnings from upstream nodes (preserved)
            - api_doc_type: Optional document type to tag in records

    Returns:
        Dictionary with:
            - indexed_records: List of inserted chunk records
            - vector_ids: List of vector IDs from Milvus
            - indexed_catalog_records: List of catalog records (v2 mode only)
            - catalog_vector_ids: List of catalog vector IDs (v2 mode only)
            - errors: Appended errors from indexing (preserves upstream)
            - warnings: Appended warnings from indexing (preserves upstream)
            - is_success: Whether indexing succeeded

    Example:
        >>> state = {"chunks": [{"edu_content": {...}}]}
        >>> result = vector_indexer_node(state)
        >>> print(result["indexed_records"])
    """
    # Step 1: Extract chunks from state
    chunks = state.get("chunks") or []
    if not chunks:
        return merge_upstream_lists(
            state,
            {"warnings": ["没有检测到有效切片，跳过入库"], "is_success": True},
        )

    # Step 2: Create VectorIndexerConfig from environment
    config = VectorIndexerConfig.from_env()

    # Step 3: Extract and validate EduContent documents
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

    # Prepare flat rows with content fingerprints
    flat_rows: list[dict[str, Any]] = []
    for doc in documents:
        flat = doc.to_flat_dict()
        flat["content_hash"] = content_fingerprint(doc)
        flat_rows.append(flat)

    # Add document type if specified
    api_doc_type = (state.get("api_doc_type") or "").strip()
    if api_doc_type:
        for fr in flat_rows:
            fr["doc_type"] = api_doc_type

    # Step 4: Execute indexing via strategy pattern
    indexer = create_indexer(config)
    result: IndexerResult = indexer.index(documents, flat_rows)

    # Step 5: Convert IndexerResult to LangGraph state format
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

    # Merge with upstream state (preserves existing errors/warnings)
    return merge_upstream_lists(state, output)


__all__ = ["vector_indexer_node"]
