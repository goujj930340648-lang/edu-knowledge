"""查询侧向量：与入库时同一套 Embedding 后端对齐。"""

from __future__ import annotations

from processor.vector_indexer.embedding_service import (
    EmbeddingError,
    EmbeddingResult,
    get_embedding_service,
)
from utils.milvus_search_edu import rag_mode


def embed_query_for_chunks(text: str) -> tuple[list[float], dict[int, float] | None]:
    """
    返回 (稠密向量, 稀疏向量或 None)。
    - ``MILVUS_RAG_MODE=v2`` 且本地 BGE-M3：稠密 + 稀疏。
    - 否则：仅稠密（legacy 单表或 OpenAI Embedding）。
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("empty query text")

    service = get_embedding_service()

    if rag_mode() == "v2":
        # v2 mode: try dense + sparse
        dense_result, sparse_result = service.embed_dense_sparse([t])

        if isinstance(dense_result, EmbeddingError):
            raise RuntimeError(f"Dense embedding failed: {dense_result.message}")

        if isinstance(sparse_result, EmbeddingError):
            # Sparse not supported or failed, fall back to dense only
            return dense_result.embeddings[0], None

        return dense_result.embeddings[0], sparse_result.embeddings[0]

    # Legacy mode: dense only
    result = service.embed_documents([t])

    if isinstance(result, EmbeddingError):
        raise RuntimeError(f"Embedding failed: {result.message}")

    return result.embeddings[0], None


__all__ = ["embed_query_for_chunks"]
