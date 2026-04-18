"""查询侧向量：与入库时同一套 Embedding 后端对齐。"""

from __future__ import annotations

from utils.client import get_embedding_client
from utils.local_bge_client import get_local_bge_client, should_use_local_bge_embedding
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
    if rag_mode() == "v2" and should_use_local_bge_embedding():
        bge = get_local_bge_client()
        dense, sparse = bge.embed_documents_dense_sparse([t])
        return dense[0], sparse[0]
    if should_use_local_bge_embedding():
        dense = get_local_bge_client().embed_documents_dense_only([t])
        return dense[0], None
    emb = get_embedding_client().embed_documents([t])
    return emb[0], None


__all__ = ["embed_query_for_chunks"]
