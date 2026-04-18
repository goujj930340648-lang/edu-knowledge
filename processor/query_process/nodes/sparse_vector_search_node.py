"""v2：仅稀疏向量检索（与稠密路、HyDE 并行）。"""

from __future__ import annotations

from typing import Any

from processor.query_process.base import BaseNode
from processor.query_process.embed_util import embed_query_for_chunks
from processor.query_state import QueryGraphState
from utils.milvus_search_edu import (
    course_name_filter_expr,
    rag_mode,
    sparse_search_chunks_v2,
)


class SparseVectorSearchNode(BaseNode):
    name = "sparse_vector_search_node"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        rq = (
            state.get("retrieval_query")
            or state.get("rewritten_query")
            or state.get("user_query")
            or ""
        ).strip()
        if not rq:
            return {}
        if rag_mode() != "v2":
            return {"sparse_embedding_chunks": []}
        confirmed = state.get("catalog_confirmed") or []
        expr = course_name_filter_expr([str(x) for x in confirmed if x])
        limit = self.config.embedding_search_limit
        try:
            _dense, sparse = embed_query_for_chunks(rq)
        except Exception as e:
            self.logger.warning("sparse 查询向量化失败: %s", e)
            return {"sparse_embedding_chunks": []}
        if sparse is None:
            self.logger.warning("v2 稀疏向量为空")
            return {"sparse_embedding_chunks": []}
        try:
            hits = sparse_search_chunks_v2(sparse, limit=limit, expr=expr)
        except Exception as e:
            self.logger.warning("sparse Milvus 检索失败: %s", e)
            return {"sparse_embedding_chunks": []}
        self.logger.debug("sparse hits: %d", len(hits))
        return {"sparse_embedding_chunks": hits}


__all__ = ["SparseVectorSearchNode"]
