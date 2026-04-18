"""v2：仅稠密向量检索（与稀疏路、HyDE 并行）。"""

from __future__ import annotations

from typing import Any

from processor.query_process.base import BaseNode
from processor.query_process.embed_util import embed_query_for_chunks
from processor.query_state import QueryGraphState
from utils.milvus_search_edu import course_name_filter_expr, dense_search_chunks_v2, rag_mode


class DenseVectorSearchNode(BaseNode):
    name = "dense_vector_search_node"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        rq = (
            state.get("retrieval_query")
            or state.get("rewritten_query")
            or state.get("user_query")
            or ""
        ).strip()
        if not rq:
            return {}
        confirmed = state.get("catalog_confirmed") or []
        expr = course_name_filter_expr([str(x) for x in confirmed if x])
        limit = self.config.embedding_search_limit
        try:
            dense, _sparse = embed_query_for_chunks(rq)
        except Exception as e:
            self.logger.warning("dense 查询向量化失败: %s", e)
            return {"dense_embedding_chunks": []}

        if rag_mode() != "v2":
            return {}
        try:
            hits = dense_search_chunks_v2(dense, limit=limit, expr=expr)
        except Exception as e:
            self.logger.warning("dense Milvus 检索失败: %s", e)
            return {"dense_embedding_chunks": []}

        self.logger.debug("dense hits: %d", len(hits))
        return {"dense_embedding_chunks": hits}


__all__ = ["DenseVectorSearchNode"]
