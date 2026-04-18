"""主查询向量检索：v2 混合检索或 legacy 稠密检索。"""

from __future__ import annotations

from typing import Any

from processor.query_process.base import BaseNode
from processor.query_process.embed_util import embed_query_for_chunks
from processor.query_state import QueryGraphState
from utils.milvus_search_edu import (
    course_name_filter_expr,
    dense_search_legacy_chunks,
    hybrid_search_chunks_v2,
    rag_mode,
)


class HybridVectorSearchNode(BaseNode):
    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        rq = (state.get("retrieval_query") or state.get("rewritten_query") or state.get("user_query") or "").strip()
        if not rq:
            return {}
        confirmed = state.get("catalog_confirmed") or []
        expr = course_name_filter_expr([str(x) for x in confirmed if x])

        limit = self.config.embedding_search_limit
        try:
            dense, sparse = embed_query_for_chunks(rq)
        except Exception as e:
            self.logger.warning("查询向量化失败: %s", e)
            return {}

        mode = rag_mode()
        try:
            if mode == "v2":
                if sparse is None:
                    self.logger.warning("v2 模式需要稠密+稀疏向量，请检查 BGE-M3 与 EMBEDDING_BACKEND")
                    return {}
                hits = hybrid_search_chunks_v2(
                    dense,
                    sparse,
                    limit=limit,
                    expr=expr,
                    use_rrf=False,
                    dense_weight=0.5,
                    sparse_weight=0.5,
                )
            else:
                hits = dense_search_legacy_chunks(dense, limit=limit, expr=expr)
        except Exception as e:
            self.logger.warning("Milvus 检索失败: %s", e)
            return {}

        self.logger.debug("hybrid hits: %d", len(hits))
        return {"embedding_chunks": hits}


__all__ = ["HybridVectorSearchNode"]
