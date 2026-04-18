"""HyDE：假设文档 + 同构检索。"""

from __future__ import annotations

from typing import Any

from prompts.query_prompt import HYDE_SYSTEM_TEMPLATE, HYDE_USER_TEMPLATE
from processor.query_process.base import BaseNode
from processor.query_process.embed_util import embed_query_for_chunks
from processor.query_state import QueryGraphState
from utils.client import get_llm_client
from utils.milvus_search_edu import (
    course_name_filter_expr,
    dense_search_legacy_chunks,
    hybrid_search_chunks_v2,
    rag_mode,
)


class HyDeVectorSearchNode(BaseNode):
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        rq = (state.get("retrieval_query") or state.get("rewritten_query") or state.get("user_query") or "").strip()
        if not rq:
            return {}

        catalog = state.get("catalog_confirmed") or []
        catalog_hint = "、".join(str(x) for x in catalog if x) or "（未限定）"

        try:
            llm = get_llm_client()
            user_p = HYDE_USER_TEMPLATE.format(
                catalog_hint=catalog_hint,
                rewritten_query=rq,
            )
            prompt = f"{HYDE_SYSTEM_TEMPLATE}\n\n{user_p}"
            hy_doc = llm.chat(prompt=prompt, json_mode=False).strip()
        except Exception as e:
            self.logger.warning("HyDE LLM 失败: %s", e)
            return {}

        if not hy_doc or len(hy_doc) < 20:
            return {}

        confirmed = state.get("catalog_confirmed") or []
        expr = course_name_filter_expr([str(x) for x in confirmed if x])
        limit = self.config.hyde_search_limit

        try:
            dense, sparse = embed_query_for_chunks(hy_doc)
        except Exception as e:
            self.logger.warning("HyDE 向量化失败: %s", e)
            return {}

        mode = rag_mode()
        try:
            if mode == "v2":
                if sparse is None:
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
            self.logger.warning("HyDE Milvus 检索失败: %s", e)
            return {}

        self.logger.debug("hyde hits: %d", len(hits))
        return {"hyde_embedding_chunks": hits}


__all__ = ["HyDeVectorSearchNode"]
