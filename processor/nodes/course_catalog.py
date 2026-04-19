"""
课程 / 项目 / 题库名对齐（对应 knowledge 的 ``ItemNameConfirmedNode``，教育领域语义）。
"""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any

from prompts.query_prompt import CATALOG_EXTRACT_SYSTEM, CATALOG_EXTRACT_USER
from processor.utils.base import BaseNode
from processor.query_state import QueryGraphState
from utils.client import get_llm_client
from utils.local_bge_client import should_use_local_bge_embedding
from utils.milvus_search_edu import dense_search_names_batch


def _strip_json_fence(s: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip())
    return re.sub(r"\s*```$", "", cleaned)


class CourseCatalogNode(BaseNode):
    name = "course_catalog_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            state["error"] = "user_query 为空"
            return state

        history_hint = self._history_hint(state)
        extraction = self._llm_extract(user_query, history_hint)
        raw_names = extraction.get("entity_names") or []
        entity_names = [str(x).strip() for x in raw_names if isinstance(x, str) and str(x).strip()]
        rewritten = extraction.get("rewritten_query") or user_query
        if not isinstance(rewritten, str) or not rewritten.strip():
            rewritten = user_query

        state["entity_names"] = entity_names
        state["rewritten_query"] = rewritten.strip()
        state["retrieval_query"] = state["rewritten_query"]

        confirmed: list[str] = []
        options: list[str] = []
        if entity_names and should_use_local_bge_embedding():
            confirmed, options = self._align_names(entity_names)
        elif entity_names:
            self.logger.info("未启用本地 BGE，跳过名称表对齐，检索将不按 course_name 过滤")

        state["catalog_confirmed"] = confirmed
        state["catalog_options"] = options
        self._decide(confirmed, options, state)
        return state

    def _history_hint(self, state: QueryGraphState) -> str:
        msgs = state.get("messages")
        if not msgs:
            return "（无）"
        parts: list[str] = []
        for m in msgs[-6:]:
            if isinstance(m, dict):
                role = m.get("role", "")
                text = (m.get("content") or m.get("text") or "")[:500]
                if text:
                    parts.append(f"{role}:{text}")
            else:
                parts.append(str(m)[:500])
        return "\n".join(parts) if parts else "（无）"

    def _llm_extract(self, user_query: str, history_hint: str) -> dict[str, Any]:
        out: dict[str, Any] = {"entity_names": [], "rewritten_query": user_query}
        try:
            llm = get_llm_client()
            user_p = CATALOG_EXTRACT_USER.format(
                user_query=user_query,
                history_hint=history_hint,
            )
            prompt = f"{CATALOG_EXTRACT_SYSTEM}\n\n{user_p}"
            raw = llm.chat(prompt=prompt, json_mode=True)
            data = json.loads(_strip_json_fence(raw))
        except (JSONDecodeError, RuntimeError, OSError) as e:
            self.logger.warning("课程名抽取失败，使用原问题: %s", e)
            return out

        en = data.get("entity_names")
        if isinstance(en, list):
            out["entity_names"] = [str(x) for x in en if isinstance(x, str)]
        rq = data.get("rewritten_query")
        if isinstance(rq, str) and rq.strip():
            out["rewritten_query"] = rq.strip()
        return out

    def _align_names(self, entity_names: list[str]) -> tuple[list[str], list[str]]:
        try:
            from processor.vector_indexer.embedding_service import (
                EmbeddingError,
                get_embedding_service,
            )

            service = get_embedding_service()
            result = service.embed_dense_only(entity_names)

            if isinstance(result, EmbeddingError):
                self.logger.warning("名称向量化失败: %s", result.message)
                return [], []

            vecs = result.embeddings
        except Exception as e:
            self.logger.warning("名称向量化失败: %s", e)
            return [], []

        try:
            batch = dense_search_names_batch(
                vecs,
                top_k=5,
                output_fields=["item_name", "item_type"],
            )
        except Exception as e:
            self.logger.warning("名称表检索失败: %s", e)
            return [], []

        confirmed: set[str] = set()
        options: set[str] = set()
        high, mid = self.config.catalog_high_confidence, self.config.catalog_mid_confidence
        gap = self.config.catalog_score_gap
        max_opt = self.config.catalog_max_options

        for i, name in enumerate(entity_names):
            if i >= len(batch):
                break
            matches = batch[i]
            if not matches:
                continue
            matches = sorted(matches, key=lambda x: float(x.get("distance") or 0.0), reverse=True)
            top = matches[0]
            top_score = float(top.get("distance") or 0.0)
            ent = top.get("entity") or {}
            top_item = str(ent.get("item_name") or "").strip()
            if not top_item:
                continue

            if top_score >= high:
                if len(matches) > 1:
                    second = float(matches[1].get("distance") or 0.0)
                    if top_score - second < gap:
                        for m in matches[:max_opt]:
                            sc = float(m.get("distance") or 0.0)
                            if sc >= mid:
                                en = m.get("entity") or {}
                                nm = str(en.get("item_name") or "").strip()
                                if nm:
                                    options.add(nm)
                    else:
                        confirmed.add(top_item)
                else:
                    confirmed.add(top_item)
            elif top_score >= mid:
                for m in matches[:max_opt]:
                    sc = float(m.get("distance") or 0.0)
                    if sc >= mid:
                        en = m.get("entity") or {}
                        nm = str(en.get("item_name") or "").strip()
                        if nm:
                            options.add(nm)

        final_options = [o for o in options if o not in confirmed]
        return list(confirmed), final_options[:max_opt]

    def _decide(
        self,
        confirmed: list[str],
        options: list[str],
        state: QueryGraphState,
    ) -> None:
        if options:
            prefix = ""
            if confirmed:
                prefix = f"已匹配到「{'、'.join(confirmed)}」。关于其余范围，"
            state["answer"] = (
                f"{prefix}我不确定您指的是哪一门课程或题库？可选：{'、'.join(options)} 。"
                f"请回复更准确的课程名或题库名后再问。"
            )
            return
        if confirmed:
            state["answer"] = None
            return
        # 抽取了名称但未命中索引：不拦截，下游按全库检索（仅用改写问句）
        state["answer"] = None


__all__ = ["CourseCatalogNode"]
