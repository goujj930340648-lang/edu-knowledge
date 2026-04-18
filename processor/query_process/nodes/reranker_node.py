"""BGE Reranker + 动态截断（与 knowledge 思路一致）。"""

from __future__ import annotations

from typing import Any

from processor.query_process.base import BaseNode
from processor.query_state import QueryGraphState, RetrievedHit

MIN_RRF_ITEM_LENGTH = 2


class RerankerNode(BaseNode):
    name = "reranker_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        q = (
            state.get("retrieval_query")
            or state.get("rewritten_query")
            or state.get("user_query")
            or ""
        ).strip()

        docs = self._collect_docs(state)
        refined = self._refine_rank(q, docs)
        cutoff = self._apply_cliff_truncation(
            refined,
            self.config.rerank_max_top_k,
            self.config.rerank_min_top_k,
        )
        state["reranked_docs"] = cutoff

        hits: list[RetrievedHit] = []
        for d in cutoff:
            hits.append(
                {
                    "content": d.get("content", ""),
                    "score": float(d.get("score", 0.0)),
                    "source_file": d.get("source_file") or "",
                    "course_name": d.get("course_name"),
                    "chapter_name": d.get("chapter_name"),
                    "project_name": d.get("project_name"),
                    "content_type": d.get("content_type") or "",
                    "chunk_id": d.get("chunk_id"),
                }
            )
        state["retrieved_hits"] = hits
        return state

    def _collect_docs(self, state: QueryGraphState) -> list[dict[str, Any]]:
        final_docs: list[dict[str, Any]] = []
        rrf_chunks = state.get("rrf_chunks") or []
        for item in rrf_chunks:
            hit, rrf_score = self._unpack_rrf_item(item)
            if not hit:
                continue
            entity = hit.get("entity")
            if not isinstance(entity, dict):
                continue
            content = str(entity.get("content") or "").strip()
            if not content:
                continue
            cid = str(entity.get("content_hash") or entity.get("id") or "")
            final_docs.append(
                {
                    "content": content,
                    "title": str(entity.get("course_name") or entity.get("source_file") or cid),
                    "chunk_id": cid,
                    "source_file": entity.get("source_file") or "",
                    "course_name": entity.get("course_name"),
                    "chapter_name": entity.get("chapter_name"),
                    "project_name": entity.get("project_name"),
                    "bank_name": entity.get("bank_name"),
                    "content_type": entity.get("content_type") or "",
                    "rrf_score": rrf_score,
                }
            )
        return final_docs

    @staticmethod
    def _unpack_rrf_item(item: Any) -> tuple[dict[str, Any] | None, float | None]:
        if isinstance(item, tuple) and len(item) >= MIN_RRF_ITEM_LENGTH:
            hit, score = item[0], item[1]
            if isinstance(hit, dict):
                s = float(score) if isinstance(score, (int, float)) else None
                return hit, s
            return None, None
        if isinstance(item, dict):
            return item, None
        return None, None

    def _refine_rank(self, user_query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return []
        if not user_query:
            return [{**d, "score": 0.0} for d in docs]

        try:
            from utils.local_bge_reranker import get_local_bge_reranker

            reranker = get_local_bge_reranker()
        except Exception as e:
            self.logger.warning("Reranker 不可用，跳过重排: %s", e)
            return [{**d, "score": float(d.get("rrf_score") or 0.0)} for d in docs]

        pairs = [(user_query, d.get("content", "")) for d in docs]
        try:
            try:
                raw_scores = reranker.compute_score(pairs, normalize=True)
            except TypeError:
                raw_scores = reranker.compute_score(pairs)
        except Exception as e:
            # 常见：transformers meta 权重 + FlagReranker ``model.to(device)`` 失败；降级为 RRF 顺序
            self.logger.warning("Reranker compute_score 失败，使用 RRF 分数降级: %s", e)
            return [{**d, "score": float(d.get("rrf_score") or 0.0)} for d in docs]
        scores = self._normalize_scores(raw_scores, len(docs))
        ranked = [{**d, "score": s} for d, s in zip(docs, scores, strict=True)]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    @staticmethod
    def _normalize_scores(raw: Any, expected: int) -> list[float]:
        try:
            import numpy as np
        except ImportError:
            if isinstance(raw, (int, float)):
                return [float(raw)] * expected if expected else []
            if isinstance(raw, (list, tuple)):
                return [float(x) for x in raw][:expected]
            return [0.0] * expected

        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            return [float(raw.item())] * expected if expected == 1 else [0.0] * expected
        if np.isscalar(raw) or not isinstance(raw, (list, tuple, np.ndarray)):
            return [float(raw)] * expected if expected == 1 else [0.0] * expected
        arr = np.asarray(raw)
        out = [float(x) for x in np.ravel(arr)]
        if len(out) != expected:
            raise ValueError(f"rerank 分数条数 {len(out)} 与文档 {expected} 不一致")
        return out

    def _apply_cliff_truncation(
        self,
        refine_docs: list[dict[str, Any]],
        max_top_k: int,
        min_top_k: int,
    ) -> list[dict[str, Any]]:
        if not refine_docs:
            return []
        from utils.local_bge_reranker import cliff_truncation_count

        scores = [float(doc.get("score", 0.0)) for doc in refine_docs]
        n = cliff_truncation_count(
            scores,
            max_top_k=max_top_k,
            min_top_k=min_top_k,
            gap_ratio=self.config.rerank_gap_ratio,
            gap_abs=self.config.rerank_gap_abs,
        )
        return refine_docs[:n]


__all__ = ["RerankerNode"]
