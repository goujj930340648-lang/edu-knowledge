"""两路检索结果 RRF 融合（与 knowledge ``RrfMergeNode`` 同构）。"""

from __future__ import annotations

from typing import Any

from processor.query_process.base import BaseNode
from processor.query_state import QueryGraphState


def _chunk_key(entity: dict[str, Any]) -> str | None:
    if not entity:
        return None
    h = entity.get("content_hash")
    if h is not None and str(h).strip():
        return f"h:{h}"
    pk = entity.get("id")
    if pk is not None:
        return f"id:{pk}"
    return None


class RrfMergeNode(BaseNode):
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        dense_c = self._validate_search_result(state.get("dense_embedding_chunks") or [])
        sparse_c = self._validate_search_result(state.get("sparse_embedding_chunks") or [])
        embedding_chunks = self._validate_search_result(state.get("embedding_chunks", []))
        hyde_chunks = self._validate_search_result(state.get("hyde_embedding_chunks", []))
        if dense_c or sparse_c:
            rrf_input: list[tuple[list[dict[str, Any]], float]] = [
                (dense_c, 1.0),
                (sparse_c, 1.0),
                (hyde_chunks, 1.0),
            ]
        else:
            rrf_input = [(embedding_chunks, 1.0), (hyde_chunks, 1.0)]
        merged = self._merge_rrf_docs(rrf_input, self.config.rrf_k, self.config.rrf_max_results)
        state["rrf_chunks"] = merged
        return state

    def _merge_rrf_docs(
        self,
        rrf_input: list[tuple[list[dict[str, Any]], float]],
        rrf_k: int,
        rrf_max_results: int,
    ) -> list[tuple[dict[str, Any], float]]:
        chunk_score: dict[str, float] = {}
        chunk_data: dict[str, dict[str, Any]] = {}
        for search_result, weight in rrf_input:
            for rank, hit in enumerate(search_result, 1):
                entity = hit.get("entity")
                if not isinstance(entity, dict):
                    continue
                key = _chunk_key(entity)
                if not key:
                    continue
                chunk_score[key] = chunk_score.get(key, 0.0) + weight / (rrf_k + rank)
                chunk_data.setdefault(key, hit)
        ranked = sorted(
            [(chunk_data[k], chunk_score[k]) for k in chunk_score],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:rrf_max_results] if rrf_max_results else ranked

    def _validate_search_result(self, search_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not search_result:
            return []
        validated: list[dict[str, Any]] = []
        for result in search_result:
            if not result or not isinstance(result, dict):
                continue
            entity = result.get("entity")
            if not isinstance(entity, dict):
                continue
            if _chunk_key(entity) is None:
                continue
            validated.append(result)
        return validated


__all__ = ["RrfMergeNode"]
