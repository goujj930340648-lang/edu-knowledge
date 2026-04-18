"""检索列表 RRF 融合（与 ``RrfMergeNode`` 同构，供 API 检索层复用）。"""

from __future__ import annotations

from typing import Any


def chunk_entity_key(entity: dict[str, Any]) -> str | None:
    if not entity:
        return None
    h = entity.get("content_hash")
    if h is not None and str(h).strip():
        return f"h:{h}"
    pk = entity.get("id")
    if pk is not None:
        return f"id:{pk}"
    return None


def rrf_merge_hits(
    ranked_lists: list[tuple[list[dict[str, Any]], float]],
    *,
    rrf_k: int = 60,
    max_results: int = 12,
) -> list[dict[str, Any]]:
    """
    ``ranked_lists`` 每项为 (Milvus 风格 hit 列表, 权重)；返回按 RRF 分排序的 hit 列表（截断到 ``max_results``）。
    """
    chunk_score: dict[str, float] = {}
    chunk_data: dict[str, dict[str, Any]] = {}
    for search_result, weight in ranked_lists:
        for rank, hit in enumerate(search_result, 1):
            if not hit or not isinstance(hit, dict):
                continue
            entity = hit.get("entity")
            if not isinstance(entity, dict):
                continue
            key = chunk_entity_key(entity)
            if not key:
                continue
            chunk_score[key] = chunk_score.get(key, 0.0) + weight / (rrf_k + rank)
            chunk_data.setdefault(key, hit)
    scored: list[tuple[dict[str, Any], float]] = [
        (chunk_data[k], chunk_score[k]) for k in chunk_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out = [h for h, _ in scored]
    return out[:max_results] if max_results else out


__all__ = ["chunk_entity_key", "rrf_merge_hits"]
