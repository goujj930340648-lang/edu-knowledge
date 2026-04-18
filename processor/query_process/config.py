"""查询流程配置（环境变量）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class QueryConfig:
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )
    rerank_max_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10")))
    rerank_min_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3")))
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    rrf_max_results: int = field(default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "12")))
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "8"))
    )
    hyde_search_limit: int = field(default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "8")))
    catalog_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("CATALOG_HIGH_CONFIDENCE", "0.7"))
    )
    catalog_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("CATALOG_MID_CONFIDENCE", "0.45"))
    )
    catalog_score_gap: float = field(
        default_factory=lambda: float(os.getenv("CATALOG_SCORE_GAP", "0.15"))
    )
    catalog_max_options: int = field(
        default_factory=lambda: int(os.getenv("CATALOG_MAX_OPTIONS", "5"))
    )


@lru_cache(maxsize=1)
def get_query_config() -> QueryConfig:
    return QueryConfig()


__all__ = ["QueryConfig", "get_query_config"]
