"""
教育知识库 Milvus 检索封装：与 ``init_milvus`` / ``vector_indexer`` 字段名对齐。

- **v2**：``vector_dense`` + ``vector_sparse`` 混合检索（``WeightedRanker`` / ``RRFRanker``）。
- **legacy**：单字段 ``vector`` 稠密检索。

供查询图节点调用；连接与 ``utils.client.MilvusIndexerClient`` 使用相同 URI/TOKEN。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from utils.client import maybe_strip_proxy_for_milvus


def _uri_token() -> tuple[str, str | None]:
    uri = (
        os.environ.get("MILVUS_URI") or os.environ.get("MILVUS_URL") or "http://127.0.0.1:19530"
    ).strip()
    token = (os.environ.get("MILVUS_TOKEN") or "").strip() or None
    return uri, token


@lru_cache(maxsize=1)
def _ensure_connection() -> None:
    from pymilvus import connections

    maybe_strip_proxy_for_milvus()
    uri, token = _uri_token()
    if connections.has_connection("default"):
        return
    kw: dict[str, Any] = {"alias": "default", "uri": uri}
    if token:
        kw["token"] = token
    connections.connect(**kw)


def chunks_collection_name() -> str:
    return (
        os.environ.get("MILVUS_CHUNKS_COLLECTION")
        or os.environ.get("CHUNKS_COLLECTION")
        or "edu_knowledge_chunks_v1"
    ).strip() or "edu_knowledge_chunks_v1"


def legacy_collection_name() -> str:
    return (
        os.environ.get("MILVUS_COLLECTION") or "edu_knowledge_vectors_v1"
    ).strip() or "edu_knowledge_vectors_v1"


def names_collection_name() -> str:
    return (
        os.environ.get("MILVUS_NAMES_COLLECTION")
        or os.environ.get("ITEM_NAME_COLLECTION")
        or "edu_knowledge_item_names_v1"
    ).strip() or "edu_knowledge_item_names_v1"


def rag_mode() -> str:
    return os.environ.get("MILVUS_RAG_MODE", "legacy").strip().lower()


def course_name_filter_expr(course_names: list[str]) -> str | None:
    """``course_name in [...]``；空列表则不做标量过滤。"""
    if not course_names:
        return None
    escaped = [n.replace("\\", "\\\\").replace('"', '\\"') for n in course_names if n and str(n).strip()]
    if not escaped:
        return None
    return "course_name in [" + ",".join(f'"{e}"' for e in escaped) + "]"


def _format_hits(
    raw: Any,
    output_fields: list[str],
) -> list[dict[str, Any]]:
    """将 ``Collection.search/hybrid_search`` 单查询结果转为与 knowledge 一致的 dict 列表。"""
    out: list[dict[str, Any]] = []
    if not raw:
        return out
    try:
        sr0 = raw[0]
    except (IndexError, TypeError):
        return out
    for hit in sr0:
        entity: dict[str, Any] = {}
        if hasattr(hit, "entity") and hit.entity is not None:
            ent = hit.entity
            if hasattr(ent, "to_dict"):
                entity = ent.to_dict()
            elif isinstance(ent, dict):
                entity = dict(ent)
            else:
                for f in output_fields:
                    try:
                        entity[f] = ent.get(f)
                    except Exception:
                        pass
        else:
            continue
        out.append(
            {
                "id": getattr(hit, "id", None),
                "distance": getattr(hit, "distance", None),
                "entity": entity,
            }
        )
    return out


# v2 切片表：与 ``init_milvus._create_chunks_collection`` 固定列一致（其余走动态字段时可再扩展）
CHUNKS_V2_OUTPUT_FIELDS: list[str] = [
    "content",
    "content_hash",
    "course_name",
    "content_type",
    "source_file",
    "document_class",
    "question_type",
    "doc_type",
]

# legacy：动态列常用子集（按实际入库 ``to_flat_dict`` 字段）
CHUNKS_LEGACY_OUTPUT_FIELDS: list[str] = [
    "content",
    "content_hash",
    "course_name",
    "content_type",
    "source_file",
    "chapter_name",
    "project_name",
    "bank_name",
    "document_class",
    "question_type",
]


def hybrid_search_chunks_v2(
    dense: list[float],
    sparse: dict[int, float],
    *,
    limit: int = 8,
    expr: str | None = None,
    output_fields: list[str] | None = None,
    use_rrf: bool = False,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """v2 切片表：稠密 + 稀疏混合检索。"""
    from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker, utility

    _ensure_connection()
    name = chunks_collection_name()
    if not utility.has_collection(name):
        return []
    col = Collection(name)
    col.load()
    fields = output_fields or CHUNKS_V2_OUTPUT_FIELDS
    dense_req = AnnSearchRequest(
        data=[dense],
        anns_field="vector_dense",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=limit,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse],
        anns_field="vector_sparse",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=limit,
        expr=expr,
    )
    ranker = RRFRanker(k=60) if use_rrf else WeightedRanker(dense_weight, sparse_weight)
    raw = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=ranker,
        limit=limit,
        output_fields=fields,
    )
    return _format_hits(raw, fields)


def dense_search_chunks_v2(
    dense: list[float],
    *,
    limit: int = 8,
    expr: str | None = None,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """v2 切片表：仅 ``vector_dense`` 检索（与 HyDE / 稀疏路并行后 RRF 融合）。"""
    from pymilvus import Collection, utility

    _ensure_connection()
    name = chunks_collection_name()
    if not utility.has_collection(name):
        return []
    col = Collection(name)
    col.load()
    fields = output_fields or CHUNKS_V2_OUTPUT_FIELDS
    raw = col.search(
        data=[dense],
        anns_field="vector_dense",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=limit,
        expr=expr,
        output_fields=fields,
    )
    return _format_hits(raw, fields)


def sparse_search_chunks_v2(
    sparse: dict[int, float],
    *,
    limit: int = 8,
    expr: str | None = None,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """v2 切片表：仅 ``vector_sparse`` 检索。"""
    from pymilvus import Collection, utility

    _ensure_connection()
    name = chunks_collection_name()
    if not utility.has_collection(name):
        return []
    col = Collection(name)
    col.load()
    fields = output_fields or CHUNKS_V2_OUTPUT_FIELDS
    raw = col.search(
        data=[sparse],
        anns_field="vector_sparse",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=limit,
        expr=expr,
        output_fields=fields,
    )
    return _format_hits(raw, fields)


def dense_search_legacy_chunks(
    vector: list[float],
    *,
    limit: int = 8,
    expr: str | None = None,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """legacy 单表：``vector`` 列检索。"""
    from pymilvus import Collection, utility

    _ensure_connection()
    name = legacy_collection_name()
    if not utility.has_collection(name):
        return []
    col = Collection(name)
    col.load()
    fields = output_fields or CHUNKS_LEGACY_OUTPUT_FIELDS
    raw = col.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=limit,
        expr=expr,
        output_fields=fields,
    )
    return _format_hits(raw, fields)


def dense_search_names_batch(
    dense_vectors: list[list[float]],
    *,
    top_k: int = 5,
    output_fields: list[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """
    名称表：仅稠密向量，批量检索。
    返回与 ``dense_vectors`` 等长的命中列表列表。
    """
    from pymilvus import Collection, utility

    if not dense_vectors:
        return []
    _ensure_connection()
    name = names_collection_name()
    if not utility.has_collection(name):
        return [[] for _ in dense_vectors]
    col = Collection(name)
    col.load()
    fields = output_fields or ["item_name", "item_type", "item_fingerprint"]
    raw = col.search(
        data=dense_vectors,
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=fields,
    )
    batch: list[list[dict[str, Any]]] = []
    for sr in raw:
        hits: list[dict[str, Any]] = []
        for hit in sr:
            entity: dict[str, Any] = {}
            if hasattr(hit, "entity") and hit.entity is not None:
                ent = hit.entity
                entity = ent.to_dict() if hasattr(ent, "to_dict") else dict(ent)
            hits.append(
                {
                    "id": getattr(hit, "id", None),
                    "distance": getattr(hit, "distance", None),
                    "entity": entity,
                }
            )
        batch.append(hits)
    return batch


__all__ = [
    "CHUNKS_LEGACY_OUTPUT_FIELDS",
    "CHUNKS_V2_OUTPUT_FIELDS",
    "chunks_collection_name",
    "course_name_filter_expr",
    "dense_search_legacy_chunks",
    "dense_search_names_batch",
    "hybrid_search_chunks_v2",
    "legacy_collection_name",
    "names_collection_name",
    "rag_mode",
]
