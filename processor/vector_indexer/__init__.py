"""Vector indexer package."""

from processor.vector_indexer.config import VectorIndexerConfig
from processor.vector_indexer.utils import (
    MILVUS_VARCHAR_CONTENT_MAX,
    batch_query_field,
    catalog_display_name,
    content_fingerprint,
    extract_catalog_items,
    item_fingerprint,
    merge_upstream_lists,
    sanitize_milvus_row,
    truncate_content_field,
)
from processor.vector_indexer.vector_indexer import vector_indexer_node

__all__ = [
    "VectorIndexerConfig",
    "vector_indexer_node",
    # Utility functions (now public API)
    "content_fingerprint",
    "item_fingerprint",
    "catalog_display_name",
    "extract_catalog_items",
    "truncate_content_field",
    "sanitize_milvus_row",
    "batch_query_field",
    "merge_upstream_lists",
    "MILVUS_VARCHAR_CONTENT_MAX",
]
