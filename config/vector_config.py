"""Vector indexer configuration module.

This module encapsulates all configuration for the vector_indexer component,
including Milvus collection names, RAG mode, and embedding backend settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _env_local_bge_path_ok() -> bool:
    """若 BGE_M3_PATH 指向已存在目录，则可用本地 BGE-M3（与 env.example 说明一致）。"""
    raw = (os.environ.get("BGE_M3_PATH") or "").strip()
    if not raw:
        return False
    try:
        return Path(raw).is_dir()
    except OSError:
        return False


@dataclass(frozen=True)
class VectorIndexerConfig:
    """Configuration for vector indexer.

    Attributes:
        rag_mode: RAG mode - 'legacy' (single collection) or 'v2' (dual tables with hybrid vectors)
        legacy_collection: Collection name for legacy mode (default: 'edu_knowledge_vectors_v1')
        v2_names_collection: Collection name for v2 mode names table (default: 'edu_knowledge_item_names_v1')
        v2_chunks_collection: Collection name for v2 mode chunks table (default: 'edu_knowledge_chunks_v1')
        skip_dedup: Skip deduplication queries when True (useful for bulk imports)
        embedding_backend: Embedding backend to use

    Example:
        >>> config = VectorIndexerConfig.from_env()
        >>> if config.rag_mode == "v2":
        ...     print(f"Using v2 mode with {config.v2_names_collection}")
    """

    rag_mode: Literal["legacy", "v2"]
    legacy_collection: str
    v2_names_collection: str
    v2_chunks_collection: str
    skip_dedup: bool
    embedding_backend: str

    @classmethod
    def from_env(cls) -> "VectorIndexerConfig":
        """Create configuration from environment variables.

        Environment Variables:
            MILVUS_RAG_MODE: RAG mode ('legacy' or 'v2', default: 'legacy')
            MILVUS_COLLECTION: Legacy mode collection name (default: 'edu_knowledge_vectors_v1')
            MILVUS_NAMES_COLLECTION: v2 mode names collection (default: 'edu_knowledge_item_names_v1')
            MILVUS_CHUNKS_COLLECTION: v2 mode chunks collection (default: 'edu_knowledge_chunks_v1')
            ITEM_NAME_COLLECTION: Alternative name for MILVUS_NAMES_COLLECTION (deprecated)
            CHUNKS_COLLECTION: Alternative name for MILVUS_CHUNKS_COLLECTION (deprecated)
            MILVUS_SKIP_DEDUP: Skip deduplication ('1', 'true', 'yes' to enable, default: false)
            EMBEDDING_BACKEND: 显式指定后端；留空时若 ``BGE_M3_PATH`` 为已存在目录则默认 ``local_bge_m3``，否则 ``openai``

        Returns:
            VectorIndexerConfig: Configuration instance
        """
        rag_mode = os.environ.get("MILVUS_RAG_MODE", "legacy").strip().lower()

        # Validate rag_mode
        if rag_mode not in ("legacy", "v2"):
            raise ValueError(
                f"Invalid MILVUS_RAG_MODE: {rag_mode!r}. Must be 'legacy' or 'v2'"
            )

        # Legacy collection name
        legacy_collection = (
            os.environ.get("MILVUS_COLLECTION") or "edu_knowledge_vectors_v1"
        ).strip()
        if not legacy_collection:
            legacy_collection = "edu_knowledge_vectors_v1"

        # v2 collection names (support legacy env vars for backward compatibility)
        v2_names_collection = (
            os.environ.get("MILVUS_NAMES_COLLECTION")
            or os.environ.get("ITEM_NAME_COLLECTION")
            or "edu_knowledge_item_names_v1"
        ).strip()
        if not v2_names_collection:
            v2_names_collection = "edu_knowledge_item_names_v1"

        v2_chunks_collection = (
            os.environ.get("MILVUS_CHUNKS_COLLECTION")
            or os.environ.get("CHUNKS_COLLECTION")
            or "edu_knowledge_chunks_v1"
        ).strip()
        if not v2_chunks_collection:
            v2_chunks_collection = "edu_knowledge_chunks_v1"

        # Parse skip_dedup flag - accept various truthy values
        skip_dedup_str = os.environ.get("MILVUS_SKIP_DEDUP", "").strip().lower()
        skip_dedup = skip_dedup_str in ("1", "true", "yes", "on")

        # Embedding backend：显式优先；未设置时与 env.example 对齐——有本地模型目录则走本地
        embedding_backend = os.environ.get("EMBEDDING_BACKEND", "").strip()
        if not embedding_backend:
            embedding_backend = (
                "local_bge_m3" if _env_local_bge_path_ok() else "openai"
            )

        return cls(
            rag_mode=rag_mode,  # type: ignore[arg-type]
            legacy_collection=legacy_collection,
            v2_names_collection=v2_names_collection,
            v2_chunks_collection=v2_chunks_collection,
            skip_dedup=skip_dedup,
            embedding_backend=embedding_backend,
        )

    def get_collection_name(self) -> str:
        """Get the appropriate collection name based on rag_mode.

        Returns:
            str: Collection name for the current mode
        """
        if self.rag_mode == "v2":
            raise ValueError(
                "v2 mode uses two collections. Use v2_names_collection and v2_chunks_collection instead."
            )
        return self.legacy_collection


__all__ = ["VectorIndexerConfig"]
