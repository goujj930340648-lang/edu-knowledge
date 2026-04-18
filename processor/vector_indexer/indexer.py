"""
Vector indexer strategy implementation.

This module implements the Strategy pattern for vector indexing, with:
- IndexerResult: Dataclass for indexing results
- BaseIndexer: Abstract base class with common functionality
- LegacyIndexer: Single collection + dense vectors (OpenAI/legacy mode)
- V2Indexer: Dual collections + hybrid vectors (BGE-M3/v2 mode)
- create_indexer: Factory function for indexer instantiation

Key design decisions:
1. V2Indexer calls embed_documents() only ONCE with mode="hybrid"
2. Deduplication is unified in BaseIndexer._deduplicate_content_hashes()
3. Sequential execution (no parallelization) for simplicity
4. Each indexer class handles its own logic without conditionals
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from schema.edu_content import EduContent

from config.vector_config import VectorIndexerConfig
from processor.vector_indexer.embedding_service import (
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
    get_embedding_service,
)
from processor.vector_indexer.utils import (
    MILVUS_VARCHAR_CONTENT_MAX,
    batch_query_field,
    catalog_display_name,
    content_fingerprint,
    extract_catalog_items,
    item_fingerprint,
    sanitize_milvus_row,
    truncate_content_field,
)
from utils.client import get_milvus_client


@dataclass(frozen=True)
class IndexerResult:
    """Result from vector indexing operation.

    Attributes:
        indexed_records: List of chunk records inserted into Milvus
        vector_ids: List of vector IDs for the inserted chunks
        indexed_catalog_records: List of catalog/name records (v2 mode only)
        catalog_vector_ids: List of vector IDs for catalog entries (v2 mode only)
        warnings: List of warning messages (deduplication, truncation, etc.)
        errors: List of error messages (indexed as fatal)
        is_success: Whether the indexing operation succeeded

    Note:
        Legacy mode only uses indexed_records and vector_ids.
        V2 mode uses all fields including catalog records.
    """

    indexed_records: list[dict[str, Any]] = field(default_factory=list)
    vector_ids: list[str] = field(default_factory=list)
    indexed_catalog_records: list[dict[str, Any]] = field(default_factory=list)
    catalog_vector_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_success: bool = True


class BaseIndexer(ABC):
    """Abstract base class for vector indexers.

    Provides common functionality for deduplication, content fingerprinting,
    and defines the interface for concrete indexer implementations.

    Attributes:
        config: VectorIndexerConfig instance
        service: EmbeddingService instance
        milvus: MilvusIndexerClient instance
    """

    def __init__(
        self,
        config: VectorIndexerConfig,
        service: EmbeddingService,
    ) -> None:
        """Initialize base indexer.

        Args:
            config: VectorIndexerConfig instance
            service: EmbeddingService instance
        """
        self.config = config
        self.service = service
        self.milvus = get_milvus_client()

    def _deduplicate_content_hashes(
        self,
        hashes: list[str],
        collection_name: str,
    ) -> set[str] | dict[str, Any]:
        """Deduplicate content hashes against existing Milvus records.

        This unified method handles deduplication for both legacy and v2 modes,
        respecting the skip_dedup configuration flag.

        Args:
            hashes: List of content hash strings to check
            collection_name: Milvus collection name to query

        Returns:
            Set of existing hashes if successful, or error dict if failed
        """
        if self.config.skip_dedup:
            return set()

        try:
            existing = batch_query_field(
                self.milvus,
                collection_name,
                "content_hash",
                hashes,
            )
            return existing
        except Exception as e:
            return {"errors": [f"去重查询失败: {e}"], "is_success": False}

    def _filter_unique_documents(
        self,
        documents: list[EduContent],
        flat_rows: list[dict[str, Any]],
        skipped_hashes: set[str],
    ) -> tuple[list[EduContent], list[dict[str, Any]]]:
        """Filter out documents with existing content hashes.

        Args:
            documents: List of EduContent objects
            flat_rows: List of flattened dictionaries with content_hash
            skipped_hashes: Set of content hashes to skip

        Returns:
            Tuple of (filtered_documents, filtered_rows)
        """
        to_embed_docs: list[EduContent] = []
        to_embed_rows: list[dict[str, Any]] = []

        for doc, row in zip(documents, flat_rows):
            if row["content_hash"] in skipped_hashes:
                continue
            to_embed_docs.append(doc)
            to_embed_rows.append(row)

        return to_embed_docs, to_embed_rows

    @abstractmethod
    def index(
        self,
        documents: list[EduContent],
        flat_rows: list[dict[str, Any]],
    ) -> IndexerResult:
        """Index documents into Milvus.

        Args:
            documents: List of EduContent objects to index
            flat_rows: List of flattened dictionaries with metadata and hashes

        Returns:
            IndexerResult with indexing status and records
        """
        ...


class LegacyIndexer(BaseIndexer):
    """Legacy indexer for single collection with dense vectors.

    Uses:
    - Single Milvus collection (config.legacy_collection)
    - Dense-only embeddings (OpenAI-compatible API)
    - content_hash for deduplication
    """

    def index(
        self,
        documents: list[EduContent],
        flat_rows: list[dict[str, Any]],
    ) -> IndexerResult:
        """Index documents in legacy mode.

        Args:
            documents: List of EduContent objects
            flat_rows: List of flattened dictionaries with content_hash

        Returns:
            IndexerResult with indexed records
        """
        # Step 1: Deduplicate by content_hash
        hashes = [r["content_hash"] for r in flat_rows]
        dedup_result = self._deduplicate_content_hashes(
            hashes,
            self.config.legacy_collection,
        )

        # Check if deduplication failed
        if isinstance(dedup_result, dict) and "errors" in dedup_result:
            return IndexerResult(
                errors=dedup_result["errors"],
                is_success=False,
            )

        skipped_hashes: set[str] = dedup_result  # type: ignore[assignment]

        # Step 2: Filter unique documents
        to_embed_docs, to_embed_rows = self._filter_unique_documents(
            documents,
            flat_rows,
            skipped_hashes,
        )

        if not to_embed_rows:
            return IndexerResult(
                warnings=[
                    f"全部切片均已存在（content_hash 命中 {len(skipped_hashes)} 条），跳过写入"
                ],
                is_success=True,
            )

        # Step 3: Generate embeddings
        texts = [d.content for d in to_embed_docs]
        try:
            result = self.service.embed_documents(texts)

            if isinstance(result, EmbeddingError):
                return IndexerResult(
                    errors=[f"Embedding 失败: {result.message}"],
                    is_success=False,
                )

            vectors = result.embeddings
        except Exception as e:
            return IndexerResult(
                errors=[f"Embedding 失败: {e}"],
                is_success=False,
            )

        if len(vectors) != len(to_embed_rows):
            return IndexerResult(
                errors=[
                    f"Embedding 返回条数与文档不一致: {len(vectors)} != {len(to_embed_rows)}"
                ],
                is_success=False,
            )

        # Step 4: Prepare records with truncation
        trunc_notes: list[str] = []
        records = []
        for row, vec in zip(to_embed_rows, vectors):
            payload = dict(row)
            payload["vector"] = vec
            payload, did_trunc = truncate_content_field(payload)
            if did_trunc:
                trunc_notes.append(
                    f"content 已截断至 {MILVUS_VARCHAR_CONTENT_MAX} 字符（Milvus VarChar 上限）"
                )
            records.append(sanitize_milvus_row(payload))

        # Step 5: Insert into Milvus
        try:
            res = self.milvus.insert(
                collection_name=self.config.legacy_collection,
                data=records,
            )
        except Exception as e:
            return IndexerResult(
                errors=[f"Milvus 入库失败: {e}"],
                is_success=False,
            )

        pks = getattr(res, "primary_keys", None)
        vector_ids = [str(x) for x in list(pks)] if pks is not None else []

        # Step 6: Build warnings
        warnings: list[str] = []
        if skipped_hashes:
            warnings.append(f"已跳过 {len(skipped_hashes)} 条重复切片（content_hash 已存在）")
        if trunc_notes:
            warnings.append(
                f"{len(trunc_notes)} 条切片：{trunc_notes[0]}"
                + ("（其余同上）" if len(trunc_notes) > 1 else "")
            )

        return IndexerResult(
            indexed_records=records,
            vector_ids=vector_ids,
            warnings=warnings,
            is_success=True,
        )


class V2Indexer(BaseIndexer):
    """V2 indexer for dual collections with hybrid vectors.

    Uses:
    - Dual collections (names + chunks)
    - Hybrid embeddings (dense + sparse) via BGE-M3
    - item_fingerprint for names deduplication
    - content_hash for chunks deduplication

    Key optimization: Calls embed_documents() only ONCE with mode="hybrid"
    to get both dense and sparse vectors simultaneously.
    """

    def index(
        self,
        documents: list[EduContent],
        flat_rows: list[dict[str, Any]],
    ) -> IndexerResult:
        """Index documents in v2 mode.

        Args:
            documents: List of EduContent objects
            flat_rows: List of flattened dictionaries with content_hash

        Returns:
            IndexerResult with indexed chunks and catalog records
        """
        warnings: list[str] = []

        # Step 1: Index names catalog (if any items)
        catalog_inserts: list[dict[str, Any]] = []
        catalog_ids: list[str] = []

        catalog_rows = extract_catalog_items(documents)
        if catalog_rows:
            catalog_result = self._index_names_catalog(catalog_rows)
            if not catalog_result.is_success:
                return catalog_result  # Fatal error

            catalog_inserts = catalog_result.indexed_catalog_records
            catalog_ids = catalog_result.catalog_vector_ids
            warnings.extend(catalog_result.warnings)

        # Step 2: Deduplicate chunks by content_hash
        hashes = [r["content_hash"] for r in flat_rows]
        dedup_result = self._deduplicate_content_hashes(
            hashes,
            self.config.v2_chunks_collection,
        )

        # Check if deduplication failed
        if isinstance(dedup_result, dict) and "errors" in dedup_result:
            return IndexerResult(
                errors=dedup_result["errors"],
                is_success=False,
            )

        skipped_hashes: set[str] = dedup_result  # type: ignore[assignment]

        # Step 3: Filter unique documents
        to_embed_docs, to_embed_rows = self._filter_unique_documents(
            documents,
            flat_rows,
            skipped_hashes,
        )

        if not to_embed_rows:
            wmsg = f"全部切片均已存在（content_hash 命中 {len(skipped_hashes)} 条），跳过切片写入"
            warnings.append(wmsg)
            return IndexerResult(
                indexed_catalog_records=catalog_inserts,
                catalog_vector_ids=catalog_ids,
                warnings=warnings,
                is_success=True,
            )

        # Step 4: Generate hybrid embeddings (dense + sparse) in ONE call
        texts = [d.content for d in to_embed_docs]
        try:
            # KEY DESIGN: Single call with mode="hybrid" to get both vectors
            result = self.service.embed_documents(texts, mode="hybrid")  # type: ignore[call-arg]

            if isinstance(result, EmbeddingError):
                return IndexerResult(
                    errors=[f"混合向量生成失败: {result.message}"],
                    is_success=False,
                )

            # Extract dense and sparse from EmbeddingResult
            dense_vecs = result.dense_vectors  # type: ignore[attr-defined]
            sparse_vecs = result.sparse_vectors  # type: ignore[attr-defined]

        except Exception as e:
            return IndexerResult(
                errors=[f"混合向量生成失败: {e}"],
                is_success=False,
            )

        if len(dense_vecs) != len(to_embed_rows) or len(sparse_vecs) != len(to_embed_rows):
            return IndexerResult(
                errors=["稠密/稀疏向量条数与切片不一致"],
                is_success=False,
            )

        # Step 5: Prepare chunk records with truncation
        chunk_records: list[dict[str, Any]] = []
        content_trunc_count = 0
        fixed_scalar = {
            "content",
            "vector_dense",
            "vector_sparse",
            "course_name",
            "content_type",
            "content_hash",
            "source_file",
            "document_class",
            "question_type",
        }

        for doc, row, dv, sv in zip(to_embed_docs, to_embed_rows, dense_vecs, sparse_vecs):
            base: dict[str, Any] = {
                "content": row.get("content") or doc.content,
                "vector_dense": dv,
                "vector_sparse": sv,
                "course_name": row.get("course_name") or "",
                "content_type": row.get("content_type") or "",
                "content_hash": row["content_hash"],
                "source_file": row.get("source_file") or "",
                "document_class": row.get("document_class") or "",
                "question_type": row.get("question_type") or "",
            }
            for k, v in row.items():
                if k in fixed_scalar:
                    continue
                base[k] = v
            base, did_trunc = truncate_content_field(base)
            if did_trunc:
                content_trunc_count += 1
            chunk_records.append(sanitize_milvus_row(base))

        # Step 6: Insert chunks into Milvus
        try:
            res_c = self.milvus.insert(
                collection_name=self.config.v2_chunks_collection,
                data=chunk_records,
            )
            pkc = getattr(res_c, "primary_keys", None)
            chunk_ids = [str(x) for x in list(pkc)] if pkc is not None else []
        except Exception as e:
            return IndexerResult(
                errors=[f"切片表 Milvus 写入失败: {e}"],
                is_success=False,
            )

        # Step 7: Build warnings
        if skipped_hashes:
            warnings.append(f"已跳过 {len(skipped_hashes)} 条重复切片（content_hash 已存在）")
        if content_trunc_count:
            warnings.append(
                f"{content_trunc_count} 条切片的 content 超过 Milvus VarChar 上限（{MILVUS_VARCHAR_CONTENT_MAX}），"
                "已截断后写入；长文 syllabus 建议后续改为按章节多切片。"
            )

        return IndexerResult(
            indexed_records=chunk_records,
            vector_ids=chunk_ids,
            indexed_catalog_records=catalog_inserts,
            catalog_vector_ids=catalog_ids,
            warnings=warnings,
            is_success=True,
        )

    def _index_names_catalog(
        self,
        catalog_rows: list[dict[str, Any]],
    ) -> IndexerResult:
        """Index names catalog items.

        Args:
            catalog_rows: List of catalog item dictionaries

        Returns:
            IndexerResult with catalog records
        """
        warnings: list[str] = []

        # Step 1: Deduplicate by item_fingerprint
        fp_list = [r["item_fingerprint"] for r in catalog_rows]
        dedup_result = self._deduplicate_content_hashes(
            fp_list,
            self.config.v2_names_collection,
        )

        # Check if deduplication failed
        if isinstance(dedup_result, dict) and "errors" in dedup_result:
            return IndexerResult(
                errors=dedup_result["errors"],
                is_success=False,
            )

        existing_fp: set[str] = dedup_result  # type: ignore[assignment]

        # Step 2: Filter new items
        new_items = [r for r in catalog_rows if r["item_fingerprint"] not in existing_fp]

        if existing_fp:
            warnings.append(f"已跳过 {len(existing_fp)} 条重复名称项（item_fingerprint）")

        if not new_items:
            return IndexerResult(
                warnings=warnings,
                is_success=True,
            )

        # Step 3: Generate embeddings for names
        texts = [r["item_name"] for r in new_items]
        try:
            result = self.service.embed_dense_only(texts)

            if isinstance(result, EmbeddingError):
                return IndexerResult(
                    errors=[f"名称表 Embedding 失败: {result.message}"],
                    is_success=False,
                )

            name_vecs = result.embeddings
        except Exception as e:
            return IndexerResult(
                errors=[f"名称表 Embedding 失败: {e}"],
                is_success=False,
            )

        if len(name_vecs) != len(new_items):
            return IndexerResult(
                errors=["名称表向量条数与记录不一致"],
                is_success=False,
            )

        # Step 4: Prepare catalog records
        catalog_inserts = []
        for r, vec in zip(new_items, name_vecs):
            catalog_inserts.append(
                {
                    "item_name": r["item_name"],
                    "item_type": r["item_type"],
                    "item_fingerprint": r["item_fingerprint"],
                    "vector": vec,
                }
            )

        # Step 5: Insert into Milvus
        try:
            res_n = self.milvus.insert(
                collection_name=self.config.v2_names_collection,
                data=catalog_inserts,
            )
            pkn = getattr(res_n, "primary_keys", None)
            catalog_ids = [str(x) for x in list(pkn)] if pkn is not None else []
        except Exception as e:
            return IndexerResult(
                errors=[f"名称表 Milvus 写入失败: {e}"],
                is_success=False,
            )

        return IndexerResult(
            indexed_catalog_records=catalog_inserts,
            catalog_vector_ids=catalog_ids,
            warnings=warnings,
            is_success=True,
        )


def create_indexer(config: VectorIndexerConfig) -> BaseIndexer:
    """Factory function to create appropriate indexer based on config.

    Args:
        config: VectorIndexerConfig instance

    Returns:
        LegacyIndexer for legacy mode, V2Indexer for v2 mode

    Example:
        >>> config = VectorIndexerConfig.from_env()
        >>> indexer = create_indexer(config)
        >>> result = indexer.index(documents, flat_rows)
    """
    service = get_embedding_service()

    if config.rag_mode == "v2":
        return V2Indexer(config, service)
    else:
        return LegacyIndexer(config, service)


__all__ = [
    "IndexerResult",
    "BaseIndexer",
    "LegacyIndexer",
    "V2Indexer",
    "create_indexer",
]
