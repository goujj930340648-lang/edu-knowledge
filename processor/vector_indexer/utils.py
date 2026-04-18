"""
Utility functions for vector indexer module.

This module contains pure utility functions extracted from vector_indexer.py.
All functions are pure (no dependencies on global state) and have comprehensive
documentation for easier testing and maintenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from schema.edu_content import ContentMetadata, EduContent
from schema.metadata import DocumentClass

from utils.client import MilvusIndexerClient

# Milvus 2.x single-field VarChar limit
MILVUS_VARCHAR_CONTENT_MAX = 65535

# Vector field keys that should be preserved as-is
_VECTOR_KEYS = frozenset({"vector", "vector_dense", "vector_sparse"})


def truncate_content_field(
    row: dict[str, Any], max_length: int = MILVUS_VARCHAR_CONTENT_MAX
) -> tuple[dict[str, Any], bool]:
    """Truncate content field to fit within Milvus VarChar limits.

    This function handles content truncation for Milvus collections, which have
    a maximum VarChar length of 65535 characters. Long content is common when
    entire syllabus documents are stored as single chunks.

    Args:
        row: Dictionary containing the data row to be inserted into Milvus.
        max_length: Maximum allowed length for content field. Defaults to
            MILVUS_VARCHAR_CONTENT_MAX (65535).

    Returns:
        A tuple of (modified_row, was_truncated):
            - modified_row: New dictionary with content truncated if necessary.
            - was_truncated: Boolean indicating if truncation occurred.

    Examples:
        >>> row = {"content": "A" * 70000, "other": "data"}
        >>> result, truncated = truncate_content_field(row)
        >>> truncated
        True
        >>> len(result["content"])
        65535

        >>> short_row = {"content": "Short text", "other": "data"}
        >>> result, truncated = truncate_content_field(short_row)
        >>> truncated
        False
    """
    raw = row.get("content")
    if not isinstance(raw, str) or len(raw) <= max_length:
        return row, False
    out = dict(row)
    out["content"] = raw[:max_length]
    return out, True


def content_fingerprint(doc: EduContent) -> str:
    """Generate a stable fingerprint for content deduplication.

    This function creates a SHA256 hash based on the content's source file,
    chapter name, and actual text content. This fingerprint is used for
    deduplication to avoid inserting duplicate content into Milvus.

    The fingerprint is stable across multiple runs for the same content,
    enabling efficient deduplication.

    Args:
        doc: EduContent object containing the content and metadata.

    Returns:
        A 64-character hexadecimal string representing the SHA256 hash.

    Examples:
        >>> from schema.edu_content import EduContent, ContentMetadata, ContentType, DocumentClass
        >>> meta = ContentMetadata(
        ...     source_file="test.pdf",
        ...     chapter_name="Chapter 1",
        ...     document_class=DocumentClass.LECTURE,
        ...     content_type=ContentType.DOC_CHUNK
        ... )
        >>> content = EduContent(content="Test content", metadata=meta)
        >>> fp = content_fingerprint(content)
        >>> len(fp)
        64

        # Same content produces same fingerprint
        >>> fp2 = content_fingerprint(content)
        >>> fp == fp2
        True
    """
    meta = doc.metadata
    raw = f"{meta.source_file}\0{meta.chapter_name or ''}\0{doc.content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def item_fingerprint(item_type: str, item_name: str) -> str:
    """Generate a stable fingerprint for catalog item deduplication.

    This function creates a SHA256 hash based on item type and name for
    deduplication in the names catalog collection. This prevents duplicate
    entries for the same course, project, or question bank.

    Args:
        item_type: Type of catalog item (e.g., "syllabus", "project", "bank").
        item_name: Name of the catalog item.

    Returns:
        A 64-character hexadecimal string representing the SHA256 hash.

    Examples:
        >>> fp1 = item_fingerprint("syllabus", "Math 101")
        >>> fp2 = item_fingerprint("syllabus", "Math 101")
        >>> fp1 == fp2
        True

        # Different items produce different fingerprints
        >>> fp3 = item_fingerprint("syllabus", "Physics 101")
        >>> fp1 != fp3
        True
    """
    return hashlib.sha256(f"{item_type}\0{item_name}".encode("utf-8")).hexdigest()


def catalog_display_name(meta: ContentMetadata) -> str | None:
    """Extract display name for catalog entries.

    This function determines the appropriate display name for catalog items:
    - Prioritizes course_name if available
    - Falls back to source_file stem (filename without extension)
    - Returns None if neither is available

    The fallback to source_file stem aligns with lecture_extractor's virtual
    H1 semantics, ensuring consistency when course_name is not populated by
    the classifier.

    Args:
        meta: ContentMetadata object containing document metadata.

    Returns:
        Display name string, or None if no suitable name is found.

    Examples:
        >>> from schema.edu_content import ContentMetadata, ContentType, DocumentClass
        >>> meta = ContentMetadata(
        ...     source_file="course.pdf",
        ...     chapter_name="Chapter 1",
        ...     course_name="Math 101",
        ...     document_class=DocumentClass.LECTURE,
        ...     content_type=ContentType.DOC_CHUNK
        ... )
        >>> catalog_display_name(meta)
        'Math 101'

        >>> meta2 = ContentMetadata(
        ...     source_file="physics_course.pdf",
        ...     document_class=DocumentClass.LECTURE,
        ...     content_type=ContentType.DOC_CHUNK
        ... )
        >>> catalog_display_name(meta2)
        'physics_course'
    """
    cn = (meta.course_name or "").strip()
    if cn:
        return cn
    sf = (meta.source_file or "").strip()
    if not sf:
        return None
    stem = Path(sf).stem.strip()
    return stem or None


def extract_catalog_items(documents: list[EduContent]) -> list[dict[str, Any]]:
    """Extract catalog items from a list of EduContent documents.

    This function extracts unique catalog items (courses, projects, question banks)
    from documents for insertion into the names catalog collection.

    Extraction rules:
    - SYLLABUS and LECTURE → item_type="syllabus" (for course search alignment)
    - PROJECT → item_type="project" (uses project_name or course_name)
    - QUESTION_BANK → item_type="bank" (requires bank_name)

    When course_name is missing, the source_file stem is used to ensure the
    v2 names catalog is not constantly empty.

    Args:
        documents: List of EduContent objects to extract catalog items from.

    Returns:
        List of dictionaries, each containing:
            - item_type: Type of catalog item ("syllabus", "project", "bank")
            - item_name: Display name of the item
            - item_fingerprint: Unique hash for deduplication

    Examples:
        >>> from schema.edu_content import EduContent, ContentMetadata, ContentType, DocumentClass
        >>> docs = [
        ...     EduContent(
        ...         content="Syllabus",
        ...         metadata=ContentMetadata(
        ...             source_file="math.pdf",
        ...             document_class=DocumentClass.SYLLABUS,
        ...             course_name="Math 101",
        ...             content_type=ContentType.DOC_CHUNK
        ...         )
        ...     )
        ... ]
        >>> items = extract_catalog_items(docs)
        >>> len(items)
        1
        >>> items[0]["item_type"]
        'syllabus'
        >>> items[0]["item_name"]
        'Math 101'
    """
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for doc in documents:
        meta = doc.metadata
        dc = meta.document_class.value if meta.document_class else None
        item_type: str | None = None
        item_name: str | None = None
        if dc == DocumentClass.SYLLABUS.value:
            item_name = catalog_display_name(meta)
            if item_name:
                item_type, item_name = "syllabus", item_name
        elif dc == DocumentClass.LECTURE.value:
            item_name = catalog_display_name(meta)
            if item_name:
                item_type, item_name = "syllabus", item_name
        elif dc == DocumentClass.PROJECT.value:
            cand = (meta.project_name or meta.course_name or "").strip()
            if not cand:
                cand = (catalog_display_name(meta) or "").strip()
            if cand:
                item_type, item_name = "project", cand
        elif dc == DocumentClass.QUESTION_BANK.value and meta.bank_name:
            item_type, item_name = "bank", meta.bank_name.strip()
        if not item_type or not item_name:
            continue
        key = (item_type, item_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "item_type": item_type,
                "item_name": item_name,
                "item_fingerprint": item_fingerprint(item_type, item_name),
            }
        )
    return rows


def sanitize_milvus_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert nested structures to Milvus-compatible scalar types.

    This function prepares a data row for insertion into Milvus by:
    - Preserving vector fields as-is
    - Converting None to empty string
    - Converting bool to int (Milvus doesn't support bool)
    - JSON-serializing complex nested structures

    Args:
        row: Dictionary containing the data to be sanitized.

    Returns:
        A new dictionary with all values converted to Milvus-compatible types.

    Examples:
        >>> row = {
        ...     "content": "Test",
        ...     "vector": [0.1, 0.2],
        ...     "count": 42,
        ...     "flag": True,
        ...     "nested": {"key": "value"},
        ...     "none_value": None
        ... }
        >>> result = sanitize_milvus_row(row)
        >>> result["content"]
        'Test'
        >>> result["vector"]
        [0.1, 0.2]
        >>> result["flag"]
        1
        >>> result["nested"]
        '{"key": "value"}'
        >>> result["none_value"]
        ''
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in _VECTOR_KEYS:
            out[k] = v
        elif v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float)):
            out[k] = v
        elif isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def batch_query_field(
    milvus: MilvusIndexerClient,
    collection_name: str,
    field: str,
    values: list[str],
    batch_size: int = 64,
) -> set[str]:
    """Query existing values in a Milvus collection field with batching.

    This function checks which values already exist in a specific field of a
    Milvus collection, using batching to handle large value lists efficiently.

    Args:
        milvus: MilvusIndexerClient instance for querying.
        collection_name: Name of the Milvus collection to query.
        field: Field name to check for existing values.
        values: List of values to check for existence.
        batch_size: Number of values to query per batch. Defaults to 64.

    Returns:
        Set of values that already exist in the specified field.

    Examples:
        >>> # Mock example
        >>> existing = batch_query_field(
        ...     milvus_client,
        ...     "my_collection",
        ...     "content_hash",
        ...     ["hash1", "hash2", "hash3"]
        ... )
        >>> "hash1" in existing
        True
    """
    existing: set[str] = set()
    for i in range(0, len(values), batch_size):
        part = values[i : i + batch_size]
        existing |= milvus.fetch_existing_in_field(collection_name, field, part)
    return existing


def merge_upstream_lists(
    state: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    """Merge upstream errors and warnings with current node results.

    LangGraph replaces TypedDict state keys entirely rather than merging.
    This function ensures that errors and warnings from upstream nodes
    (file_router, classifier, extractors) are preserved when the current
    node returns its own errors/warnings.

    Args:
        state: Current LangGraph state containing upstream errors/warnings.
        patch: New state updates from the current node.

    Returns:
        Merged state dictionary with combined errors and warnings.

    Examples:
        >>> state = {"errors": ["Error 1"], "warnings": ["Warning 1"]}
        >>> patch = {"errors": ["Error 2"], "data": "value"}
        >>> result = merge_upstream_lists(state, patch)
        >>> result["errors"]
        ['Error 1', 'Error 2']
        >>> result["warnings"]
        ['Warning 1']
        >>> result["data"]
        'value'
    """
    out = dict(patch)
    prior_e = list(state.get("errors") or [])
    prior_w = list(state.get("warnings") or [])
    new_e = list(out.get("errors") or [])
    if prior_e or new_e:
        out["errors"] = prior_e + new_e
    new_w = list(out.get("warnings") or [])
    if prior_w or new_w:
        out["warnings"] = prior_w + new_w
    return out
