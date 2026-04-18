"""Integration tests for vector_indexer module.

These tests verify the end-to-end functionality of the vector_indexer module,
including both legacy and v2 modes, with proper mocking of external dependencies.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from processor.vector_indexer import vector_indexer_node
from processor.vector_indexer.utils import (
    content_fingerprint,
    item_fingerprint,
    catalog_display_name,
    extract_catalog_items,
    truncate_content_field,
    sanitize_milvus_row,
    MILVUS_VARCHAR_CONTENT_MAX,
)
from processor.import_state import ImportGraphState
from schema.edu_content import EduContent, ContentMetadata, DocumentClass


class TestVectorIndexerIntegration:
    """Integration tests for vector_indexer_node function."""

    def test_legacy_mode_basic_insert(
        self,
        sample_import_state,
        mock_embedding_service,
        mock_milvus_client,
        mock_env_vars,
        monkeypatch,
    ):
        """Test basic insert in legacy mode."""
        # Setup mocks
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_embedding_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(sample_import_state)

        # Verify success
        assert result["is_success"] is True
        assert "indexed_records" in result
        assert "vector_ids" in result
        assert len(result["indexed_records"]) == 5
        assert len(result["vector_ids"]) == 5

        # Verify embedding was called
        mock_embedding_service.embed_documents.assert_called_once()
        call_args = mock_embedding_service.embed_documents.call_args[0][0]
        assert len(call_args) == 5

        # Verify Milvus insert was called
        mock_milvus_client.insert.assert_called_once()

    def test_legacy_mode_deduplication(
        self,
        sample_import_state,
        mock_embedding_service,
        mock_milvus_client,
        mock_env_vars,
        monkeypatch,
    ):
        """Test deduplication in legacy mode."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")

        # Mock existing hashes
        existing_hash = content_fingerprint(
            EduContent.model_validate(sample_import_state["chunks"][0]["edu_content"])
        )
        mock_milvus_client.fetch_existing_in_field.return_value = {existing_hash}

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_embedding_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(sample_import_state)

        # Verify deduplication worked
        assert result["is_success"] is True, f"Test failed with errors: {result.get('errors')}, warnings: {result.get('warnings')}"
        assert len(result["indexed_records"]) == 4  # One was deduplicated
        assert "warnings" in result
        assert any("已跳过 1 条重复切片" in w for w in result["warnings"])

    def test_v2_mode_basic_insert(
        self,
        sample_import_state,
        mock_bge_service,
        mock_milvus_client,
        mock_env_vars,
        monkeypatch,
    ):
        """Test basic insert in v2 mode."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")
        monkeypatch.setenv("MILVUS_NAMES_COLLECTION", "test_names")
        monkeypatch.setenv("MILVUS_CHUNKS_COLLECTION", "test_chunks")

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_bge_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(sample_import_state)

        # Verify success
        assert result["is_success"] is True, f"Test failed with errors: {result.get('errors')}, warnings: {result.get('warnings')}"
        assert "indexed_records" in result
        assert "vector_ids" in result
        assert "indexed_catalog_records" in result
        assert "catalog_vector_ids" in result

        # Verify BGE service was called for chunks with hybrid mode
        # Check that embed_documents was called with mode="hybrid"
        assert mock_bge_service.embed_documents.call_count >= 1
        # Find the hybrid call
        hybrid_call = None
        for call in mock_bge_service.embed_documents.call_args_list:
            args, kwargs = call
            if kwargs.get("mode") == "hybrid" or len(args) > 1:
                hybrid_call = call
                break
        assert hybrid_call is not None, "No hybrid mode call found"

        # Verify Milvus insert was called twice (names + chunks)
        assert mock_milvus_client.insert.call_count == 2

    def test_v2_mode_without_local_bge(
        self,
        sample_import_state,
        mock_bge_service,
        mock_env_vars,
        monkeypatch,
    ):
        """Test v2 mode works with the new service (auto-detects backend)."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")

        # Mock the embedding service to avoid actual API calls
        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_bge_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=MagicMock(
                fetch_existing_in_field=MagicMock(return_value=set()),
                insert=MagicMock(return_value=MagicMock(primary_keys=[1, 2, 3])),
            ),
        ):
            result = vector_indexer_node(sample_import_state)

        # With mock service, it should succeed
        assert result["is_success"] is True

    def test_empty_chunks_handling(
        self,
        mock_env_vars,
    ):
        """Test handling of empty chunks."""
        empty_state = ImportGraphState(
            chunks=[],
            errors=[],
            warnings=[],
            is_success=False,
        )

        result = vector_indexer_node(empty_state)

        # Verify warning about no chunks
        assert result["is_success"] is True
        assert "warnings" in result
        assert any("没有检测到有效切片" in w for w in result["warnings"])

    def test_content_truncation(
        self,
        mock_env_vars,
        mock_embedding_service,
        mock_milvus_client,
    ):
        """Test content truncation for long content."""
        from schema.edu_content import ContentType

        # Create content that exceeds Milvus VarChar limit
        long_content = "A" * (MILVUS_VARCHAR_CONTENT_MAX + 1000)

        meta = ContentMetadata(
            source_file="long.pdf",
            chapter_name="Long Chapter",
            document_class=DocumentClass.LECTURE,
            course_name="Long Course",
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        edu_content = EduContent(content=long_content, metadata=meta)

        state = ImportGraphState(
            chunks=[{"edu_content": edu_content.model_dump()}],
            errors=[],
            warnings=[],
            is_success=False,
        )

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_embedding_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(state)

        # Verify success with truncation warning
        assert result["is_success"] is True
        assert "warnings" in result
        assert any("content 已截断" in w for w in result["warnings"])

        # Verify content was truncated in inserted records
        inserted_records = result["indexed_records"]
        assert len(inserted_records) == 1
        assert len(inserted_records[0]["content"]) == MILVUS_VARCHAR_CONTENT_MAX

    def test_skip_dedup_flag(
        self,
        sample_import_state,
        mock_embedding_service,
        mock_milvus_client,
        mock_env_vars,
        monkeypatch,
    ):
        """Test MILVUS_SKIP_DEDUP flag."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")
        monkeypatch.setenv("MILVUS_SKIP_DEDUP", "1")

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_embedding_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(sample_import_state)

        # Verify no deduplication query was made
        mock_milvus_client.fetch_existing_in_field.assert_not_called()

        # Verify all records were inserted
        assert len(result["indexed_records"]) == 5

    def test_merge_upstream_errors_and_warnings(
        self,
        sample_import_state,
        mock_embedding_service,
        mock_milvus_client,
        mock_env_vars,
        monkeypatch,
    ):
        """Test that upstream errors and warnings are preserved."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")

        # Add upstream errors and warnings
        sample_import_state["errors"] = ["Upstream error 1"]
        sample_import_state["warnings"] = ["Upstream warning 1"]

        with patch(
            "processor.vector_indexer.indexer.get_embedding_service",
            return_value=mock_embedding_service,
        ), patch(
            "processor.vector_indexer.indexer.get_milvus_client",
            return_value=mock_milvus_client,
        ):
            result = vector_indexer_node(sample_import_state)

        # Verify upstream errors/warnings are preserved
        assert "errors" in result
        assert "warnings" in result
        assert "Upstream error 1" in result["errors"]
        assert "Upstream warning 1" in result["warnings"]


class TestVectorIndexerHelpers:
    """Tests for helper functions in vector_indexer module."""

    def test_content_fingerprint(self):
        """Test content fingerprint generation."""
        from schema.edu_content import ContentType

        meta = ContentMetadata(
            source_file="test.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.LECTURE,
            course_name="Test Course",
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        content = EduContent(content="Test content", metadata=meta)

        fp1 = content_fingerprint(content)
        fp2 = content_fingerprint(content)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 hex length

        # Different content should produce different fingerprint
        content2 = EduContent(content="Different content", metadata=meta)
        fp3 = content_fingerprint(content2)
        assert fp1 != fp3

    def test_item_fingerprint(self):
        """Test item fingerprint generation."""
        fp1 = item_fingerprint("syllabus", "Course Name")
        fp2 = item_fingerprint("syllabus", "Course Name")

        assert fp1 == fp2
        assert len(fp1) == 64

        # Different items should produce different fingerprints
        fp3 = item_fingerprint("syllabus", "Different Course")
        assert fp1 != fp3

    def test_catalog_display_name(self):
        """Test catalog display name extraction."""
        from schema.edu_content import ContentType

        # Test with course_name
        meta1 = ContentMetadata(
            source_file="test.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.LECTURE,
            course_name="Math 101",
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        assert catalog_display_name(meta1) == "Math 101"

        # Test without course_name (should use source_file stem)
        meta2 = ContentMetadata(
            source_file="physics_course.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.LECTURE,
            course_name=None,
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        assert catalog_display_name(meta2) == "physics_course"

    def test_extract_catalog_items(self):
        """Test catalog item extraction."""
        from schema.edu_content import ContentType

        documents = []

        # Add syllabus
        meta1 = ContentMetadata(
            source_file="math.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.SYLLABUS,
            course_name="Math 101",
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        documents.append(EduContent(content="Syllabus content", metadata=meta1))

        # Add project
        meta2 = ContentMetadata(
            source_file="project.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.PROJECT,
            course_name="CS 101",
            project_name="Final Project",
            page_number=1,
            content_type=ContentType.DOC_CHUNK,
        )
        documents.append(EduContent(content="Project content", metadata=meta2))

        # Add question bank
        meta3 = ContentMetadata(
            source_file="bank.pdf",
            chapter_name="Chapter 1",
            document_class=DocumentClass.QUESTION_BANK,
            bank_name="Quiz Bank 1",
            page_number=1,
            content_type=ContentType.QUESTION,
        )
        documents.append(EduContent(content="Bank content", metadata=meta3))

        items = extract_catalog_items(documents)

        assert len(items) == 3
        assert items[0]["item_type"] == "syllabus"
        assert items[0]["item_name"] == "Math 101"
        assert items[1]["item_type"] == "project"
        assert items[1]["item_name"] == "Final Project"
        assert items[2]["item_type"] == "bank"
        assert items[2]["item_name"] == "Quiz Bank 1"

    def test_truncate_content_field(self):
        """Test content field truncation."""
        # Content within limit
        row1 = {"content": "Short content", "other": "data"}
        result1, truncated1 = truncate_content_field(row1)
        assert not truncated1
        assert result1 == row1

        # Content exceeding limit
        long_content = "A" * (MILVUS_VARCHAR_CONTENT_MAX + 100)
        row2 = {"content": long_content, "other": "data"}
        result2, truncated2 = truncate_content_field(row2)
        assert truncated2
        assert len(result2["content"]) == MILVUS_VARCHAR_CONTENT_MAX
        assert result2["other"] == "data"

    def test_sanitize_milvus_row(self):
        """Test row sanitization for Milvus."""
        row = {
            "content": "Test content",
            "vector": [0.1, 0.2, 0.3],
            "count": 42,
            "ratio": 0.5,
            "flag": True,
            "nested": {"key": "value"},
            "none_value": None,
        }

        result = sanitize_milvus_row(row)

        assert result["content"] == "Test content"
        assert result["vector"] == [0.1, 0.2, 0.3]
        assert result["count"] == 42
        assert result["ratio"] == 0.5
        assert result["flag"] == 1  # Boolean converted to int
        assert result["nested"] == '{"key": "value"}'  # JSON serialized
        assert result["none_value"] == ""  # None converted to empty string
