"""Pytest configuration and shared fixtures for vector_indexer tests."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "MILVUS_COLLECTION": "test_vectors",
        "MILVUS_RAG_MODE": "legacy",
        "MILVUS_SKIP_DEDUP": "0",
        "EMBEDDING_BACKEND": "mock",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns EmbeddingResult."""
    from processor.vector_indexer.embedding_service import EmbeddingResult, EmbeddingBackend

    mock_service = MagicMock()

    def embed_documents_func(texts):
        # Return EmbeddingResult with the same number of vectors as input texts
        return EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] for _ in range(len(texts))],
            backend=EmbeddingBackend.OPENAI,
            processing_time_ms=100.0,
        )

    def embed_dense_only_func(texts):
        # Return EmbeddingResult with the same number of vectors as input texts
        return EmbeddingResult(
            embeddings=[[0.4, 0.5, 0.6] for _ in range(len(texts))],
            backend=EmbeddingBackend.OPENAI,
            processing_time_ms=100.0,
        )

    def embed_dense_sparse_func(texts):
        # Return tuple of EmbeddingResult for dense and sparse
        dense = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] for _ in range(len(texts))],
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=150.0,
        )
        sparse = EmbeddingResult(
            embeddings=[{0: 0.5, 1: 0.3} for _ in range(len(texts))],  # type: ignore
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=150.0,
        )
        return (dense, sparse)

    mock_service.embed_documents.side_effect = embed_documents_func
    mock_service.embed_dense_only.side_effect = embed_dense_only_func
    mock_service.embed_dense_sparse.side_effect = embed_dense_sparse_func
    return mock_service


@pytest.fixture
def mock_bge_service():
    """Mock BGE-M3 embedding service that returns EmbeddingResult."""
    from processor.vector_indexer.embedding_service import EmbeddingResult, EmbeddingBackend

    mock_service = MagicMock()

    def embed_documents_func(texts):
        # Return EmbeddingResult with the same number of vectors as input texts
        return EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] for _ in range(len(texts))],
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=100.0,
        )

    def embed_dense_only_func(texts):
        # Return EmbeddingResult with the same number of vectors as input texts
        return EmbeddingResult(
            embeddings=[[0.4, 0.5, 0.6] for _ in range(len(texts))],
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=100.0,
        )

    def embed_dense_sparse_func(texts):
        # Return tuple of EmbeddingResult for dense and sparse
        dense = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] for _ in range(len(texts))],
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=150.0,
        )
        sparse = EmbeddingResult(
            embeddings=[{0: 0.5, 1: 0.3} for _ in range(len(texts))],  # type: ignore
            backend=EmbeddingBackend.BGE_M3,
            processing_time_ms=150.0,
        )
        return (dense, sparse)

    mock_service.embed_documents.side_effect = embed_documents_func
    mock_service.embed_dense_only.side_effect = embed_dense_only_func
    mock_service.embed_dense_sparse.side_effect = embed_dense_sparse_func
    return mock_service


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client (deprecated - use mock_embedding_service)."""
    mock_client = MagicMock()

    def embed_documents_func(texts):
        # Return the same number of vectors as input texts
        return [[0.1, 0.2, 0.3] for _ in range(len(texts))]

    mock_client.embed_documents.side_effect = embed_documents_func
    return mock_client


@pytest.fixture
def mock_local_bge_client():
    """Mock local BGE client (deprecated - use mock_bge_service)."""
    mock_client = MagicMock()

    def embed_documents_dense_only_func(texts):
        # Return the same number of vectors as input texts
        return [[0.1, 0.2, 0.3] for _ in range(len(texts))]

    def embed_documents_dense_sparse_func(texts):
        # Return the same number of vectors as input texts
        return (
            [[0.1, 0.2, 0.3] for _ in range(len(texts))],  # dense
            [{0: 0.5, 1: 0.3} for _ in range(len(texts))],  # sparse
        )

    def embed_items_dense_func(texts):
        # Return the same number of vectors as input texts
        return [[0.4, 0.5, 0.6] for _ in range(len(texts))]

    mock_client.embed_documents_dense_only.side_effect = embed_documents_dense_only_func
    mock_client.embed_documents_dense_sparse.side_effect = embed_documents_dense_sparse_func
    mock_client.embed_items_dense.side_effect = embed_items_dense_func
    mock_client.should_use_local_bge_embedding.return_value = False
    return mock_client


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client."""
    mock_client = MagicMock()
    mock_client.fetch_existing_in_field.return_value = set()
    mock_client.insert.return_value = MagicMock(primary_keys=[1, 2, 3, 4, 5])
    return mock_client


@pytest.fixture
def sample_edu_content():
    """Create sample EduContent objects for testing."""
    from schema.edu_content import EduContent, ContentMetadata, DocumentClass, ContentType

    contents = []
    for i in range(5):
        meta = ContentMetadata(
            source_file=f"test_file_{i}.pdf",
            chapter_name=f"Chapter {i}",
            document_class=DocumentClass.LECTURE,
            course_name=f"Test Course {i % 2}",
            page_number=i + 1,
            content_type=ContentType.DOC_CHUNK,
        )
        content = EduContent(
            content=f"This is test content number {i}. " * 10,
            metadata=meta,
        )
        contents.append(content)

    return contents


@pytest.fixture
def sample_import_state(sample_edu_content):
    """Create sample ImportGraphState for testing."""
    from processor.import_state import ImportGraphState

    chunks = []
    for content in sample_edu_content:
        chunks.append({"edu_content": content.model_dump()})

    return ImportGraphState(
        chunks=chunks,
        errors=[],
        warnings=[],
        is_success=False,
    )
