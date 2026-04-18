"""Additional fixtures for vector_indexer tests."""

from unittest.mock import MagicMock
from schema.edu_content import EduContent, ContentMetadata, DocumentClass


def create_test_document(
    content: str = "Test content",
    source_file: str = "test.pdf",
    chapter_name: str = "Chapter 1",
    document_class: DocumentClass = DocumentClass.LECTURE,
    course_name: str = "Test Course",
) -> EduContent:
    """Helper function to create test EduContent objects."""
    from schema.edu_content import ContentType

    meta = ContentMetadata(
        source_file=source_file,
        chapter_name=chapter_name,
        document_class=document_class,
        course_name=course_name,
        page_number=1,
        content_type=ContentType.DOC_CHUNK,
    )
    return EduContent(content=content, metadata=meta)


def create_milvus_insert_result(primary_keys: list) -> MagicMock:
    """Helper to create mock Milvus insert result."""
    mock_result = MagicMock()
    mock_result.primary_keys = primary_keys
    return mock_result


def create_batch_query_result(existing_values: set) -> MagicMock:
    """Helper to create mock batch query result."""
    return existing_values
