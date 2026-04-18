"""LangGraph 导入与查询处理器。

子模块按需加载，避免 ``import processor.query_process`` 时拉取 docx / 抽取链等重依赖。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "content_classifier_node",
    "file_router_node",
    "lecture_extractor_node",
    "question_extractor_node",
    "syllabus_extractor_node",
    "vector_indexer_node",
    "ImportGraphState",
]


def __getattr__(name: str) -> Any:
    if name == "content_classifier_node":
        from processor.extractors.content_classifier import content_classifier_node

        return content_classifier_node
    if name == "file_router_node":
        from processor.file_router import file_router_node

        return file_router_node
    if name == "lecture_extractor_node":
        from processor.extractors.lecture_extractor import lecture_extractor_node

        return lecture_extractor_node
    if name == "question_extractor_node":
        from processor.extractors.question_extractor import question_extractor_node

        return question_extractor_node
    if name == "syllabus_extractor_node":
        from processor.extractors.syllabus_extractor import syllabus_extractor_node

        return syllabus_extractor_node
    if name == "vector_indexer_node":
        from processor.vector_indexer import vector_indexer_node

        return vector_indexer_node
    if name == "ImportGraphState":
        from processor.import_state import ImportGraphState

        return ImportGraphState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
