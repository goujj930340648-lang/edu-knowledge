from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CatalogIngestRequest(BaseModel):
    file_path: str = Field(..., description="《课程介绍.md》绝对或相对路径")


class QuestionsIngestRequest(BaseModel):
    file_path: str = Field(..., description="《题目资料.md》路径")


class DocumentsIngestRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
    doc_type: Literal["course_doc", "project_doc"] = "course_doc"
    source_mappings: list[dict[str, Any]] | None = None


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    stream: bool = False


class SourceMappingBatchRequest(BaseModel):
    """人工或脚本批量写入 ``source_mapping``（默认 ``mapping_type=manual``）。"""

    mappings: list[dict[str, Any]] = Field(..., min_length=1, description="每条含 source_file、doc_id、series_code 等")
