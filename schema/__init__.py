"""教育知识库领域模型与路由元数据。"""

from schema.edu_content import (
    ContentMetadata,
    ContentType,
    CourseIntroStructure,
    DocChunkStructure,
    EduContent,
    QuestionStructure,
)
from schema.metadata import (
    DocumentClass,
    FileKind,
    FileRouterOutput,
    QuestionTypeCategory,
    infer_file_kind_from_name,
)

__all__ = [
    "ContentMetadata",
    "ContentType",
    "CourseIntroStructure",
    "DocChunkStructure",
    "EduContent",
    "QuestionStructure",
    "DocumentClass",
    "FileKind",
    "FileRouterOutput",
    "QuestionTypeCategory",
    "infer_file_kind_from_name",
]
