"""
教育内容领域模型：分为「通用元数据」与「类型专属结构」两层。

- 元数据：检索过滤、溯源、与 Milvus 标量字段对齐。
- 结构体：题目解析、课程展示等业务字段，按 content_type 选用对应块。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schema.metadata import DocumentClass, QuestionTypeCategory


class ContentType(str, Enum):
    COURSE_INFO = "课程介绍"
    DOC_CHUNK = "文档片段"
    QUESTION = "题目"


class ContentMetadata(BaseModel):
    """通用元数据：检索、过滤与来源溯源。"""

    content_type: ContentType = Field(..., description="内容类型")
    course_name: Optional[str] = Field(None, description="课程名称")
    project_name: Optional[str] = Field(None, description="项目名称")
    chapter_name: Optional[str] = Field(None, description="章节名称")
    bank_name: Optional[str] = Field(None, description="题库名称（题目集场景）")
    source_file: str = Field(..., description="来源文件名")
    document_class: Optional[DocumentClass] = Field(
        None,
        description="导入管线分类结果，便于与 Content_Classifier 对齐",
    )


class QuestionStructure(BaseModel):
    """题目专属结构（与《题目资料》及需求中的题库字段对齐）。"""

    question_id: Optional[str] = Field(None, description="题目编码")
    question_type: Optional[Union[str, QuestionTypeCategory]] = Field(
        None,
        description="题型：可与 QuestionTypeCategory 一致，或保留文档原始标签",
    )
    options: Optional[List[str]] = Field(None, description="选项列表")
    answer: Optional[str] = Field(None, description="标准答案")
    analysis: Optional[str] = Field(None, description="解析")
    difficulty: Optional[str] = Field(None, description="难度（如 易/中/难 或星级）")
    knowledge_points: Optional[str] = Field(None, description="考查知识点，可多条用分号分隔")
    score: Optional[float] = Field(None, description="分值")


class CourseIntroStructure(BaseModel):
    """课程介绍专属结构（对应 课程介绍.md）。"""

    prerequisites: Optional[str] = Field(None, description="先修要求")
    learning_goals: Optional[str] = Field(None, description="学习目标")
    target_audience: Optional[str] = Field(None, description="适合人群")
    chapter_structure: Optional[List[str]] = Field(
        None,
        description="章节大纲（标题列表，与《课程介绍》排版对齐）",
    )
    project_content: Optional[str] = Field(None, description="项目实战内容简述")


class DocChunkStructure(BaseModel):
    """文档片段：通常仅依赖正文 content；此处保留可扩展附加字段。"""

    model_config = ConfigDict(extra="allow")


class EduContent(BaseModel):
    """
    统一的教育内容模型：既能存入 Milvus 做向量检索，
    也能保留结构化字段供业务展示。

    - ``metadata`` + ``content``：检索与溯源主路径。
    - ``question`` / ``course_intro`` / ``doc_chunk``：按 ``metadata.content_type`` 选用其一（或文档片段留空）。
    """

    metadata: ContentMetadata = Field(..., description="通用元数据")
    content: str = Field(..., description="用于向量化的正文内容")

    question: Optional[QuestionStructure] = Field(
        None, description="题目专属；题型为 QUESTION 时使用"
    )
    course_intro: Optional[CourseIntroStructure] = Field(
        None, description="课程介绍专属；类型为 COURSE_INFO 时使用"
    )
    doc_chunk: Optional[DocChunkStructure] = Field(
        None, description="文档片段扩展；类型为 DOC_CHUNK 时可省略或填扩展字段"
    )

    @model_validator(mode="after")
    def _hint_single_structure_block(self) -> EduContent:
        """非强制：避免多种结构体同时被误填。"""
        blocks = [
            self.question is not None,
            self.course_intro is not None,
            self.doc_chunk is not None,
        ]
        if sum(blocks) > 1:
            raise ValueError(
                "至多填写一种类型专属结构：question、course_intro、doc_chunk 三选一（或全为空）"
            )
        return self

    def to_flat_dict(self) -> Dict[str, Any]:
        """展开为单层 dict，便于写入动态字段或兼容旧接口。"""
        out: Dict[str, Any] = {
            "content": self.content,
            "content_type": self.metadata.content_type.value,
            "course_name": self.metadata.course_name,
            "project_name": self.metadata.project_name,
            "chapter_name": self.metadata.chapter_name,
            "bank_name": self.metadata.bank_name,
            "source_file": self.metadata.source_file,
        }
        if self.metadata.document_class is not None:
            out["document_class"] = self.metadata.document_class.value
        if self.question:
            qt = self.question.question_type
            qt_out = qt.value if isinstance(qt, QuestionTypeCategory) else qt
            out.update(
                {
                    "question_id": self.question.question_id,
                    "question_type": qt_out,
                    "options": self.question.options,
                    "answer": self.question.answer,
                    "analysis": self.question.analysis,
                    "difficulty": self.question.difficulty,
                    "knowledge_points": self.question.knowledge_points,
                    "score": self.question.score,
                }
            )
        if self.course_intro:
            out.update(
                {
                    "prerequisites": self.course_intro.prerequisites,
                    "learning_goals": self.course_intro.learning_goals,
                    "target_audience": self.course_intro.target_audience,
                    "chapter_structure": self.course_intro.chapter_structure,
                    "project_content": self.course_intro.project_content,
                }
            )
        if self.doc_chunk is not None:
            out["doc_chunk"] = self.doc_chunk.model_dump(exclude_unset=True)
        return out


__all__ = [
    "ContentType",
    "ContentMetadata",
    "QuestionStructure",
    "CourseIntroStructure",
    "DocChunkStructure",
    "EduContent",
    "DocumentClass",
    "QuestionTypeCategory",
]
