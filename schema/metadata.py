"""
基础枚举与路由元数据：供 File_Router、Content_Classifier 及下游节点统一引用。

字符串取值与 ``processor.import_state`` 中的 Literal 约定保持一致，便于写入 LangGraph state。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileKind(str, Enum):
    """按扩展名识别的主流文件类型（路由分支）。"""

    MD = "md"
    DOCX = "docx"
    PDF = "pdf"


class DocumentClass(str, Enum):
    """
    Content_Classifier 输出的文档类别（与业务中文标签对应，入库时再映射到 ``ContentType``）。

    - ``syllabus``: 课程大纲
    - ``lecture``: 教学讲义
    - ``question_bank``: 题目集 / 题库
    - ``project``: 项目实战 / 实验指导（步骤、环境、代码）
    """

    SYLLABUS = "syllabus"
    LECTURE = "lecture"
    QUESTION_BANK = "question_bank"
    PROJECT = "project"


class QuestionTypeCategory(str, Enum):
    """题型枚举（与《题目资料》类文档常见字段对齐，分类器可作粗粒度提示）。"""

    SINGLE_CHOICE = "单选题"
    MULTIPLE_CHOICE = "多选题"
    TRUE_FALSE = "判断题"
    FILL_BLANK = "填空题"
    SHORT_ANSWER = "简答题"
    CALCULATION = "计算题"
    COMPREHENSIVE = "综合题"
    OTHER = "其他"


class FileRouterOutput(BaseModel):
    """
    File_Router 节点的领域输出，可写入 ``ImportGraphState`` 对应字段。

    与 TypedDict state 并行存在：便于单测、校验与 API 文档；图中仍以 dict 更新为主。
    """

    file_kind: FileKind = Field(..., description="识别出的文件类型")
    normalized_text: Optional[str] = Field(
        None,
        description="MD 原文或 Word 转出的纯文本，供分类与抽取",
    )
    conversion_engine: Optional[str] = Field(
        None,
        description="转换引擎标识（如 python-docx），便于排错",
    )
    router_error: Optional[str] = Field(
        None,
        description="路由或转换失败原因；成功时应为 None",
    )

    def to_state_patch(self) -> dict[str, Any]:
        """转为 ``ImportGraphState`` 可合并的片段（省略值为 None 的可选键）。"""
        out: dict[str, Any] = {"file_kind": self.file_kind.value}
        if self.normalized_text is not None:
            out["normalized_text"] = self.normalized_text
        if self.conversion_engine is not None:
            out["conversion_engine"] = self.conversion_engine
        if self.router_error is not None:
            out["router_error"] = self.router_error
        return out


def infer_file_kind_from_name(filename: str) -> Optional[FileKind]:
    """
    仅根据文件名后缀推断 ``FileKind``；无匹配后缀时返回 None。

    扩展名比较大小写不敏感。
    """
    lower = filename.lower().strip()
    if lower.endswith(".md") or lower.endswith(".markdown"):
        return FileKind.MD
    if lower.endswith(".docx"):
        return FileKind.DOCX
    if lower.endswith(".pdf"):
        return FileKind.PDF
    return None


__all__ = [
    "FileKind",
    "DocumentClass",
    "QuestionTypeCategory",
    "FileRouterOutput",
    "infer_file_kind_from_name",
]
