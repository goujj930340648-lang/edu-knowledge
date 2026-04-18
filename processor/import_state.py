"""
导入图状态：文件路由、内容分类、元数据抽取、语义切片与向量入库。
与查询图（query_state.QueryGraphState）字段正交，不混用。
"""

from __future__ import annotations

from typing import Any, List, Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired


FileKind = Literal["md", "docx", "pdf"]
"""File_Router 识别的后缀类别。"""

DocumentClass = Literal["syllabus", "lecture", "question_bank", "project"]
"""
Content_Classifier 输出（与业务中文标签对应，入库时再映射到 schema.ContentType）。

- syllabus: 课程大纲
- lecture: 教学讲义
- question_bank: 题目集
- project: 项目 / 实验文档
"""


class LectureMetadata(TypedDict, total=False):
    """Edu_Metadata_Extractor：讲义 / 大纲侧常用字段。"""

    course_name: NotRequired[str | None]
    chapter_name: NotRequired[str | None]
    project_name: NotRequired[str | None]
    target_audience: NotRequired[str | None]
    prerequisites: NotRequired[str | None]
    learning_goals: NotRequired[str | None]
    chapter_structure: NotRequired[List[str]]
    """章节大纲（标题列表）；与 ``CourseIntroStructure.chapter_structure`` 对齐。"""
    project_content: NotRequired[str | None]


class QuestionBankMetadata(TypedDict, total=False):
    """Edu_Metadata_Extractor：题目侧常用字段。"""

    question_type: NotRequired[str | None]
    bank_name: NotRequired[str | None]
    """题库名。"""
    question_code: NotRequired[str | None]
    """题目编码（文档级或块级由切片节点再细化）。"""


class ChunkDraft(TypedDict):
    """
    语义块（尚未向量化）：可由 Smart_Chunker、Question_Extractor 或 Syllabus_Extractor 写入。

    - ``edu_content``：题目或课程介绍等完整 ``EduContent`` 序列化结果。
    """

    chunk_id: str  # <--- [新增] 加上这一行，告诉检查器允许存在这个键
    text: str
    order: NotRequired[int]
    heading_path: NotRequired[List[str]]
    """Markdown 标题栈或 Word 大纲路径，保证块级上下文可追溯。"""
    source_span: NotRequired[dict[str, Any]]
    """页码、锚点、段落 id 等，便于调试与展示。"""
    edu_content: NotRequired[dict[str, Any]]
    """题目抽取节点写入的 ``EduContent.model_dump(mode=\"json\")``，供入库与展示。"""


class ImportGraphState(TypedDict):
    """
    LangGraph 导入图状态。

    必填字段保证任务可追踪；其余由各节点按顺序或条件分支写入。
    """

    # --- 任务与输入 ---
    job_id: str
    original_filename: str
    task_priority: NotRequired[int]
    """任务优先级（数值越大越优先）；用于限流 / 队列调度时参考。"""
    source_path: NotRequired[str]
    """本地路径或可读 URI；若仅内存上传可为空。"""
    source_bytes_key: NotRequired[str]
    """对象存储或临时键，避免大文件塞进 state（可选）。"""
    api_doc_type: NotRequired[str]
    """HTTP 导入时指定：``course_doc`` / ``project_doc``，写入 Milvus 标量 ``doc_type`` 并覆盖分类分支。"""
    ingest_task_id: NotRequired[str]
    """与 ``ingest_task`` 关联，便于文档导入后回写 Mongo。"""

    # --- File_Router ---
    file_kind: NotRequired[FileKind]
    normalized_text: NotRequired[str]
    """MD 原文或 Word 转换后的全文，供分类与抽取。"""
    conversion_engine: NotRequired[str]
    """Word 转换引擎标识（便于排错）。"""
    router_error: NotRequired[str]
    """路由或转换失败时写入；PDF 未实现也可在此说明。"""

    # --- Content_Classifier ---
    document_class: NotRequired[DocumentClass]
    classifier_confidence: NotRequired[float]
    classifier_raw: NotRequired[str]
    """LLM 原始输出片段，便于审计（可选）。"""

    # --- Edu_Metadata_Extractor（按 document_class 选用，可并存为「文档级默认」）---
    lecture_metadata: NotRequired[LectureMetadata]
    question_bank_metadata: NotRequired[QuestionBankMetadata]

    # --- Smart_Chunker ---
    chunks: NotRequired[List[ChunkDraft]]

    # --- Vector_Indexer ---
    indexed_records: NotRequired[List[dict[str, Any]]]
    """待写入或已写入的扁平记录（如 EduContent.to_flat_dict() + vector + content_hash）。"""
    vector_ids: NotRequired[List[str]]
    """Milvus 主键或业务 id 列表，与成功写入的记录顺序对齐（可选）。"""
    indexed_catalog_records: NotRequired[List[dict[str, Any]]]
    """``MILVUS_RAG_MODE=v2`` 时写入名称表（如 ``edu_knowledge_item_names_v1``）的行（可选）。"""
    catalog_vector_ids: NotRequired[List[str]]
    """名称表主键 id 列表，与 ``indexed_catalog_records`` 顺序对齐（可选）。"""
    is_success: NotRequired[bool]
    """向量入库是否全部成功（无致命错误且至少完成一次有效写入或空跑成功）。"""

    # --- 汇总 ---
    errors: NotRequired[List[str]]
    warnings: NotRequired[List[str]]


__all__ = [
    "FileKind",
    "DocumentClass",
    "LectureMetadata",
    "QuestionBankMetadata",
    "ChunkDraft",
    "ImportGraphState",
]
