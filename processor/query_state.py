"""
查询图（RAG / 多轮问答）状态：用户问题、检索结果与生成输出。
不包含导入流水线（文件路由、切片、入库）相关字段。
"""

from __future__ import annotations

from typing import Any, List, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired


class CitationRef(TypedDict, total=False):
    """单条引用，用于答案脚注与可追溯展示。"""

    source_file: str
    course_name: str | None
    chapter_name: str | None
    project_name: str | None
    snippet: str
    chunk_id: NotRequired[str]
    question_id: NotRequired[str]
    image_urls: NotRequired[List[str]]
    """MinIO 预签名临时图链（与 chunk 同属文档时）。"""


class RetrievedHit(TypedDict, total=False):
    """检索命中一条（与 Milvus 标量字段对齐的常用子集）。"""

    chunk_id: NotRequired[str]
    content: str
    score: float
    source_file: NotRequired[str]
    course_name: NotRequired[str | None]
    chapter_name: NotRequired[str | None]
    project_name: NotRequired[str | None]
    content_type: NotRequired[str]


class QueryGraphState(TypedDict):
    """
    LangGraph 查询图状态。

    必填字段保证图入口可启动；其余在节点执行过程中逐步填充。
    """

    # --- 会话与输入 ---
    user_query: str
    thread_id: NotRequired[str]
    session_id: NotRequired[str]
    """与 thread 并存时的业务会话标识（可选）。"""
    messages: NotRequired[List[Any]]
    """多轮对话历史（通常为 LangChain BaseMessage 列表，类型为 Any 以避免强依赖）。"""
    task_id: NotRequired[str]
    """与异步任务 / SSE 对齐时的任务 ID（可选）。"""
    is_stream: NotRequired[bool]
    """与 ``stream`` 二选一语义：是否流式输出（查询流水线兼容 knowledge 命名）。"""

    # --- 检索准备 ---
    rewritten_query: NotRequired[str]
    """查询改写 / 澄清后的检索用语。"""
    query_rewrite_note: NotRequired[str]
    """多轮指代消解节点产生的简要说明（调试用）。"""
    retrieval_query: NotRequired[str]
    """实际发往向量库或混合检索的查询文本；缺省时可由节点回退为 rewritten_query 或 user_query。"""
    retrieval_filters: NotRequired[dict[str, Any]]
    """结构化过滤：课程名、章节、项目、内容类型等（与 Milvus 标量字段一致）。"""
    entity_names: NotRequired[List[str]]
    """LLM 从问题中抽取的课程名 / 项目名 / 题库名（原始）。"""
    catalog_confirmed: NotRequired[List[str]]
    """与名称表对齐后的高置信「课程/题库」名，用于 ``course_name`` 过滤。"""
    catalog_options: NotRequired[List[str]]
    """名称对齐歧义时的候选列表（打断并提示用户）。"""

    # --- 与 knowledge query_process 对齐的中间结果（Milvus hit 形态） ---
    embedding_chunks: NotRequired[List[Any]]
    """主查询向量检索命中（hybrid 或 legacy dense）。"""
    dense_embedding_chunks: NotRequired[List[Any]]
    """v2：仅稠密路命中（与稀疏、HyDE 并行后 RRF）。"""
    sparse_embedding_chunks: NotRequired[List[Any]]
    """v2：仅稀疏路命中。"""
    hyde_embedding_chunks: NotRequired[List[Any]]
    """HyDE 假设文档检索命中。"""
    rrf_chunks: NotRequired[List[Any]]
    """RRF 融合后的 (hit, score) 或 hit 列表（见 ``rrf_merge_node``）。"""
    reranked_docs: NotRequired[List[dict[str, Any]]]
    """重排序并截断后的文档列表（供生成提示词）。"""
    prompt: NotRequired[str]
    """调试用：最终送入 LLM 的完整提示词。"""

    # --- 检索结果 ---
    retrieved_hits: NotRequired[List[RetrievedHit]]
    """混合检索 / 重排序后的命中列表。"""

    # --- 生成与引用 ---
    citations: NotRequired[List[CitationRef]]
    """结构化引用列表，供答案渲染与校验。"""
    answer: NotRequired[str]
    """最终或流式聚合后的回答正文。"""

    # --- 控制与错误 ---
    error: NotRequired[str]
    """单轮可恢复错误说明；严重失败时由节点写入。"""
    stream: NotRequired[bool]
    """是否走流式输出（由编排或入口设置）。"""


__all__ = [
    "CitationRef",
    "RetrievedHit",
    "QueryGraphState",
]
