"""
导入管线 LangGraph：File_Router → Content_Classifier → 按文档类分支抽取 → Vector_Indexer。

``project_extractor`` 与 ``lecture_extractor`` 共用同一套基于标题的切片逻辑（见 ``lecture_extractor_node``）。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from processor.content_classifier import content_classifier_node
from processor.file_router import file_router_node
from processor.import_state import ImportGraphState
from processor.lecture_extractor import lecture_extractor_node
from processor.question_extractor import question_extractor_node
from processor.syllabus_extractor import syllabus_extractor_node
from processor.vector_indexer import vector_indexer_node


def routing_logic(state: ImportGraphState) -> str:
    """根据 ``document_class`` 选择抽取节点；未知或缺省走通用讲义占位。"""
    doc_class = state.get("document_class")
    mapping = {
        "question_bank": "question_extractor",
        "syllabus": "syllabus_extractor",
        "lecture": "lecture_extractor",
        "project": "project_extractor",
    }
    return mapping.get(doc_class, "lecture_extractor")


def build_import_workflow() -> StateGraph:
    """构建未编译的 StateGraph，便于测试或插入检查点。"""
    g = StateGraph(ImportGraphState)
    g.add_node("file_router", file_router_node)
    g.add_node("content_classifier", content_classifier_node)
    g.add_node("question_extractor", question_extractor_node)
    g.add_node("syllabus_extractor", syllabus_extractor_node)
    g.add_node("lecture_extractor", lecture_extractor_node)
    # 项目文档与讲义共用标题切片；节点名不同仅便于图上区分分支
    g.add_node("project_extractor", lecture_extractor_node)
    g.add_node("vector_indexer", vector_indexer_node)

    g.add_edge(START, "file_router")
    g.add_edge("file_router", "content_classifier")
    g.add_conditional_edges(
        "content_classifier",
        routing_logic,
        {
            "question_extractor": "question_extractor",
            "syllabus_extractor": "syllabus_extractor",
            "lecture_extractor": "lecture_extractor",
            "project_extractor": "project_extractor",
        },
    )
    for name in (
        "question_extractor",
        "syllabus_extractor",
        "lecture_extractor",
        "project_extractor",
    ):
        g.add_edge(name, "vector_indexer")
    g.add_edge("vector_indexer", END)
    return g


def build_import_graph():
    """编译后的可执行图（``invoke`` / ``ainvoke``）。"""
    return build_import_workflow().compile()


__all__ = [
    "build_import_graph",
    "build_import_workflow",
    "lecture_extractor_node",
    "routing_logic",
]
