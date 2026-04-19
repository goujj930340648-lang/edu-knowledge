"""
教育知识库查询主图（与 ``knowledge/processor/query_process/main_graph`` 同构）。

- **legacy**：课程名对齐 →（需检索时）**查询改写** → 并行「Hybrid + HyDE」→ RRF → 重排序 → 答案。
- **v2**：课程名对齐 → **查询改写** → 并行「Dense + Sparse + HyDE」→ RRF → 重排序 → 答案。
"""

from __future__ import annotations

from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from processor.utils.base import BaseNode
from processor.nodes.answer import AnswerOutputNode
from processor.nodes.course_catalog import CourseCatalogNode
from processor.nodes.dense_search import DenseVectorSearchNode
from processor.nodes.hybrid_search import HybridVectorSearchNode
from processor.nodes.hyde_search import HyDeVectorSearchNode
from processor.nodes.reranker import RerankerNode
from processor.nodes.rrf_merge import RrfMergeNode
from processor.nodes.query_rewrite import QueryRewriteNode
from processor.nodes.sparse_search import SparseVectorSearchNode
from processor.query_state import QueryGraphState
from utils.milvus_search_edu import rag_mode


class MultiSearchFanoutNode(BaseNode):
    """并行检索占位节点（与 knowledge ``multi_search`` 对应）。"""

    name = "multi_search"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        return state


class JoinFaninNode(BaseNode):
    name = "join"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        return {}


def route_after_catalog(state: QueryGraphState) -> bool:
    """已有可直接返回的答案（如歧义澄清）则跳过后续检索。"""
    return bool(state.get("answer"))


def _wire_common_tail(workflow: StateGraph) -> None:
    workflow.add_edge("join", "rrf_merge_node")
    workflow.add_edge("rrf_merge_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)


def create_query_graph() -> CompiledStateGraph:
    workflow = StateGraph(QueryGraphState)
    nodes: dict[str, BaseNode] = {
        "course_catalog_node": CourseCatalogNode(),
        "query_rewrite_node": QueryRewriteNode(),
        "multi_search": MultiSearchFanoutNode(),
        "join": JoinFaninNode(),
        "rrf_merge_node": RrfMergeNode(),
        "reranker_node": RerankerNode(),
        "answer_output_node": AnswerOutputNode(),
    }
    if rag_mode() == "v2":
        nodes["dense_vector_search_node"] = DenseVectorSearchNode()
        nodes["sparse_vector_search_node"] = SparseVectorSearchNode()
        nodes["hyde_vector_search_node"] = HyDeVectorSearchNode()
    else:
        nodes["hybrid_vector_search_node"] = HybridVectorSearchNode()
        nodes["hyde_vector_search_node"] = HyDeVectorSearchNode()

    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("course_catalog_node")
    workflow.add_conditional_edges(
        "course_catalog_node",
        route_after_catalog,
        {
            True: "answer_output_node",
            False: "query_rewrite_node",
        },
    )
    workflow.add_edge("query_rewrite_node", "multi_search")
    if rag_mode() == "v2":
        workflow.add_edge("multi_search", "dense_vector_search_node")
        workflow.add_edge("multi_search", "sparse_vector_search_node")
        workflow.add_edge("multi_search", "hyde_vector_search_node")
        workflow.add_edge("dense_vector_search_node", "join")
        workflow.add_edge("sparse_vector_search_node", "join")
        workflow.add_edge("hyde_vector_search_node", "join")
    else:
        workflow.add_edge("multi_search", "hybrid_vector_search_node")
        workflow.add_edge("multi_search", "hyde_vector_search_node")
        workflow.add_edge("hybrid_vector_search_node", "join")
        workflow.add_edge("hyde_vector_search_node", "join")
    _wire_common_tail(workflow)
    return workflow.compile()


query_app = create_query_graph()

__all__ = ["create_query_graph", "query_app", "route_after_catalog"]
