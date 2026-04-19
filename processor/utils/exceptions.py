"""查询流水线异常。"""

from __future__ import annotations


class QueryProcessError(RuntimeError):
    """节点或编排失败时抛出。"""

    def __init__(self, message: str, node_name: str = "") -> None:
        super().__init__(message)
        self.node_name = node_name


__all__ = ["QueryProcessError"]
