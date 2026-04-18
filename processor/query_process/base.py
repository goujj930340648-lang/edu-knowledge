"""查询节点基类（与 knowledge ``query_process.base`` 同构，无 SSE/任务队列依赖）。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from processor.query_process.config import QueryConfig, get_query_config
from processor.query_process.exceptions import QueryProcessError

T = TypeVar("T")


class BaseNode(ABC):
    """LangGraph 节点：子类实现 ``process``。"""

    name: str = "base_node"

    def __init__(self, config: QueryConfig | None = None) -> None:
        self.config = config or get_query_config()
        self.logger = logging.getLogger(f"edu_query.{self.name}")

    def __call__(self, state: T) -> T | dict[str, Any]:
        try:
            self.logger.info("--- %s 开始 ---", self.name)
            out = self.process(state)
            self.logger.info("--- %s 完成 ---", self.name)
            return out
        except Exception as e:
            self.logger.exception("%s 执行失败", self.name)
            raise QueryProcessError(f"{self.name}: {e}", node_name=self.name) from e

    @abstractmethod
    def process(self, state: T) -> T | dict[str, Any]:
        raise NotImplementedError


__all__ = ["BaseNode"]
