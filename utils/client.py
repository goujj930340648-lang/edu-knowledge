"""
LLM 客户端：OpenAI 兼容 Chat Completions（可通过 ACTIVE_API_BASE 指向自建网关）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from openai import OpenAI


class LLMClient:
    """薄封装：与导入/抽取节点约定 ``chat(prompt=..., json_mode=...)`` 形态。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ACTIVE_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "未配置 ACTIVE_API_KEY，无法调用 LLM。请在环境变量或 .env 中设置。"
            )
        base = base_url if base_url is not None else os.environ.get("ACTIVE_API_BASE")
        base = base.strip() if base else None
        self._model = (default_model or os.environ.get("LLM_DEFAULT_MODEL") or "gpt-4o-mini").strip()
        self._client = OpenAI(api_key=key, base_url=base)

    def chat(self, prompt: str, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            return ""
        return content

    def chat_stream(
        self, prompt: str, *, json_mode: bool = False
    ) -> Iterator[str]:
        """OpenAI 兼容流式输出，按服务端返回的 delta 逐段产出（通常接近 token 粒度）。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        stream = self._client.chat.completions.create(**kwargs)
        for event in stream:
            ch = event.choices[0]
            delta = ch.delta.content if ch.delta else None
            if delta:
                yield delta


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """进程内单例，避免重复构造 HTTP 客户端。"""
    return LLMClient()


def reset_llm_client_cache() -> None:
    """测试用：清除单例缓存。"""
    get_llm_client.cache_clear()


class EmbeddingClient:
    """
    OpenAI 兼容 Embeddings API（``client.embeddings.create``），
    可指向自建网关或 BGE-M3 等兼容服务。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        key = api_key if api_key is not None else (
            os.environ.get("EMBEDDING_API_KEY") or os.environ.get("ACTIVE_API_KEY", "")
        ).strip()
        if not key:
            raise RuntimeError(
                "未配置 EMBEDDING_API_KEY（或 ACTIVE_API_KEY），无法生成向量。"
            )
        base = base_url if base_url is not None else (
            os.environ.get("EMBEDDING_API_BASE") or os.environ.get("ACTIVE_API_BASE")
        )
        base = base.strip() if base else None
        self._model = (
            default_model
            or os.environ.get("EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ).strip()
        self._client = OpenAI(api_key=key, base_url=base)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化；单批过大时自动分片。"""
        if not texts:
            return []
        batch_size = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")))
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=chunk)
            ordered = sorted(resp.data, key=lambda x: x.index)
            for item in ordered:
                out.append(item.embedding)
        return out


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


def reset_embedding_client_cache() -> None:
    get_embedding_client.cache_clear()


def maybe_strip_proxy_for_milvus() -> None:
    """
    若设置 MILVUS_DISABLE_PROXY=1/true，则移除常见 *_PROXY 环境变量。
    部分环境下系统/IDE 配置的 HTTP 代理会导致 gRPC 走错误路径，表现为 TCP 通但 channel 永不 ready。
    """
    raw = os.environ.get("MILVUS_DISABLE_PROXY", "").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)


class MilvusIndexerClient:
    """
    Milvus 写入与按 ``content_hash`` 去重查询的薄封装。

    使用 ``MilvusClient``（与 pymilvus 2.4+ 推荐方式一致，见 knowledge 项目
    ``StorageClients._create_milvus_client``），避免旧版 ``connections.connect``
    在部分远程/跨机场景下 TCP 可达但 gRPC 握手失败的问题。
    集合需事先创建，且包含 ``vector`` 与各标量字段（见 processor.vector_indexer 注释）。
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._uri = (
            uri
            or os.environ.get("MILVUS_URI")
            or os.environ.get("MILVUS_URL")
            or "http://127.0.0.1:19530"
        ).strip()
        self._token = (token or os.environ.get("MILVUS_TOKEN") or "").strip() or None
        # 远程 Milvus 时 gRPC channel ready 可能较慢，默认 90s（可用 MILVUS_TIMEOUT 覆盖）
        raw_t = (os.environ.get("MILVUS_TIMEOUT") or "90").strip()
        try:
            self._timeout = float(timeout if timeout is not None else raw_t)
        except ValueError:
            self._timeout = 90.0
        self._client: Any = None

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        maybe_strip_proxy_for_milvus()
        from pymilvus import MilvusClient

        kw: dict[str, Any] = {"uri": self._uri, "timeout": self._timeout}
        if self._token:
            kw["token"] = self._token
        self._client = MilvusClient(**kw)

    def fetch_existing_in_field(
        self, collection_name: str, field: str, values: list[str]
    ) -> set[str]:
        """``field in (...)`` 查询，返回已存在的标量值集合（用于 ``content_hash`` / ``item_fingerprint`` 等去重）。"""
        if not values:
            return set()
        self._ensure_connected()
        escaped = [v.replace("\\", "\\\\").replace('"', '\\"') for v in values]
        expr = f"{field} in [" + ",".join(f'"{e}"' for e in escaped) + "]"
        rows = self._client.query(
            collection_name=collection_name,
            filter=expr,
            output_fields=[field],
            limit=max(len(values), 1),
        )
        return {r[field] for r in rows if r.get(field)}

    def fetch_existing_hashes(self, collection_name: str, hashes: list[str]) -> set[str]:
        """返回集合中已存在的 ``content_hash``。"""
        return self.fetch_existing_in_field(collection_name, "content_hash", hashes)

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> Any:
        """执行 insert 并 flush；返回带 ``primary_keys`` 属性的结果对象（兼容 vector_indexer）。"""
        if not data:
            raise ValueError("Milvus insert 数据为空")
        self._ensure_connected()
        res = self._client.insert(collection_name=collection_name, data=data)
        ids: list[Any]
        if isinstance(res, dict):
            ids = list(res.get("ids") or [])
        else:
            ids = list(getattr(res, "ids", None) or [])
        self._client.flush(collection_name=collection_name, timeout=self._timeout)
        return SimpleNamespace(primary_keys=ids)


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusIndexerClient:
    return MilvusIndexerClient()


def reset_milvus_client_cache() -> None:
    get_milvus_client.cache_clear()


__all__ = [
    "LLMClient",
    "get_llm_client",
    "reset_llm_client_cache",
    "EmbeddingClient",
    "get_embedding_client",
    "reset_embedding_client_cache",
    "MilvusIndexerClient",
    "get_milvus_client",
    "reset_milvus_client_cache",
    "maybe_strip_proxy_for_milvus",
]
