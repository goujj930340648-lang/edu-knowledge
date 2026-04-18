"""
本地 BGE-M3（FlagEmbedding）：**只加载本机或 HuggingFace 缓存里的模型**，不调 OpenAI 类 Embedding API。

优先使用你机器上的目录（与常见 ``hub/models--BAAI--bge-m3`` 布局一致），通过环境变量配置即可。

环境变量（与常见部署习惯对齐）：

- ``BGE_M3_PATH``：**优先**。若目录存在，则作为 ``model_name_or_path`` 传入 ``BGEM3FlagModel``（完全离线）。
- ``BGE_M3_MODEL_NAME`` / ``BGE_M3``：未设路径或路径不存在时，作为模型名（可为 ``BAAI/bge-m3``，由 transformers 走缓存）。
- ``BGE_DEVICE``：如 ``cuda:0``；不设则由 FlagEmbedding 自行选设备。
- ``BGE_FP16`` / ``BGE_M3_USE_FP16``：半精度开关（``1`` / ``true`` / ``True`` 等）。
- ``BGE_M3_BATCH``：encode 批大小，默认 ``12``。
- ``ITEM_NAME_DIAG``：设为 ``1`` 时在首次加载打印解析到的模型路径与设备（便于排查）。

依赖：``pip install FlagEmbedding torch``
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

_encode_sem: threading.BoundedSemaphore | None = None


def _encode_semaphore() -> threading.BoundedSemaphore:
    global _encode_sem
    if _encode_sem is None:
        try:
            n = int(os.environ.get("BGE_MAX_CONCURRENT", "4"))
        except ValueError:
            n = 4
        n = max(1, min(n, 64))
        _encode_sem = threading.BoundedSemaphore(n)
    return _encode_sem


def _parse_bool_env(keys: tuple[str, ...], default: bool = False) -> bool:
    for key in keys:
        v = os.environ.get(key, "")
        if not v.strip():
            continue
        s = v.strip().lower()
        if s in ("0", "false", "no", "off"):
            return False
        return s in ("1", "true", "yes", "on")
    return default


def _resolve_m3_model_path() -> str:
    """优先本地目录，否则为 HF 模型名。"""
    raw_path = os.environ.get("BGE_M3_PATH", "").strip()
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return str(p.resolve())
    return (
        os.environ.get("BGE_M3_MODEL_NAME")
        or os.environ.get("BGE_M3")
        or "BAAI/bge-m3"
    ).strip()


def _resolve_devices() -> Optional[str]:
    d = os.environ.get("BGE_DEVICE", "").strip()
    return d or None


def _normalize_sparse_dict(raw: dict[Any, Any]) -> dict[int, float]:
    """
    将 BGE-M3 ``lexical_weights`` 转为 Milvus ``SPARSE_FLOAT_VECTOR`` 可用的 ``{int: float}``。
    """
    out: dict[int, float] = {}
    for k, v in raw.items():
        if isinstance(k, int):
            idx = k
        elif isinstance(k, str):
            try:
                idx = int(k)
            except ValueError:
                continue
        else:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != 0.0:
            out[idx] = fv
    return out


class LocalBGEClient:
    """封装 ``BGEM3FlagModel``：批量稠密 + 稀疏；名称表仅稠密。"""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        use_fp16: bool | None = None,
        devices: str | None = None,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError(
                "未安装 FlagEmbedding，无法加载本地 BGE-M3。请执行: pip install FlagEmbedding torch"
            ) from e

        name = (model_name_or_path or _resolve_m3_model_path()).strip()
        if use_fp16 is None:
            use_fp16 = _parse_bool_env(("BGE_M3_USE_FP16", "BGE_FP16"), default=True)
        dev = devices if devices is not None else _resolve_devices()

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if dev:
            kwargs["devices"] = dev

        self._model = BGEM3FlagModel(name, **kwargs)

        if _parse_bool_env(("ITEM_NAME_DIAG",), default=False):
            print(
                f"[LocalBGEClient] model_name_or_path={name!r} use_fp16={use_fp16!r} devices={dev!r}",
                flush=True,
            )

    def embed_documents_dense_sparse(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """稠密 + 稀疏（Milvus hybrid）。"""
        if not texts:
            return [], []
        with _encode_semaphore():
            return self._embed_documents_dense_sparse_impl(texts)

    def _embed_documents_dense_sparse_impl(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        batch_size = max(1, int(os.environ.get("BGE_M3_BATCH", "12")))
        out_dense: list[list[float]] = []
        out_sparse: list[dict[int, float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            output = self._model.encode(
                batch,
                batch_size=len(batch),
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = output["dense_vecs"]
            lexical = output["lexical_weights"]
            dense_arr = np.asarray(dense)
            for row in dense_arr:
                out_dense.append([float(x) for x in np.asarray(row).tolist()])
            for lex in lexical:
                if not isinstance(lex, dict):
                    raise TypeError("lexical_weights 元素应为 dict")
                out_sparse.append(_normalize_sparse_dict(lex))
        if len(out_dense) != len(texts) or len(out_sparse) != len(texts):
            raise RuntimeError(
                f"BGE-M3 输出长度异常: dense={len(out_dense)}, sparse={len(out_sparse)}, n={len(texts)}"
            )
        return out_dense, out_sparse

    def embed_items_dense(self, texts: list[str]) -> list[list[float]]:
        """名称表：仅稠密向量。"""
        if not texts:
            return []
        with _encode_semaphore():
            return self._embed_items_dense_impl(texts)

    def _embed_items_dense_impl(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, int(os.environ.get("BGE_M3_BATCH", "12")))
        out_dense: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            output = self._model.encode(
                batch,
                batch_size=len(batch),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense_arr = np.asarray(output["dense_vecs"])
            for row in dense_arr:
                out_dense.append([float(x) for x in np.asarray(row).tolist()])
        return out_dense

    def embed_documents_dense_only(self, texts: list[str]) -> list[list[float]]:
        """legacy 单表：只写 ``vector`` 列时使用。"""
        return self.embed_items_dense(texts)


def should_use_local_bge_embedding() -> bool:
    """
    是否用本地 ``BGEM3FlagModel`` 做向量（不调 OpenAI Embedding API）。

    - 显式 ``EMBEDDING_BACKEND=openai``（或 ``http`` / ``api``）→ 走 API。
    - ``EMBEDDING_BACKEND=local_bge_m3`` / ``local`` / ``offline`` → 本地。
    - 已配置且存在的 ``BGE_M3_PATH`` → 本地（适合纯离线目录）。
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "").strip().lower()
    if backend in ("openai", "http", "api"):
        return False
    if backend in ("local_bge_m3", "local", "offline"):
        return True
    p = os.environ.get("BGE_M3_PATH", "").strip()
    if p:
        if Path(p).exists():
            return True
    return False


__all__ = [
    "LocalBGEClient",
    "should_use_local_bge_embedding",
]
