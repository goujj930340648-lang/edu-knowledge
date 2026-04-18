"""
本地 BGE Reranker（FlagEmbedding）：用于检索后重排，与导入图无强依赖，供查询侧按需调用。

环境变量（与常见本机部署习惯对齐）：

- ``BGE_RERANKER_LARGE``：模型名或**本机目录**（如 HuggingFace 缓存路径）
- ``BGE_RERANKER_DEVICE``：如 ``cuda:0``（不设则由库默认）
- ``BGE_RERANKER_FP16``：``1`` / ``true`` 等

依赖：``pip install FlagEmbedding torch``
"""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from typing import Any


def _parse_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "")
    if not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_devices() -> Any:
    d = os.environ.get("BGE_RERANKER_DEVICE") or os.environ.get("BGE_DEVICE")
    if not d or not str(d).strip():
        return None
    return str(d).strip()


def _patch_sequence_classification_from_pretrained() -> None:
    """
    FlagEmbedding 的 Reranker 在 ``compute_score`` 里会 ``model.to(device)``。
    若 ``from_pretrained`` 走了 meta / 空壳初始化（常见于 accelerate + 新版 transformers），
    部分参数会留在 meta 上，导致 ``.to(cuda)`` 报
    ``Cannot copy out of meta tensor``。

    在仍支持该参数时强制 ``low_cpu_mem_usage=False``，让权重在 CPU 上完整实例化后再搬运。
    """
    if _parse_bool("BGE_RERANKER_SKIP_LOAD_PATCH", False):
        return
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        return

    if getattr(AutoModelForSequenceClassification, "_edu_from_pretrained_patched", False):
        return

    orig = AutoModelForSequenceClassification.from_pretrained.__func__

    def _wrapped(cls: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        try:
            sig = inspect.signature(orig)
            if "low_cpu_mem_usage" in sig.parameters:
                # 显式 False：避免默认 True / accelerate 路径留下 meta 参数
                kwargs.setdefault("low_cpu_mem_usage", False)
        except (TypeError, ValueError):
            pass
        return orig(cls, *args, **kwargs)

    AutoModelForSequenceClassification.from_pretrained = classmethod(_wrapped)
    setattr(AutoModelForSequenceClassification, "_edu_from_pretrained_patched", True)


class LocalBGERerankerClient:
    """封装 ``FlagReranker``，从环境变量读模型路径与设备。"""

    def __init__(self) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise RuntimeError(
                "未安装 FlagEmbedding，无法加载 Reranker。请执行: pip install FlagEmbedding torch"
            ) from e

        model = (
            os.environ.get("BGE_RERANKER_LARGE")
            or os.environ.get("BGE_RERANKER_MODEL")
            or "BAAI/bge-reranker-v2-m3"
        ).strip()
        if not model:
            raise RuntimeError("未配置 BGE_RERANKER_LARGE（或 BGE_RERANKER_MODEL）")

        _patch_sequence_classification_from_pretrained()

        use_fp16 = _parse_bool("BGE_RERANKER_FP16", True)
        devices = _resolve_devices()
        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if devices:
            kwargs["devices"] = devices
        try:
            self._model = FlagReranker(model, **kwargs)
        except TypeError:
            kwargs.pop("devices", None)
            self._model = FlagReranker(model, **kwargs)

    def compute_score(
        self, pairs: list[tuple[str, str]], *, normalize: bool = False
    ) -> list[float]:
        """``pairs`` 为 (query, passage) 列表；``normalize=True`` 时分数映射到约 ``[0,1]``。"""
        if not pairs:
            return []
        payload = [[a, b] for a, b in pairs]
        scores = self._model.compute_score(payload, normalize=normalize)
        import numpy as np

        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(x) for x in np.asarray(scores).flatten()]


@lru_cache(maxsize=1)
def get_local_bge_reranker() -> LocalBGERerankerClient:
    return LocalBGERerankerClient()


def reset_local_bge_reranker_cache() -> None:
    get_local_bge_reranker.cache_clear()


# --- 悬崖截断 / knee（与 ``RerankerNode`` 对齐，供查询图与 API 检索复用）---

_MIN_FIT_POINTS = 2


def line_fit_rmse(scores: list[float], start: int, end: int) -> float:
    n = end - start
    if n < _MIN_FIT_POINTS:
        return 0.0
    xs = list(range(n))
    ys = scores[start:end]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / ss_xx
    intercept = mean_y - slope * mean_x
    rmse = (
        sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True)) / n
    ) ** 0.5
    return rmse


def find_knee_point(scores: list[float], early_bias: float = 0.2) -> int:
    n = len(scores)
    if n < 4:
        return n
    best_cost = float("inf")
    best_knee = n
    for knee in range(1, n):
        left_rmse = line_fit_rmse(scores, 0, knee)
        right_rmse = line_fit_rmse(scores, knee, n)
        weighted_cost = (knee * left_rmse + (n - knee) * right_rmse) / n
        position_factor = 1.0 - early_bias * (1.0 - knee / n)
        cost = weighted_cost * position_factor
        if cost < best_cost:
            best_cost = cost
            best_knee = knee
    return best_knee


def find_first_cliff_cutoff(
    scores: list[float],
    gap_ratio: float,
    gap_abs: float,
) -> int | None:
    if len(scores) < _MIN_FIT_POINTS:
        return None
    for i in range(len(scores) - 1):
        prev, nxt = scores[i], scores[i + 1]
        drop_abs = prev - nxt
        ratio_exceeds = prev > 0 and (drop_abs / prev) > gap_ratio
        abs_exceeds = drop_abs > gap_abs
        if ratio_exceeds or abs_exceeds:
            return i + 1
    return None


def cliff_truncation_count(
    scores: list[float],
    *,
    max_top_k: int,
    min_top_k: int,
    gap_ratio: float,
    gap_abs: float,
) -> int:
    """
    对**已按分数降序**的 ``scores``（只看前 ``max_top_k`` 项）计算保留条数；
    先找陡降点，否则 knee；再夹在 ``min_top_k`` 与 ``max_top_k`` 之间。
    """
    if not scores:
        return 0
    max_top_k = min(max_top_k, len(scores))
    min_top_k = min(min_top_k, max_top_k)
    head = [float(s) for s in scores[:max_top_k]]
    cliff_k = find_first_cliff_cutoff(head, gap_ratio, gap_abs)
    base_cutoff = cliff_k if cliff_k is not None else find_knee_point(head)
    cutoff_n = max(base_cutoff, min_top_k)
    return min(cutoff_n, max_top_k)


__all__ = [
    "LocalBGERerankerClient",
    "get_local_bge_reranker",
    "reset_local_bge_reranker_cache",
    "line_fit_rmse",
    "find_knee_point",
    "find_first_cliff_cutoff",
    "cliff_truncation_count",
]
