"""SSE 用内存队列（单进程）；生产环境应换 Redis 等。"""

from __future__ import annotations

import threading
from typing import Any, Iterator

_queues: dict[str, threading.Queue[Any]] = {}
_lock = threading.Lock()


def register_stream(task_id: str) -> threading.Queue[Any]:
    q: threading.Queue[Any] = threading.Queue()
    with _lock:
        _queues[task_id] = q
    return q


def push_chunks(task_id: str, chunks: list[str]) -> None:
    q = _queues.get(task_id)
    if not q:
        return
    for c in chunks:
        q.put(c)
    q.put(None)


def push_token(task_id: str, text: str) -> None:
    """推送单段文本（不结束流）；与 ``end_stream`` 配对使用。"""
    q = _queues.get(task_id)
    if q and text:
        q.put(text)


def push_citations_event(task_id: str, citations: list[Any]) -> None:
    """在文本流之前推送结构化引用，前端可先渲染参考来源占位。"""
    q = _queues.get(task_id)
    if q and citations:
        q.put({"_event": "citations", "citations": citations})


def end_stream(task_id: str) -> None:
    q = _queues.get(task_id)
    if q:
        q.put(None)


def iter_sse(task_id: str) -> Iterator[str]:
    import json

    q = _queues.get(task_id)
    if q is None:
        yield f"data: {json.dumps({'error': 'unknown task_id'}, ensure_ascii=False)}\n\n"
        return
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, dict) and item.get("_event") == "citations":
            yield f"data: {json.dumps({'citations': item.get('citations') or []}, ensure_ascii=False)}\n\n"
            continue
        if isinstance(item, str):
            yield f"data: {json.dumps({'text': item}, ensure_ascii=False)}\n\n"
            continue
        yield f"data: {json.dumps({'text': str(item)}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
    with _lock:
        _queues.pop(task_id, None)
