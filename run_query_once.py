"""
单次查询演示：``python run_query_once.py "你的问题"``

默认会打印**整条查询图状态**（含向量检索中间结果），体积很大，便于调试。
日常只看答案请加 ``--answer-only``；需要 JSON 但去掉大块中间态请加 ``--brief``。

依赖：Milvus、ACTIVE_API_KEY、与入库一致的 Embedding 配置；v2 需本地 BGE-M3。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# 调试期才需要的大字段（向量命中、RRF、完整 prompt 等），--brief 时会去掉
_VERBOSE_STATE_KEYS = frozenset(
    {
        "embedding_chunks",
        "hyde_embedding_chunks",
        "rrf_chunks",
        "reranked_docs",
        "prompt",
    }
)


def _brief_state(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in _VERBOSE_STATE_KEYS}


def main() -> int:
    p = argparse.ArgumentParser(description="教育知识库查询图单次调用")
    p.add_argument("query", nargs="?", default="", help="用户问题")
    p.add_argument("-q", "--question", default="", help="用户问题（与位置参数二选一）")
    p.add_argument(
        "-a",
        "--answer-only",
        action="store_true",
        help="只打印最终 answer 文本（不输出 JSON）",
    )
    p.add_argument(
        "-b",
        "--brief",
        action="store_true",
        help="打印精简 JSON：去掉向量检索中间态与完整 prompt（仍含 citations / retrieved_hits）",
    )
    args = p.parse_args()
    q = (args.question or args.query or "").strip()
    if not q:
        print("请传入问题，例如: python run_query_once.py 什么是递归？", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from processor.query_process.main_graph import query_app

    out = query_app.invoke({"user_query": q})

    if args.answer_only:
        print((out.get("answer") or "").strip())
        return 0

    payload: dict[str, Any] = out
    if args.brief:
        payload = _brief_state(dict(out))

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
