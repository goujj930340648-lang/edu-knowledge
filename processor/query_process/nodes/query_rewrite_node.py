"""多轮场景：将口语追问改写为含完整实体的独立检索句（指代消解）。"""

from __future__ import annotations

import os
from typing import Any

from prompts.query_prompt import QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER
from processor.query_process.base import BaseNode
from processor.query_state import QueryGraphState
from utils.client import get_llm_client


def _rewrite_disabled() -> bool:
    v = (os.environ.get("QUERY_REWRITE_DISABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _history_block(state: QueryGraphState) -> str:
    msgs = state.get("messages")
    if not msgs:
        return "（无）"
    lines: list[str] = []
    for m in msgs[-10:]:
        if isinstance(m, dict):
            role = str(m.get("role", ""))
            text = (m.get("content") or m.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text[:1200]}")
        else:
            lines.append(str(m)[:1200])
    return "\n".join(lines) if lines else "（无）"


class QueryRewriteNode(BaseNode):
    name = "query_rewrite_node"

    def process(self, state: QueryGraphState) -> dict[str, Any]:
        if _rewrite_disabled():
            return {}
        msgs = state.get("messages")
        if not msgs:
            return {}
        user_query = (state.get("user_query") or "").strip()
        draft = (
            state.get("retrieval_query")
            or state.get("rewritten_query")
            or user_query
        ).strip()
        if not user_query:
            return {}

        history_block = _history_block(state)
        prompt = f"{QUERY_REWRITE_SYSTEM}\n\n{QUERY_REWRITE_USER.format(history_block=history_block, retrieval_draft=draft, user_query=user_query)}"
        try:
            llm = get_llm_client()
            raw = llm.chat(prompt=prompt, json_mode=False).strip()
        except Exception as e:
            self.logger.warning("查询改写 LLM 失败，沿用检索初稿: %s", e)
            return {}

        line = raw.splitlines()[0].strip() if raw else ""
        line = line.strip(" \t\"'「」")
        if len(line) < 4:
            return {}
        out: dict[str, Any] = {
            "retrieval_query": line,
            "query_rewrite_note": f"{draft[:80]} → {line[:80]}",
        }
        self.logger.debug("query rewrite: %s", out.get("query_rewrite_note"))
        return out


__all__ = ["QueryRewriteNode"]
