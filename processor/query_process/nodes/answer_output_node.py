"""基于检索上下文的 LLM 答案生成。"""

from __future__ import annotations

from typing import Any

from prompts.query_prompt import ANSWER_PROMPT
from processor.query_process.base import BaseNode
from processor.query_state import CitationRef, QueryGraphState
from utils.client import get_llm_client


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        existing = (state.get("answer") or "").strip()
        if existing:
            state["prompt"] = ""
            state.setdefault("citations", [])
            return state

        q = (
            state.get("retrieval_query")
            or state.get("rewritten_query")
            or state.get("user_query")
            or ""
        ).strip()
        catalog = state.get("catalog_confirmed") or []
        catalog_hint = "、".join(str(x) for x in catalog if x) or "（未限定课程/题库）"

        reranked = state.get("reranked_docs") or []
        context_str, _ = self._format_context(reranked, self.config.max_context_chars)
        history_str = self._format_history(state)

        prompt = ANSWER_PROMPT.format(
            catalog_hint=catalog_hint,
            context=context_str or "（检索未命中任何切片）",
            history=history_str or "（无）",
            question=q,
        )
        state["prompt"] = prompt

        if state.get("stream"):
            state["citations"] = self._build_citations(reranked)
            state["answer"] = ""
            return state

        try:
            llm = get_llm_client()
            ans = llm.chat(prompt=prompt, json_mode=False).strip()
        except Exception as e:
            self.logger.exception("答案生成失败: %s", e)
            state["answer"] = f"暂时无法生成回答（{e}）"
            return state

        state["answer"] = ans or "（空响应）"
        state["citations"] = self._build_citations(reranked)
        return state

    def _format_context(
        self,
        docs: list[dict[str, Any]],
        max_chars: int,
    ) -> tuple[str, int]:
        lines: list[str] = []
        used = 0
        for i, d in enumerate(docs, 1):
            meta = f"[片段{i}]"
            cid = d.get("chunk_id") or ""
            if cid:
                meta += f" chunk_id={cid}"
            cn = d.get("course_name") or ""
            if cn:
                meta += f" 课程={cn}"
            sf = d.get("source_file") or ""
            if sf:
                meta += f" 文件={sf}"
            sc = d.get("score")
            if sc is not None:
                meta += f" 得分={float(sc):.4f}"
            body = str(d.get("content") or "")
            line = f"{meta}\n{body}"
            sep = 2 if lines else 0
            if used + sep + len(line) > max_chars:
                break
            lines.append(line)
            used += sep + len(line)
        return "\n\n".join(lines), max_chars - used

    def _format_history(self, state: QueryGraphState) -> str:
        msgs = state.get("messages")
        if not msgs:
            return ""
        parts: list[str] = []
        for m in msgs[-4:]:
            if isinstance(m, dict):
                role = m.get("role", "")
                text = (m.get("content") or m.get("text") or "").strip()
                if text:
                    parts.append(f"{role}:{text[:800]}")
        return "\n".join(parts)

    def _build_citations(self, docs: list[dict[str, Any]]) -> list[CitationRef]:
        out: list[CitationRef] = []
        for d in docs[:20]:
            c = str(d.get("content") or "")[:400]
            if not c:
                continue
            out.append(
                {
                    "source_file": str(d.get("source_file") or ""),
                    "course_name": d.get("course_name"),
                    "chapter_name": d.get("chapter_name"),
                    "project_name": d.get("project_name"),
                    "snippet": c,
                    "chunk_id": str(d.get("chunk_id") or ""),
                }
            )
        return out


__all__ = ["AnswerOutputNode"]
