"""SSE 流式对话：检索走 LangGraph；最终答案用 LLM ``chat_stream`` 逐 delta 推送。"""

from __future__ import annotations

from typing import Any

from app.streaming import end_stream, push_citations_event, push_token
from config.settings import get_settings
from storage.mongo_db import get_mongo_db
from storage.repository import insert_chat_messages

from services.chat_engine import classify_intent, run_chat_sync


def _stream_text(task_id: str, text: str, *, chunk: int = 24) -> None:
    t = text or ""
    if not t:
        push_token(task_id, "（空）")
        return
    for i in range(0, len(t), chunk):
        push_token(task_id, t[i : i + chunk])


def run_chat_sse_worker(
    task_id: str,
    user_query: str,
    *,
    session_id: str,
    messages: list[dict[str, Any]] | None,
) -> None:
    s = get_settings()
    uri = s.resolved_mongo_uri()
    intent = classify_intent(user_query)
    tid = task_id
    sid = session_id

    if not uri:
        push_token(tid, "MongoDB 未配置，无法完成对话。")
        end_stream(tid)
        return

    db = get_mongo_db(uri, s.mongodb_database)

    if intent != "knowledge_qa":
        try:
            res = run_chat_sync(
                user_query,
                session_id=sid,
                messages=messages,
                task_id=tid,
            )
            _stream_text(tid, res.get("answer") or "")
        except Exception as e:
            push_token(tid, f"【错误】{e}")
        end_stream(tid)
        return

    try:
        from processor.query_process.main_graph import query_app
        from utils.client import get_llm_client

        from services.citation_assets import enrich_citations_with_images

        state: dict[str, Any] = {
            "user_query": user_query,
            "stream": True,
        }
        if messages:
            state["messages"] = messages
        graph_out = query_app.invoke(state)
        ans = (graph_out.get("answer") or "").strip()
        citations = list(graph_out.get("citations") or [])
        citations = enrich_citations_with_images(db, citations)
        if citations:
            push_citations_event(tid, citations)

        if ans:
            _stream_text(tid, ans)
            insert_chat_messages(
                db,
                sid,
                tid,
                user_query,
                ans,
                citations=citations,
                intent=intent,
            )
            end_stream(tid)
            return

        prompt = (graph_out.get("prompt") or "").strip()
        if not prompt:
            msg = (graph_out.get("error") or "").strip() or "（无法生成：缺少提示词）"
            push_token(tid, msg)
            insert_chat_messages(
                db, sid, tid, user_query, msg, citations=citations, intent=intent
            )
            end_stream(tid)
            return

        llm = get_llm_client()
        parts: list[str] = []
        for delta in llm.chat_stream(prompt, json_mode=False):
            parts.append(delta)
            push_token(tid, delta)
        full = "".join(parts).strip() or "（空响应）"
        insert_chat_messages(
            db,
            sid,
            tid,
            user_query,
            full,
            citations=citations,
            intent=intent,
        )
    except Exception as e:
        push_token(tid, f"【错误】{e}")
        try:
            insert_chat_messages(
                db,
                sid,
                tid,
                user_query,
                f"【错误】{e}",
                citations=[],
                intent=intent,
            )
        except Exception:
            pass
    end_stream(tid)
