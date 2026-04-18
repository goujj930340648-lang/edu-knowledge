"""意图分流与问答编排（轻量规则 + 可选 LLM 回退 + 知识库查询图）。"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from config.settings import get_settings
from storage.mongo_db import get_mongo_db
from storage.repository import insert_chat_messages, search_courses, search_document_chunks_by_text, search_questions

_INTENTS = frozenset({"course_intro", "question_search", "doc_search", "knowledge_qa"})


def _truth_env(*keys: str) -> bool:
    for k in keys:
        v = (os.environ.get(k) or "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def _classify_intent_rules(q: str) -> str:
    if not q:
        return "knowledge_qa"
    if any(
        k in q
        for k in (
            "有哪些课程",
            "课程介绍",
            "课程列表",
            "适合人群",
            "学习目标",
            "系列",
            "模块",
        )
    ) and not any(k in q for k in ("题目", "题库", "选择题")):
        return "course_intro"
    if any(
        k in q
        for k in (
            "题目",
            "题库",
            "选择题",
            "判断题",
            "多选题",
            "简答题",
            "刷题",
        )
    ):
        return "question_search"
    if any(k in q for k in ("文档", "讲义", "章节", "哪一节", "课件")):
        return "doc_search"
    return "knowledge_qa"


def _intent_ambiguous(q: str) -> bool:
    has_course = any(x in q for x in ("课程", "系列", "模块", "大纲", "课"))
    has_q = any(x in q for x in ("题", "题库", "练习", "刷题"))
    has_doc = any(x in q for x in ("文档", "讲义", "章节", "课件"))
    if has_course and has_q:
        return True
    if has_doc and has_course:
        return True
    return False


def _classify_intent_llm(user_query: str) -> str | None:
    q = (user_query or "").strip()[:900]
    if not q:
        return None
    try:
        from utils.client import get_llm_client

        prompt = (
            "你是意图分类器。仅从下列四选一，输出一个 JSON 对象，键 intent，值为字符串：\n"
            "- course_intro：查课程系列、适合人群、学习目标、模块等\n"
            "- question_search：查题目、题库、题型\n"
            "- doc_search：查讲义、文档、章节切片\n"
            "- knowledge_qa：综合知识问答、需检索后生成\n"
            f"用户问题：{q}\n"
            '示例：{"intent":"course_intro"}'
        )
        raw = get_llm_client().chat(prompt=prompt, json_mode=True).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        tag = str(data.get("intent") or "").strip()
        if tag in _INTENTS:
            return tag
    except Exception:
        return None
    return None


def classify_intent(user_query: str) -> str:
    q = (user_query or "").strip()
    rules = _classify_intent_rules(q)
    s = get_settings()
    if s.intent_use_llm:
        llm = _classify_intent_llm(q)
        return llm or rules
    if _intent_ambiguous(q):
        llm = _classify_intent_llm(q)
        if llm:
            return llm
    return rules


def _kw(q: str) -> str:
    q = re.sub(r"[？?！!。，、\s]+", " ", q).strip()
    return q[:80] if q else ""


def run_chat_sync(
    user_query: str,
    *,
    session_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    s = get_settings()
    uri = s.resolved_mongo_uri()
    intent = classify_intent(user_query)
    tid = (task_id or "").strip() or str(uuid.uuid4())
    sid = (session_id or "").strip() or str(uuid.uuid4())

    out: dict[str, Any] = {
        "task_id": tid,
        "session_id": sid,
        "intent": intent,
        "answer": "",
        "citations": [],
        "structured": None,
    }

    if not uri:
        out["answer"] = "MongoDB 未配置，无法完成结构化检索与历史写入。"
        return out

    db = get_mongo_db(uri, s.mongodb_database)

    if intent == "course_intro":
        rows = search_courses(db, _kw(user_query), limit=30)
        out["structured"] = {"courses": rows}
        if not rows:
            out["answer"] = "未找到匹配的课程系列（请先导入课程目录）。"
        else:
            lines = [f"- {r.get('title', '')}（{r.get('series_code', '')}）" for r in rows[:15]]
            out["answer"] = "与课程目录相关的系列如下：\n" + "\n".join(lines)
        insert_chat_messages(db, sid, tid, user_query, out["answer"], citations=[], intent=intent)
        return out

    if intent == "question_search":
        rows = search_questions(db, keyword=_kw(user_query), limit=20)
        out["structured"] = {"questions": rows}
        if not rows:
            out["answer"] = "未找到匹配题目（请先导入题库）。"
        else:
            snippets = []
            for r in rows[:8]:
                stem = (r.get("stem") or "")[:200].replace("\n", " ")
                snippets.append(f"- [{r.get('question_code','')}] {stem}")
            out["answer"] = "检索到的题目示例：\n" + "\n".join(snippets)
        insert_chat_messages(db, sid, tid, user_query, out["answer"], citations=[], intent=intent)
        return out

    if intent == "doc_search":
        kw = _kw(user_query)
        vec_struct: list[dict[str, Any]] = []
        vec_parts: list[str] = []
        if kw and os.environ.get("MILVUS_RAG_MODE", "").strip().lower() == "v2":
            try:
                from services.vector_doc_search import search_documents_with_hydrate

                hits = search_documents_with_hydrate(
                    db,
                    kw,
                    doc_type=None,
                    limit=10,
                    use_hyde=_truth_env("CHAT_DOC_SEARCH_HYDE", "SEARCH_DOCUMENT_HYDE"),
                    use_rerank=_truth_env("CHAT_DOC_SEARCH_RERANK", "SEARCH_DOCUMENT_RERANK"),
                )
                for h in hits:
                    ch = h.get("chunk") if isinstance(h.get("chunk"), dict) else {}
                    ent = h.get("milvus") if isinstance(h.get("milvus"), dict) else {}
                    text = (ch.get("chunk_text") or ent.get("content") or "").strip()
                    if not text:
                        continue
                    sp = ch.get("section_path")
                    title = ""
                    if isinstance(sp, list) and sp:
                        title = " / ".join(str(x) for x in sp[:4])
                    if not title:
                        title = str(ent.get("source_file") or ch.get("source_file") or "")
                    vec_parts.append(f"[{title}]\n{text[:520]}")
                    vec_struct.append({"vector": ent, "chunk": ch})
            except Exception:
                pass
        mongo_rows = search_document_chunks_by_text(db, kw, limit=15) if kw else []
        out["structured"] = {
            "mongo_chunks": mongo_rows,
            "vector_hits": vec_struct,
        }
        if vec_parts:
            out["answer"] = "（Milvus 语义检索 + Mongo 回表）\n\n" + "\n\n---\n\n".join(
                vec_parts[:8]
            )
        elif mongo_rows:
            parts = []
            for r in mongo_rows[:6]:
                t = (r.get("chunk_text") or "")[:400]
                cid = str(r.get("chunk_id") or "")
                short = (cid[:12] + "…") if len(cid) > 12 else cid
                parts.append(f"[{short}]\n{t}")
            out["answer"] = "（Mongo 全文匹配）\n\n" + "\n\n---\n\n".join(parts)
        else:
            out["answer"] = (
                "未命中文档切片。请先导入文档并确认 MILVUS_RAG_MODE=v2 且已向量化入库。"
            )
        insert_chat_messages(db, sid, tid, user_query, out["answer"], citations=[], intent=intent)
        return out

    from processor.query_process.main_graph import query_app

    state: dict[str, Any] = {"user_query": user_query}
    if messages:
        state["messages"] = messages
    graph_out = query_app.invoke(state)
    out["answer"] = (graph_out.get("answer") or "").strip() or "（无回答）"
    try:
        from services.citation_assets import enrich_citations_with_images

        out["citations"] = enrich_citations_with_images(
            db, list(graph_out.get("citations") or [])
        )
    except Exception:
        out["citations"] = graph_out.get("citations") or []
    insert_chat_messages(
        db,
        sid,
        tid,
        user_query,
        out["answer"],
        citations=out["citations"],
        intent=intent,
    )
    return out


def run_chat_stream_chunks(
    user_query: str,
    *,
    session_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
) -> list[str]:
    """非 token 级流式：按块切分最终文本，供 SSE 推送。"""
    res = run_chat_sync(user_query, session_id=session_id, messages=messages, task_id=task_id)
    text = res.get("answer") or ""
    chunk_size = 48
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or ["（空）"]
