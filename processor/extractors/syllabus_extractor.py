"""
Syllabus_Extractor：LLM 抽取课程结构化字段后，按 ``utils.document_split``（对齐 knowledge ``document_split_node``）切分正文。
"""

from __future__ import annotations

import json
import hashlib  # [新增] 引入 hashlib
from typing import Any

from schema.edu_content import (
    ContentMetadata,
    ContentType,
    CourseIntroStructure,
    DocChunkStructure,
    EduContent,
)
from schema.metadata import DocumentClass as DocumentClassEnum

from processor.import_state import ChunkDraft, ImportGraphState, LectureMetadata
from processor.extractors.lecture_extractor import _plain_title
from utils.client import get_llm_client
from utils.document_split import split_markdown_document

SYLLABUS_EXTRACT_PROMPT = """你是一个课程大纲解析专家。请从文本中提取课程结构化信息。

### 约束：
1. 不要编造文档中不存在的内容；无法识别时对应字段填 null。
2. 章节大纲 ``chapter_structure`` 使用字符串数组，顺序与文档一致；若文档以编号列表呈现，保留每条为一项。

### 输出要求（仅输出一个 JSON 对象）：
- course_name: 课程标题
- target_audience: 适合人群
- prerequisites: 先修要求
- learning_goals: 学习目标
- chapter_structure: 章节大纲（字符串数组）
- project_content: 项目实战内容简述（无则 null）
"""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    s = str(v).strip()
    return s or None


def _normalize_chapter_structure(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines or None
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict):
                t = item.get("title") or item.get("name") or item.get("chapter")
                if t is not None:
                    out.append(str(t).strip())
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out or None
    return None


def _lecture_metadata_from_payload(data: dict[str, Any]) -> LectureMetadata:
    ch = _normalize_chapter_structure(data.get("chapter_structure"))
    meta: LectureMetadata = {}
    cn = _opt_str(data.get("course_name"))
    if cn is not None:
        meta["course_name"] = cn
    ta = _opt_str(data.get("target_audience"))
    if ta is not None:
        meta["target_audience"] = ta
    pr = _opt_str(data.get("prerequisites"))
    if pr is not None:
        meta["prerequisites"] = pr
    lg = _opt_str(data.get("learning_goals"))
    if lg is not None:
        meta["learning_goals"] = lg
    if ch is not None:
        meta["chapter_structure"] = ch
    pc = _opt_str(data.get("project_content"))
    if pc is not None:
        meta["project_content"] = pc
    return meta


def syllabus_extractor_node(state: ImportGraphState) -> dict[str, Any]:
    """
    ``document_class == syllabus``：LLM 抽取 → ``split_markdown_document`` 切分 → 多块 ``ChunkDraft``。
    首块带 ``course_intro``（COURSE_INFO），其余 ``DOC_CHUNK``。
    """
    if state.get("document_class") != "syllabus":
        return {}

    text = (state.get("normalized_text") or "").strip()
    if not text:
        errors = list(state.get("errors") or [])
        errors.append("Syllabus_Extractor：缺少 normalized_text。")
        return {"errors": errors, "chunks": []}

    llm = get_llm_client()
    prompt = SYLLABUS_EXTRACT_PROMPT + f"\n\n### 文档全文：\n{text}"
    response = llm.chat(prompt=prompt, json_mode=True)
    try:
        data = json.loads(_strip_json_fence(response))
    except Exception as e:
        errors = list(state.get("errors") or [])
        errors.append(f"Syllabus_Extractor：JSON 解析失败: {e!s}")
        return {"errors": errors, "chunks": []}

    if not isinstance(data, dict):
        errors = list(state.get("errors") or [])
        errors.append("Syllabus_Extractor：LLM 返回非对象 JSON。")
        return {"errors": errors, "chunks": []}

    course_name = _opt_str(data.get("course_name"))
    ch_list = _normalize_chapter_structure(data.get("chapter_structure"))

    course_intro = CourseIntroStructure(
        prerequisites=_opt_str(data.get("prerequisites")),
        learning_goals=_opt_str(data.get("learning_goals")),
        target_audience=_opt_str(data.get("target_audience")),
        chapter_structure=ch_list,
        project_content=_opt_str(data.get("project_content")),
    )

    lm = _lecture_metadata_from_payload(data)
    prev = dict(state.get("lecture_metadata") or {})
    prev.update(lm)

    file_title = course_name or state.get("original_filename", "document")
    if isinstance(file_title, str):
        file_title = file_title.strip() or "document"
    else:
        file_title = "document"

    try:
        raw_chunks = split_markdown_document(text, file_title)
    except ValueError as e:
        err = list(state.get("errors") or [])
        err.append(f"Syllabus_Extractor：{e}")
        return {"errors": err, "chunks": [], "lecture_metadata": prev}

    # [修改] 获取 original_filename 用于哈希
    file_name = str(state.get("original_filename", "unknown_doc"))

    chunks_out: list[ChunkDraft] = []
    order = 0
    first_piece = True

    for ch in raw_chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue

        combined_str = f"{file_name}_{order}_{content}"
        chunk_id = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

        tl = (ch.get("title") or "").strip() or file_title
        hp_raw = ch.get("heading_path")
        if isinstance(hp_raw, list) and hp_raw:
            path_plain = [str(x).strip() for x in hp_raw if str(x).strip()]
            leaf_plain = path_plain[-1] if path_plain else _plain_title(tl)
            section_title_line = tl if tl.startswith("#") else f"# {leaf_plain}"
            ancestor_lines = (
                [f"# {p}" for p in path_plain[:-1]] if len(path_plain) > 1 else []
            )
        else:
            pt = (str(ch.get("parent_title") or "")).strip()
            ancestor_lines: list[str] = []
            if pt and tl and pt != tl and pt.startswith("#"):
                ancestor_lines = [pt]
            section_title_line = tl if tl.startswith("#") else f"# {tl}"
            leaf_plain = _plain_title(section_title_line)
            path_plain = [_plain_title(x) for x in ancestor_lines]
            path_plain.append(leaf_plain)

        if first_piece:
            meta0 = ContentMetadata(
                content_type=ContentType.COURSE_INFO,
                course_name=course_name,
                source_file=state["original_filename"],
                document_class=DocumentClassEnum.SYLLABUS,
            )
            edu0 = EduContent(
                metadata=meta0,
                content=content,
                course_intro=course_intro,
            )
            chunks_out.append(
                {
                    "chunk_id": chunk_id,  # [新增]
                    "text": content,
                    "order": order,
                    "heading_path": path_plain,
                    "source_span": {
                        "extractor": "syllabus_extractor",
                        "order": order,
                        "part": "course_info_head",
                    },
                    "edu_content": edu0.model_dump(mode="json"),
                }
            )
            first_piece = False
        else:
            meta_i = ContentMetadata(
                content_type=ContentType.DOC_CHUNK,
                course_name=course_name,
                chapter_name=leaf_plain or None,
                source_file=state["original_filename"],
                document_class=DocumentClassEnum.SYLLABUS,
            )
            doc_chunk = DocChunkStructure.model_validate(
                {
                    "heading_path": path_plain,
                    "section_title_plain": leaf_plain,
                    "section_title_markdown": section_title_line,
                    "ancestor_heading_lines": ancestor_lines,
                }
            )
            edu_i = EduContent(
                metadata=meta_i,
                content=content,
                doc_chunk=doc_chunk,
            )
            chunks_out.append(
                {
                    "chunk_id": chunk_id,  # [新增]
                    "text": content,
                    "order": order,
                    "heading_path": path_plain,
                    "source_span": {
                        "extractor": "syllabus_extractor",
                        "order": order,
                        "part": "section",
                    },
                    "edu_content": edu_i.model_dump(mode="json"),
                }
            )
        order += 1

    if not chunks_out:
        errors = list(state.get("errors") or [])
        errors.append("Syllabus_Extractor：切分后无有效切片。")
        return {"errors": errors, "chunks": [], "lecture_metadata": prev}

    return {
        "chunks": chunks_out,
        "lecture_metadata": prev,
    }


__all__ = ["SYLLABUS_EXTRACT_PROMPT", "syllabus_extractor_node"]
