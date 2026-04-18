"""
Lecture_Extractor：讲义/项目文档切分。

切分策略与 ``knowledge/processor/import_process/nodes/document_split_node.py`` 一致：
一级按 Markdown 标题 → 二级 ``RecursiveCharacterTextSplitter`` → 合并过短同源段；
表格先经 ``MarkdownTableLinearizer`` 再按长度切。
"""

from __future__ import annotations

import re
import hashlib  # [新增] 引入 hashlib
from typing import Any

from schema.edu_content import (
    ContentMetadata,
    ContentType,
    DocChunkStructure,
    EduContent,
)
from schema.metadata import DocumentClass as DocumentClassEnum

from processor.import_state import ChunkDraft, ImportGraphState
from utils.document_split import split_markdown_document

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+)")


def _plain_title(md_heading_line: str) -> str:
    m = _HEADING_RE.match(md_heading_line.strip())
    return m.group(2).strip() if m else md_heading_line.strip()


def _create_chunk_edu(
    state: ImportGraphState,
    *,
    chunk_text: str,
    heading_path_plain: list[str],
    section_title_line: str,
    ancestor_lines: list[str],
    order: int,
) -> ChunkDraft:
    lm = state.get("lecture_metadata") or {}
    dc_raw = state.get("document_class")
    dc_enum = (
        DocumentClassEnum.PROJECT if dc_raw == "project" else DocumentClassEnum.LECTURE
    )

    leaf_plain = _plain_title(section_title_line)

    meta = ContentMetadata(
        content_type=ContentType.DOC_CHUNK,
        course_name=lm.get("course_name"),
        project_name=lm.get("project_name") if dc_raw == "project" else None,
        chapter_name=leaf_plain or None,
        source_file=state["original_filename"],
        document_class=dc_enum,
    )

    doc_chunk = DocChunkStructure.model_validate(
        {
            "heading_path": heading_path_plain,
            "section_title_plain": leaf_plain,
            "section_title_markdown": section_title_line,
            "ancestor_heading_lines": ancestor_lines,
        }
    )

    edu = EduContent(
        metadata=meta,
        content=chunk_text,
        doc_chunk=doc_chunk,
    )

    # 文件名 + 切片序号 + 文本：同文档内重复段、跨文档同内容均可区分
    file_name = str(state.get("original_filename", "unknown_doc"))
    combined_str = f"{file_name}_{order}_{chunk_text}"
    chunk_id = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

    return {
        "chunk_id": chunk_id,  # [新增] 显式指定联合哈希生成的 chunk_id
        "text": chunk_text,
        "order": order,
        "heading_path": heading_path_plain,
        "source_span": {
            "extractor": "lecture_extractor",
            "order": order,
        },
        "edu_content": edu.model_dump(mode="json"),
    }


def lecture_extractor_node(state: ImportGraphState) -> dict[str, Any]:
    """
    仅当 ``document_class`` 为 ``lecture`` 或 ``project`` 时执行。
    """
    dc = state.get("document_class")
    if dc not in ("lecture", "project"):
        return {}

    text = (state.get("normalized_text") or "").strip()
    if not text:
        errors = list(state.get("errors") or [])
        errors.append("Lecture_Extractor：缺少 normalized_text。")
        return {"errors": errors, "chunks": []}

    file_title = (state.get("lecture_metadata") or {}).get("course_name") or state.get(
        "original_filename", "document"
    )
    if isinstance(file_title, str):
        file_title = file_title.strip() or "document"
    else:
        file_title = "document"

    try:
        raw_chunks = split_markdown_document(text, file_title)
    except ValueError as e:
        err = list(state.get("errors") or [])
        err.append(f"Lecture_Extractor：{e}")
        return {"errors": err, "chunks": []}

    chunks_out: list[ChunkDraft] = []
    for idx, ch in enumerate(raw_chunks):
        content = (ch.get("content") or "").strip()
        if not content:
            continue
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
            path_plain = [_plain_title(x) for x in ancestor_lines]
            path_plain.append(_plain_title(section_title_line))

        chunks_out.append(
            _create_chunk_edu(
                state,
                chunk_text=content,
                heading_path_plain=path_plain,
                section_title_line=section_title_line,
                ancestor_lines=ancestor_lines,
                order=idx,
            )
        )

    return {"chunks": chunks_out}


__all__ = ["lecture_extractor_node", "_plain_title"]
