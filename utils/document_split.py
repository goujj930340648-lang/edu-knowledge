"""
Markdown 文档切分：一级按标题，二级 RecursiveCharacterTextSplitter，再合并过短同源段。
对齐 ``knowledge/processor/import_process/nodes/document_split_node.py`` 行为。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.markdown_table_linearizer import MarkdownTableLinearizer

logger = logging.getLogger(__name__)

SECTION_TITLE_MAX_LEN = 80

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+)")


def _plain_heading(md_line: str) -> str:
    """Markdown 标题行 → 纯文本（用于 ``heading_path`` / Mongo ``section_path``）。"""
    m = _HEADING_RE.match((md_line or "").strip())
    return m.group(2).strip() if m else (md_line or "").strip()


def _split_limits_from_env() -> tuple[int, int]:
    """默认与 knowledge ImportConfig 量级一致；可通过环境变量覆盖。"""
    try:
        mx = int(os.environ.get("DOC_SPLIT_MAX_LENGTH", "4000"))
        mn = int(os.environ.get("DOC_SPLIT_MIN_LENGTH", "400"))
    except ValueError:
        mx, mn = 4000, 400
    if mx <= 0 or mn <= 0 or mx <= mn:
        mx, mn = 4000, 400
    return mx, mn


def split_by_headings(md_content: str, file_title: str) -> list[dict[str, Any]]:
    """
    按 ``#``～``######`` 切分（代码块内 ``#`` 不视为标题），结构与 knowledge ``_splist_by_headings`` 一致。
    每节含 body / title / parent_title / file_title。
    """
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    in_fence = False
    body_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_title = ""
    hierarchy = [""] * 7
    current_level = 0
    heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

    def _flush() -> None:
        nonlocal body_lines
        body = "\n".join(body_lines)
        if current_title or body:
            parent_title = ""
            for i in range(current_level - 1, 0, -1):
                if hierarchy[i]:
                    parent_title = hierarchy[i]
                    break
            if not parent_title:
                parent_title = current_title if current_title else file_title
            heading_path_plain: list[str] = []
            upper = min(current_level, 6) if current_level > 0 else 0
            for lvl in range(1, upper + 1):
                ln = (hierarchy[lvl] or "").strip()
                if ln:
                    heading_path_plain.append(_plain_heading(ln))
            sections.append(
                {
                    "body": body,
                    "title": current_title if current_title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "heading_path": heading_path_plain,
                }
            )

    for md_line in md_content.split("\n"):
        stripped = md_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        match = heading_re.match(md_line) if not in_fence else None
        if match:
            _flush()
            current_title = md_line
            level = len(match.group(1))
            current_level = level
            hierarchy[level] = current_title
            for i in range(level + 1, 7):
                hierarchy[i] = ""
            body_lines = []
        else:
            body_lines.append(md_line)
    _flush()
    return sections


def split_long_section(
    section: dict[str, Any], max_content_length: int
) -> list[dict[str, Any]]:
    """过长节用递归字符切分；表格先线性化。"""
    body = section.get("body") or ""
    title = section.get("title") or ""
    parent_title = section.get("parent_title") or ""
    file_title = section.get("file_title") or ""

    if len(title) > SECTION_TITLE_MAX_LEN:
        title = title[:SECTION_TITLE_MAX_LEN]

    if "<table>" in body.lower():
        body = MarkdownTableLinearizer.process(body)

    title_prefix = f"{title}\n\n"
    total_length = len(title_prefix) + len(body)
    if total_length <= max_content_length:
        return [section]

    body_length = max_content_length - len(title_prefix)
    if body_length <= 0:
        return [section]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=body_length,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            "。",
            "？",
            "！",
            "；",
            ".",
            "?",
            "!",
            ";",
            " ",
            "",
        ],
        keep_separator=True,
    )
    chunks = text_splitter.split_text(body)
    if len(chunks) == 1:
        return [section]

    sub_sections: list[dict[str, Any]] = []
    hp = list(section.get("heading_path") or []) if isinstance(section.get("heading_path"), list) else []
    for index, chunk in enumerate(chunks):
        new_section = section.copy()
        new_section.update(
            {
                "body": f"{title}\n\n{chunk}",
                "title": f"{title}_{index + 1}",
                "parent_title": parent_title,
                "file_title": file_title,
                "heading_path": hp,
            }
        )
        sub_sections.append(new_section)
    return sub_sections


def merge_short_sections(
    current_sections: list[dict[str, Any]], min_content_length: int
) -> list[dict[str, Any]]:
    """同源（parent_title 相同）且过短则贪心合并。"""
    if not current_sections:
        return []
    before_count = len(current_sections)
    current_section = current_sections[0]
    final_sections: list[dict[str, Any]] = []
    for next_section in current_sections[1:]:
        same_parent = current_section["parent_title"] == next_section["parent_title"]
        body = current_section.get("body") or ""
        if same_parent and len(body) < min_content_length:
            current_section["body"] = (
                body.rstrip() + "\n\n" + (next_section.get("body") or "").rstrip()
            )
            current_section["title"] = current_section["parent_title"]
        else:
            final_sections.append(current_section)
            current_section = next_section
    final_sections.append(current_section)
    logger.info("合并短章节: %s -> %s", before_count, len(final_sections))
    return final_sections


def split_and_merge(
    sections: list[dict[str, Any]],
    max_content_length: int,
    min_content_length: int,
) -> list[dict[str, Any]]:
    current: list[dict[str, Any]] = []
    for section in sections:
        current.extend(split_long_section(section, max_content_length))
    return merge_short_sections(current, min_content_length)


def assemble_chunk_dicts(final_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """与 knowledge ``_assemble_chunks`` 一致：每条含 content / title / parent_title / file_title。"""
    out: list[dict[str, Any]] = []
    for section in final_sections:
        body = section.get("body") or ""
        title = section.get("title") or ""
        parent_title = section.get("parent_title") or ""
        file_title = section.get("file_title") or ""
        content = f"{title}\n\n{body}"
        hp = section.get("heading_path")
        heading_path = list(hp) if isinstance(hp, list) else []
        out.append(
            {
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "heading_path": heading_path,
            }
        )
    return out


def split_markdown_document(
    md_content: str,
    file_title: str,
    *,
    max_content_length: int | None = None,
    min_content_length: int | None = None,
) -> list[dict[str, Any]]:
    """
    完整切分流水线，返回 chunk 字典列表（每项含 ``content`` 全文与其它元数据）。

    未传 ``max_content_length`` / ``min_content_length`` 时从环境变量
    ``DOC_SPLIT_MAX_LENGTH`` / ``DOC_SPLIT_MIN_LENGTH`` 读取（默认 4000 / 400）。
    """
    md_content = (md_content or "").replace("\r\n", "\n").replace("\r", "\n")
    if max_content_length is None or min_content_length is None:
        emx, emn = _split_limits_from_env()
        max_content_length = max_content_length if max_content_length is not None else emx
        min_content_length = min_content_length if min_content_length is not None else emn
    if (
        max_content_length <= 0
        or min_content_length <= 0
        or max_content_length <= min_content_length
    ):
        raise ValueError(
            "切片长度参数无效：需 max_content_length > min_content_length > 0"
        )

    sections = split_by_headings(md_content, file_title)
    logger.info("一级标题切分: %s 节", len(sections))
    merged = split_and_merge(sections, max_content_length, min_content_length)
    logger.info("二次切分合并后: %s 节", len(merged))
    return assemble_chunk_dicts(merged)


__all__ = [
    "SECTION_TITLE_MAX_LEN",
    "split_by_headings",
    "split_long_section",
    "merge_short_sections",
    "split_and_merge",
    "assemble_chunk_dicts",
    "split_markdown_document",
]
