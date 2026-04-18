"""
使用 ``python-docx`` 将 Word（.docx）转为 Markdown 纯文本，供 file_router 等节点使用。

按文档流顺序遍历段落与表格：标题样式（Heading 1～6、Title、Subtitle）转为 ATX 标题，
表格转为 GitHub 风格 Markdown 表；复杂排版、公式、嵌入对象仅作尽力提取。
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

_HEADING_RE = re.compile(r"^Heading\s*(\d+)$", re.IGNORECASE)


def _iter_block_items(document: DocumentType):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _paragraph_to_md(p: Paragraph) -> str:
    text = (p.text or "").replace("\r", "").strip()
    if not text:
        return ""
    style_name = (p.style.name if p.style else "") or ""
    style_name = style_name.strip()

    m = _HEADING_RE.match(style_name)
    if m:
        level = min(max(int(m.group(1)), 1), 6)
        return f"{'#' * level} {text}"
    if style_name.lower() == "title":
        return f"# {text}"
    if style_name.lower() == "subtitle":
        return f"## {text}"

    return text


def _table_to_md(table: Table) -> str:
    rows = table.rows
    if not rows:
        return ""
    lines: list[str] = []
    for i, row in enumerate(rows):
        cells = []
        for cell in row.cells:
            t = " ".join(
                (p.text or "").replace("\r", " ").strip() for p in cell.paragraphs
            )
            t = t.replace("|", "\\|").replace("\n", " ")
            cells.append(t)
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(lines)


def docx_to_markdown(file_path: str) -> str:
    """读取 .docx，返回 Markdown 字符串。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {file_path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"需要 .docx 文件，收到: {path.suffix!r}")

    document = Document(str(path.resolve()))
    chunks: list[str] = []
    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            md = _paragraph_to_md(block)
            if md:
                chunks.append(md)
        elif isinstance(block, Table):
            md = _table_to_md(block)
            if md:
                chunks.append(md)

    return "\n\n".join(chunks).strip() + ("\n" if chunks else "")


def word_to_markdown(file_path: str) -> str:
    """与 ``docx_to_markdown`` 相同，便于语义化调用。"""
    return docx_to_markdown(file_path)


__all__ = ["docx_to_markdown", "word_to_markdown"]
