"""
从 ``.docx`` 抽取内嵌图片字节（OOXML 关系），供 MinIO 上传。

不依赖段落位置与 chunk 对齐；首版将 ``object_key`` 列表挂在 ``knowledge_document`` 上。
"""

from __future__ import annotations

IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def extract_docx_images(file_path: str) -> list[tuple[str, bytes, str]]:
    """
    返回 ``(文件名, 二进制, content_type)``。
    文件名形如 ``img_0001.png``，按文档内图片关系遍历顺序编号。
    """
    from docx import Document

    doc = Document(file_path)
    out: list[tuple[str, bytes, str]] = []
    idx = 0
    for rel in doc.part.rels.values():
        if rel.reltype != IMAGE_REL:
            continue
        try:
            part = rel.target_part
            blob = part.blob
        except Exception:
            continue
        if not blob:
            continue
        pn = ""
        try:
            pn = str(part.partname)
        except Exception:
            pass
        ext = "png"
        if "." in pn:
            ext = pn.rsplit(".", 1)[-1].lower()
        if ext not in ("png", "jpeg", "jpg", "gif", "bmp", "tif", "tiff", "emf", "wmf", "svg"):
            ext = "png"
        name = f"img_{idx:04d}.{ext}"
        idx += 1
        out.append((name, blob, _content_type_for_ext(ext)))
    return out


def _content_type_for_ext(ext: str) -> str:
    m = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "svg": "image/svg+xml",
        "emf": "image/emf",
        "wmf": "image/wmf",
    }
    return m.get(ext.lower(), "application/octet-stream")


__all__ = ["extract_docx_images", "IMAGE_REL"]
