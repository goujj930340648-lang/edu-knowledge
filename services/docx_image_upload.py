"""文档导入时：抽取 .docx 图片 → MinIO → ``asset_object`` 行。"""

from __future__ import annotations

from typing import Any

from config.settings import get_settings
from processor.docx_media import extract_docx_images
from storage.minio_client import public_url_for_key, put_document_image


def process_docx_images_for_import(
    doc_id: str,
    docx_path: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    返回 ``(object_keys, warnings, asset_rows)``。
    MinIO 未配置时返回空列表与一条说明性 warning。
    """
    warnings: list[str] = []
    keys: list[str] = []
    rows: list[dict[str, Any]] = []
    if not docx_path.lower().endswith(".docx"):
        return keys, warnings, rows
    s = get_settings()
    if not s.minio_configured():
        warnings.append("MinIO 未配置，跳过图片上传")
        return keys, warnings, rows
    bucket = (s.minio_bucket or "education-knowledge").strip()
    try:
        images = extract_docx_images(docx_path)
    except Exception as e:
        warnings.append(f"抽取图片失败: {e}")
        return keys, warnings, rows
    for name, blob, ct in images:
        ok, key, err = put_document_image(doc_id, name, blob, ct)
        if ok and key:
            keys.append(key)
            rows.append(
                {
                    "object_key": key,
                    "bucket": bucket,
                    "content_type": ct,
                    "width": 0,
                    "height": 0,
                    "source_doc_id": doc_id,
                    "source_chunk_id": "",
                    "public_url": public_url_for_key(key) or "",
                }
            )
        elif err:
            warnings.append(f"{name}: {err}")
    return keys, warnings, rows
