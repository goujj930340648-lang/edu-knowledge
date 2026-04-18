"""
File_Router：按扩展名分支，产出规范化纯文本供后续分类与抽取。

- Markdown：UTF-8 读入原文。
- Word：用 ``python-docx`` 转为 Markdown（标题 / 表格等尽力结构化）。
- PDF：默认拒绝；仅当 ``PDF_IMPORT_ENABLED`` 为真时返回明确「未实现」提示（预留）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from processor.import_state import ImportGraphState
from schema.metadata import FileKind, FileRouterOutput, infer_file_kind_from_name
from processor.adapters.pdf_adapter import convert_pdf_to_markdown
from utils.converter import word_to_markdown


def _env_pdf_enabled() -> bool:
    raw = os.environ.get("PDF_IMPORT_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_path(state: ImportGraphState) -> tuple[Optional[Path], Optional[str]]:
    """返回可读本地路径；缺失时给出错误说明。"""
    raw = state.get("source_path")
    if not raw or not str(raw).strip():
        if state.get("source_bytes_key"):
            return None, "已提供 source_bytes_key，但字节拉取尚未接入，请暂用 source_path。"
        return None, "缺少 source_path，无法读取文件内容。"
    path = Path(raw)
    if not path.is_file():
        return None, f"路径不是可读文件: {path}"
    return path, None


def file_router_node(state: ImportGraphState) -> dict[str, Any]:
    """
    LangGraph 节点：写入 ``file_kind``、``normalized_text``、``conversion_engine`` 或 ``router_error``。

    失败时向 ``errors`` 追加一条说明（若存在该列表）。
    """
    filename = state.get("original_filename") or ""
    patch: dict[str, Any] = {}
    errors = list(state.get("errors") or [])

    kind = infer_file_kind_from_name(filename)
    if kind is None:
        msg = f"不支持的文件扩展名: {filename!r}"
        patch["router_error"] = msg
        errors.append(msg)
        patch["errors"] = errors
        return patch

    path, path_err = _resolve_path(state)
    if path_err:
        out = FileRouterOutput(file_kind=kind, router_error=path_err)
        patch.update(out.to_state_patch())
        errors.append(path_err)
        patch["errors"] = errors
        return patch

    assert path is not None

    try:
        if kind == FileKind.MD:
            text = _read_markdown(path)
            result = FileRouterOutput(
                file_kind=kind,
                normalized_text=text,
                conversion_engine=None,
            )
        elif kind == FileKind.DOCX:
            text = word_to_markdown(str(path))
            result = FileRouterOutput(
                file_kind=kind,
                normalized_text=text,
                conversion_engine="python-docx",
            )
        else:
            # PDF（MinerU / magic-pdf，见 ``processor.pdf_magic``）
            if not _env_pdf_enabled():
                msg = (
                    "PDF 导入已关闭（默认）。请在 .env 设置 PDF_IMPORT_ENABLED=true，"
                    "并配置 MAGIC_PDF_CONFIG / PDF_MAGIC_CMD。"
                )
                result = FileRouterOutput(file_kind=kind, router_error=msg)
                errors.append(msg)
            else:
                md_text, err = convert_pdf_to_markdown(path)
                if md_text:
                    result = FileRouterOutput(
                        file_kind=kind,
                        normalized_text=md_text,
                        conversion_engine="magic-pdf",
                    )
                else:
                    msg = err or "PDF 转 Markdown 失败"
                    result = FileRouterOutput(file_kind=kind, router_error=msg)
                    errors.append(msg)
    except OSError as e:
        msg = f"读取文件失败: {e}"
        result = FileRouterOutput(file_kind=kind, router_error=msg)
        errors.append(msg)
    except Exception as e:  # pragma: no cover - 转换异常
        msg = f"转换失败: {e}"
        result = FileRouterOutput(file_kind=kind, router_error=msg)
        errors.append(msg)

    patch.update(result.to_state_patch())
    if errors:
        patch["errors"] = errors
    return patch


__all__ = ["file_router_node"]
