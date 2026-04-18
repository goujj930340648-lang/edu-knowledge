"""
PDF → Markdown：优先调用本机 MinerU / magic-pdf（与 ``magic-pdf.json`` 等配置配合）。

环境变量：

- ``PDF_IMPORT_ENABLED``：总开关（与 ``file_router`` 一致）。
- ``MAGIC_PDF_CONFIG``：配置文件路径，传给 ``-c``（若命令支持）。
- ``PDF_MAGIC_CMD``：可执行文件，默认 ``magic-pdf``；若在 PATH 外请写绝对路径。
- ``PDF_MAGIC_MODE``：默认 ``auto``（MinerU 常见参数 ``-m``）。
- ``PDF_MAGIC_TIMEOUT``：秒，默认 ``600``。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _timeout_sec() -> int:
    try:
        return max(60, int(os.environ.get("PDF_MAGIC_TIMEOUT", "600")))
    except ValueError:
        return 600


def convert_pdf_to_markdown(pdf_path: Path) -> tuple[str | None, str]:
    """
    调用外部 CLI 将 PDF 转为 Markdown，返回 ``(text, error)``。
    """
    if not pdf_path.is_file():
        return None, f"文件不存在: {pdf_path}"

    bin_name = (os.environ.get("PDF_MAGIC_CMD") or "magic-pdf").strip()
    mode = (os.environ.get("PDF_MAGIC_MODE") or "auto").strip() or "auto"
    cfg = (os.environ.get("MAGIC_PDF_CONFIG") or "").strip()

    out_root = Path(tempfile.mkdtemp(prefix="edu_magic_pdf_"))
    try:
        cmd: list[str] = [bin_name, "-p", str(pdf_path.resolve()), "-o", str(out_root), "-m", mode]
        if cfg:
            cfg_path = Path(cfg)
            if cfg_path.is_file():
                cmd.extend(["-c", str(cfg_path.resolve())])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_timeout_sec(),
            shell=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
            return None, f"magic-pdf 失败: {err[:800]}"

        md_files = sorted(out_root.rglob("*.md"), key=lambda p: (-p.stat().st_size, str(p)))
        if not md_files:
            return None, f"未在输出目录找到 .md: {out_root}"

        parts: list[str] = []
        for mf in md_files[:12]:
            try:
                parts.append(mf.read_text(encoding="utf-8", errors="replace"))
            except OSError as e:
                return None, f"读取生成 Markdown 失败: {e}"
        text = "\n\n---\n\n".join(parts).strip()
        if not text:
            return None, "生成的 Markdown 为空"
        return text, ""
    except FileNotFoundError:
        return (
            None,
            f"未找到可执行文件 {bin_name!r}，请安装 MinerU/magic-pdf 或设置 PDF_MAGIC_CMD 为绝对路径。",
        )
    except subprocess.TimeoutExpired:
        return None, f"PDF 转换超时（>{_timeout_sec()}s）"
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


__all__ = ["convert_pdf_to_markdown"]
