#!/usr/bin/env python3
"""
跑一次导入图（用于联调）：读取单个文件，走完整 LangGraph。

支持扩展名（与 ``file_router`` 一致）::

    .md / .markdown   直接 UTF-8 读取
    .docx             用 ``python-docx`` 转 Markdown
    .pdf              需 ``PDF_IMPORT_ENABLED=true``，且当前多为占位实现

用法::

    python run_import_once.py path/to/讲义.docx
    python run_import_once.py path/to/笔记.md

Word：安装依赖中的 ``python-docx``（``pip install -r requirements.txt``）即可。

依赖环境变量（见 .env.example）：

- ACTIVE_API_KEY：分类与抽取必填
- MILVUS_URI：Milvus 地址
- 向量：本地 BGE（``BGE_M3_PATH`` 等）或 OpenAI 兼容 Embedding

若项目根目录有 .env，会自动 load_dotenv（需安装 python-dotenv）。
使用 ``override=True``，使 .env 覆盖 conda/系统里已存在的同名变量（避免 MILVUS_URI、LLM_DEFAULT_MODEL 被旧环境挡住）。
启动时会 ``ensure_milvus_collections()``：缺表则建、已有则跳过（与 ``python init_milvus.py`` 相同逻辑）。
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=True)


def main() -> int:
    _load_dotenv()
    from utils.client import maybe_strip_proxy_for_milvus

    maybe_strip_proxy_for_milvus()

    from init_milvus import ensure_milvus_collections

    ok_milvus, milvus_err = ensure_milvus_collections()
    if not ok_milvus:
        print(f"[error] Milvus 集合准备失败: {milvus_err}", file=sys.stderr)
        return 1

    supported = {".md", ".markdown", ".docx", ".pdf"}
    parser = argparse.ArgumentParser(
        description="运行导入图一次（单文件：.md / .docx / .pdf）"
    )
    parser.add_argument(
        "document",
        type=Path,
        help="本地文档路径（Markdown 或 Word 等）",
    )
    args = parser.parse_args()
    path = args.document.resolve()
    if not path.is_file():
        print(f"[error] 不是可读文件: {path}", file=sys.stderr)
        return 1
    suf = path.suffix.lower()
    if suf not in supported:
        print(
            f"[warn] 扩展名 {suf!r} 不在推荐列表 {supported} 中，仍将交给 file_router 尝试",
            file=sys.stderr,
        )
    if suf == ".docx":
        print(
            "[hint] .docx 由 python-docx 转为 Markdown（pip install -r requirements.txt）",
            file=sys.stderr,
        )

    os.environ.setdefault("MILVUS_URI", os.environ.get("MILVUS_URL", "http://127.0.0.1:19530"))

    from main_graph import build_import_graph

    initial: dict = {
        "job_id": str(uuid.uuid4()),
        "original_filename": path.name,
        "source_path": str(path),
    }

    graph = build_import_graph()
    try:
        out = graph.invoke(initial)
    except Exception as e:
        print(f"[error] 图执行失败: {e}", file=sys.stderr)
        return 1

    errs = out.get("errors") or []
    warns = out.get("warnings") or []
    if errs:
        print("[errors]")
        for x in errs:
            print(f"  - {x}")
    if warns:
        print("[warnings]")
        for x in warns:
            print(f"  - {x}")
    print("[state 摘要]")
    print(f"  document_class: {out.get('document_class')!r}")
    print(f"  chunks 数量: {len(out.get('chunks') or [])}")
    print(f"  is_success: {out.get('is_success')!r}")
    print(f"  vector_ids 数量: {len(out.get('vector_ids') or [])}")
    cat_ids = out.get("catalog_vector_ids") or []
    if cat_ids:
        print(f"  catalog_vector_ids 数量（v2 名称表）: {len(cat_ids)}")
    if errs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
