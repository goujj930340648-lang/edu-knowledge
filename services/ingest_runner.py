"""后台导入任务（与 FastAPI BackgroundTasks 配合）。"""

from __future__ import annotations

import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any

from config.settings import get_settings
from parsers.catalog_md import parse_course_catalog
from parsers.questions_md import parse_question_bank_file
from storage.mongo_db import get_mongo_db
from services.docx_image_upload import process_docx_images_for_import
from services.source_mapping_rules import upsert_rule_source_mapping_for_file
from storage.repository import (
    persist_document_import,
    update_ingest_task,
    upsert_course_catalog,
    upsert_question_data,
    upsert_source_mapping,
)


def _db():
    s = get_settings()
    uri = s.resolved_mongo_uri()
    if not uri:
        raise RuntimeError("MongoDB 未配置（MONGODB_URI 或 MONGO_URL）")
    return get_mongo_db(uri, s.mongodb_database)


def run_catalog_import(
    task_id: str,
    file_path: str,
    cleanup_path: str | None = None,
) -> None:
    db = _db()
    update_ingest_task(db, task_id, {"status": "running"}, push_log="开始导入课程目录")
    try:
        p = Path(file_path)
        text = p.read_text(encoding="utf-8")
        parsed = parse_course_catalog(text)
        stats = upsert_course_catalog(db, parsed["series"], parsed["modules"])
        logs = [
            f"系列写入: {stats['series_upserted']}, 模块写入: {stats['modules_upserted']}",
            f"解析警告数: {len(parsed.get('warnings') or [])}",
        ]
        update_ingest_task(
            db,
            task_id,
            {
                "status": "completed",
                "summary": stats,
                "parser_warnings": parsed.get("warnings") or [],
            },
            push_log="; ".join(logs),
        )
    except Exception as e:
        update_ingest_task(
            db,
            task_id,
            {"status": "failed", "error": str(e), "trace": traceback.format_exc()},
            push_log=f"失败: {e}",
        )
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except OSError:
                pass


def run_questions_import(
    task_id: str,
    file_path: str,
    cleanup_path: str | None = None,
) -> None:
    db = _db()
    update_ingest_task(db, task_id, {"status": "running"}, push_log="开始导入题库")
    try:
        p = Path(file_path)
        text = p.read_text(encoding="utf-8")
        parsed = parse_question_bank_file(text)
        stats = upsert_question_data(db, parsed["banks"], parsed["items"])
        update_ingest_task(
            db,
            task_id,
            {
                "status": "completed",
                "summary": stats,
                "parser_warnings": parsed.get("warnings") or [],
            },
            push_log=f"题库写入: {stats}; 警告数: {len(parsed.get('warnings') or [])}",
        )
    except Exception as e:
        update_ingest_task(
            db,
            task_id,
            {"status": "failed", "error": str(e), "trace": traceback.format_exc()},
            push_log=f"失败: {e}",
        )
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except OSError:
                pass


def run_documents_import(
    task_id: str,
    file_paths: list[str],
    doc_type: str,
    source_mappings: list[dict] | None,
    cleanup_dir: str | None = None,
) -> None:
    db = _db()
    update_ingest_task(db, task_id, {"status": "running"}, push_log="开始导入文档")
    sub_ok = 0
    sub_fail = 0
    try:
        from init_milvus import ensure_milvus_collections
        from main_graph import build_import_graph

        ok_m, err_m = ensure_milvus_collections()
        if not ok_m:
            raise RuntimeError(f"Milvus 不可用: {err_m}")

        graph = build_import_graph()
        sub_updates: list[dict] = []

        for fp in file_paths:
            p = Path(fp)
            st: dict = {
                "job_id": str(uuid.uuid4()),
                "original_filename": p.name,
                "source_path": str(p.resolve()),
                "api_doc_type": doc_type,
                "ingest_task_id": task_id,
            }
            try:
                if not p.is_file():
                    raise FileNotFoundError(f"文件不存在: {p}")
                out = graph.invoke(st)
                errs = out.get("errors") or []
                if errs:
                    raise RuntimeError("; ".join(str(x) for x in errs))
                doc_id = str(uuid.uuid4())
                img_keys: list[str] = []
                asset_rows: list[dict] = []
                if p.suffix.lower() == ".docx":
                    img_keys, img_warns, asset_rows = process_docx_images_for_import(
                        doc_id, str(p.resolve())
                    )
                    for w in img_warns:
                        update_ingest_task(
                            db, task_id, {}, push_log=f"图片: {w}"
                        )
                persist_document_import(
                    db,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    source_path=str(p.resolve()),
                    source_file=p.name,
                    title=p.stem,
                    ingest_task_id=task_id,
                    graph_state=out,
                    image_object_keys=img_keys,
                    asset_rows=asset_rows or None,
                )
                try:
                    upsert_rule_source_mapping_for_file(db, p.name, doc_id)
                except Exception as map_e:
                    update_ingest_task(
                        db,
                        task_id,
                        {},
                        push_log=f"来源映射规则跳过: {map_e}",
                    )
                sub_ok += 1
                sub_updates.append(
                    {
                        "file": p.name,
                        "status": "completed",
                        "doc_id": doc_id,
                        "chunks": len(out.get("chunks") or []),
                    }
                )
                update_ingest_task(
                    db, task_id, {"sub_tasks": sub_updates}, push_log=f"完成: {p.name}"
                )
            except Exception as e:
                sub_fail += 1
                sub_updates.append({"file": p.name, "status": "failed", "error": str(e)})
                update_ingest_task(
                    db,
                    task_id,
                    {"sub_tasks": sub_updates},
                    push_log=f"失败: {p.name} — {e}",
                )

        if source_mappings:
            upsert_source_mapping(db, source_mappings)

        if sub_fail == 0:
            final = "completed"
        elif sub_ok == 0:
            final = "failed"
        else:
            final = "partial_success"
        update_ingest_task(
            db,
            task_id,
            {"status": final, "summary": {"ok": sub_ok, "failed": sub_fail}},
            push_log=f"文档导入结束: {final}",
        )
    except Exception as e:
        update_ingest_task(
            db,
            task_id,
            {"status": "failed", "error": str(e), "trace": traceback.format_exc()},
            push_log=f"批处理失败: {e}",
        )
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
