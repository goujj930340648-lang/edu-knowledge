from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pymongo.database import Database

from app.deps import require_mongo, verify_api_key
from app.schemas import (
    CatalogIngestRequest,
    DocumentsIngestRequest,
    QuestionsIngestRequest,
    SourceMappingBatchRequest,
)
from services.ingest_runner import run_catalog_import, run_documents_import, run_questions_import
from storage.minio_client import presigned_get_url
from storage.repository import (
    create_ingest_task,
    get_ingest_task,
    list_source_mappings,
    upsert_source_mapping,
)

router = APIRouter(tags=["ingest"], dependencies=[Depends(verify_api_key)])

_ALLOWED_DOC = frozenset({".md", ".markdown", ".docx", ".pdf"})


def _safe_upload_name(name: str | None) -> str:
    base = Path(name or "upload").name.replace("..", "_")
    return base[:220] if base else "upload.bin"


@router.post("/catalog", status_code=202)
def ingest_catalog(
    body: CatalogIngestRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
) -> dict:
    p = Path(body.file_path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"文件不可读: {p}")
    task_id = str(uuid.uuid4())
    create_ingest_task(db, task_id, "catalog")
    background_tasks.add_task(run_catalog_import, task_id, str(p.resolve()))
    return {"task_id": task_id}


@router.post("/catalog/upload", status_code=202)
async def ingest_catalog_upload(
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
    file: UploadFile = File(..., description="课程介绍 .md / .markdown"),
) -> dict:
    suf = Path(file.filename or "").suffix.lower()
    if suf not in (".md", ".markdown"):
        raise HTTPException(status_code=400, detail="请上传 .md 或 .markdown")
    tmp = Path(tempfile.gettempdir()) / f"edu_cat_{uuid.uuid4().hex}{suf or '.md'}"
    tmp.write_bytes(await file.read())
    task_id = str(uuid.uuid4())
    create_ingest_task(db, task_id, "catalog")
    background_tasks.add_task(run_catalog_import, task_id, str(tmp), str(tmp))
    return {"task_id": task_id, "saved_temp": str(tmp)}


@router.post("/questions", status_code=202)
def ingest_questions(
    body: QuestionsIngestRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
) -> dict:
    p = Path(body.file_path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"文件不可读: {p}")
    task_id = str(uuid.uuid4())
    create_ingest_task(db, task_id, "questions")
    background_tasks.add_task(run_questions_import, task_id, str(p.resolve()))
    return {"task_id": task_id}


@router.post("/questions/upload", status_code=202)
async def ingest_questions_upload(
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
    file: UploadFile = File(..., description="题目资料 .md / .markdown"),
) -> dict:
    suf = Path(file.filename or "").suffix.lower()
    if suf not in (".md", ".markdown"):
        raise HTTPException(status_code=400, detail="请上传 .md 或 .markdown")
    tmp = Path(tempfile.gettempdir()) / f"edu_q_{uuid.uuid4().hex}{suf or '.md'}"
    tmp.write_bytes(await file.read())
    task_id = str(uuid.uuid4())
    create_ingest_task(db, task_id, "questions")
    background_tasks.add_task(run_questions_import, task_id, str(tmp), str(tmp))
    return {"task_id": task_id, "saved_temp": str(tmp)}


@router.post("/documents", status_code=202)
def ingest_documents(
    body: DocumentsIngestRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
) -> dict:
    resolved: list[str] = []
    sub_tasks: list[dict] = []
    for fp in body.file_paths:
        p = Path(fp).expanduser()
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"文件不可读: {p}")
        resolved.append(str(p.resolve()))
        sub_tasks.append({"file": p.name, "status": "pending"})
    task_id = str(uuid.uuid4())
    create_ingest_task(db, task_id, "documents", sub_tasks=sub_tasks)
    background_tasks.add_task(
        run_documents_import,
        task_id,
        resolved,
        body.doc_type,
        body.source_mappings,
    )
    return {"task_id": task_id}


@router.post("/documents/upload", status_code=202)
async def ingest_documents_upload(
    background_tasks: BackgroundTasks,
    db: Database = Depends(require_mongo),
    doc_type: Literal["course_doc", "project_doc"] = Form("course_doc"),
    files: list[UploadFile] = File(..., description="多个 .md / .docx"),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")
    tdir = Path(tempfile.mkdtemp(prefix=f"edu_docu_{uuid.uuid4().hex}_"))
    paths: list[str] = []
    try:
        for uf in files:
            name = _safe_upload_name(uf.filename)
            suf = Path(name).suffix.lower()
            if suf not in _ALLOWED_DOC:
                raise HTTPException(status_code=400, detail=f"不支持的类型: {name}（仅 md/docx）")
            dest = tdir / f"{uuid.uuid4().hex[:16]}_{name}"
            dest.write_bytes(await uf.read())
            paths.append(str(dest.resolve()))
    except HTTPException:
        shutil.rmtree(tdir, ignore_errors=True)
        raise
    task_id = str(uuid.uuid4())
    sub_tasks = [{"file": Path(p).name, "status": "pending"} for p in paths]
    create_ingest_task(db, task_id, "documents", sub_tasks=sub_tasks)
    background_tasks.add_task(
        run_documents_import,
        task_id,
        paths,
        doc_type,
        None,
        str(tdir),
    )
    return {"task_id": task_id, "files": [Path(p).name for p in paths]}


@router.get("/assets/presign")
def presign_minio_object(
    object_key: str = Query(
        ...,
        description="对象键，如 documents/{doc_id}/images/xxx.png",
    ),
    expires_seconds: int = Query(3600, ge=60, le=604800),
) -> dict:
    """返回短时有效的 GET 预签名 URL（需 MinIO 已配置）。"""
    ok, url, err = presigned_get_url(object_key, expires_seconds=expires_seconds)
    if not ok:
        raise HTTPException(status_code=400, detail=err or "presign 失败")
    return {"url": url, "expires_seconds": expires_seconds}


@router.get("/source-mapping")
def list_source_mapping_entries(
    doc_id: str | None = None,
    source_file: str | None = None,
    limit: int = 200,
    db: Database = Depends(require_mongo),
) -> dict:
    items = list_source_mappings(
        db, doc_id=doc_id, source_file=source_file, limit=limit
    )
    return {"items": items, "count": len(items)}


@router.post("/source-mapping")
def upsert_source_mapping_batch(
    body: SourceMappingBatchRequest,
    db: Database = Depends(require_mongo),
) -> dict:
    rows = []
    for m in body.mappings:
        row = dict(m)
        row.setdefault("mapping_type", "manual")
        rows.append(row)
    n = upsert_source_mapping(db, rows)
    return {"upserted": n}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, db: Database = Depends(require_mongo)) -> dict:
    doc = get_ingest_task(db, task_id)
    if not doc:
        raise HTTPException(status_code=404, detail="task_id 不存在")
    return doc
