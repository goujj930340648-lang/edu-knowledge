from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pymongo.database import Database

from app.deps import require_mongo, verify_api_key
from app.schemas import ChatQueryRequest
from app.streaming import iter_sse, register_stream
from services.chat_engine import run_chat_sync
from services.chat_stream_runner import run_chat_sse_worker
from storage.repository import list_chat_history

router = APIRouter(tags=["chat"], dependencies=[Depends(verify_api_key)])


@router.post("/query")
def chat_query(
    body: ChatQueryRequest,
    _: Database = Depends(require_mongo),
) -> dict:
    sid = (body.session_id or "").strip() or str(uuid.uuid4())
    tid = str(uuid.uuid4())

    if body.stream:

        def _worker() -> None:
            run_chat_sse_worker(
                tid,
                body.query,
                session_id=sid,
                messages=body.messages,
            )

        register_stream(tid)
        threading.Thread(target=_worker, daemon=True).start()
        return {"task_id": tid, "session_id": sid, "stream": True}

    return run_chat_sync(
        body.query,
        session_id=sid,
        messages=body.messages,
        task_id=tid,
    )


@router.get("/stream/{task_id}")
def chat_stream(task_id: str) -> StreamingResponse:
    return StreamingResponse(
        iter_sse(task_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/history/{session_id}")
def chat_history(
    session_id: str,
    limit: int = 100,
    db: Database = Depends(require_mongo),
) -> dict:
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 无效")
    rows = list_chat_history(db, session_id.strip(), limit=limit)
    return {"session_id": session_id, "messages": rows}
