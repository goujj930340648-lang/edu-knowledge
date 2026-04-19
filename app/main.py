from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app.routers import chat, ingest, search

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except ImportError:
        pass
    yield


app = FastAPI(
    title="教育知识库 API",
    description="对齐 PLAN：导入 / 检索 / 对话（Mongo + Milvus）",
    lifespan=lifespan,
    docs_url=None,  # 禁用默认 Swagger UI
    redoc_url=None,  # 禁用默认 ReDoc
)


@app.get("/")
def root_page() -> RedirectResponse:
    return RedirectResponse(url="/front/index.html", status_code=302)


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# 自定义 API 文档页面
@app.get("/docs")
def custom_docs():
    return FileResponse(str(_WEB_DIR / "docs.html"))


# 自定义健康检查可视化页面
@app.get("/health")
def custom_health():
    return FileResponse(str(_WEB_DIR / "health.html"))


# 健康检查 JSON API（供前端调用）
@app.get("/api/health")
def health_api():
    from app.routers.health import health

    return health()


app.include_router(ingest.router, prefix="/ingest")
app.include_router(search.router, prefix="/search")
app.include_router(chat.router, prefix="/chat")

if _WEB_DIR.is_dir():
    app.mount(
        "/front",
        StaticFiles(directory=str(_WEB_DIR), html=False),
        name="front",
    )
