#!/usr/bin/env python3
"""启动 FastAPI：``python run_server.py``（先配置 .env 中 MONGODB_URI）。"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from dotenv import load_dotenv

        # 与 config/settings.py、app/main.py 一致：始终加载项目根 .env（非 scripts/.env）
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except ImportError:
        pass

    import uvicorn

    from config.settings import get_settings

    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
