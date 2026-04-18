"""通用工具（避免在 ``import utils.*`` 时强依赖 docx 等可选包）。"""

from __future__ import annotations

from typing import Any

__all__ = ["docx_to_markdown", "word_to_markdown"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from utils.converter import docx_to_markdown, word_to_markdown

        return {"docx_to_markdown": docx_to_markdown, "word_to_markdown": word_to_markdown}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
