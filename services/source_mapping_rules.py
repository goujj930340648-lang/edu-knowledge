"""
PLAN：``source_mapping`` 规则初匹配（文件名关键词 vs 课程系列文本）。

命中则 ``mapping_type=rule`` upsert；无命中则跳过（留给人工接口扩展）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pymongo.database import Database

from storage.repository import upsert_source_mapping


def _tokens_from_filename(source_file: str) -> list[str]:
    stem = Path(source_file).stem
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9+]{1,}", stem)
    out: list[str] = []
    for p in parts:
        t = p.strip()
        if len(t) >= 2:
            out.append(t)
    return out


def suggest_rule_mapping_row(db: Database, source_file: str) -> dict[str, Any] | None:
    tokens = _tokens_from_filename(source_file)
    if not tokens:
        return None
    best: dict[str, Any] | None = None
    best_score = 0
    for s in db.course_series.find(
        {},
        {"series_code": 1, "title": 1, "category_path": 1, "description": 1},
    ):
        text = " ".join(
            str(x or "")
            for x in (s.get("title"), s.get("category_path"), s.get("description"))
        )
        score = sum(1 for tok in tokens if tok and tok in text)
        if score > best_score:
            best_score = score
            best = s
    if not best or best_score < 1:
        return None
    return {
        "source_file": Path(source_file).name,
        "series_code": best.get("series_code") or "",
        "module_code": "",
        "bank_code": "",
        "project_name": "",
        "mapping_type": "rule",
    }


def upsert_rule_source_mapping_for_file(
    db: Database,
    source_file: str,
    doc_id: str,
) -> bool:
    row = suggest_rule_mapping_row(db, source_file)
    if not row:
        return False
    row["doc_id"] = doc_id
    upsert_source_mapping(db, [row])
    return True
