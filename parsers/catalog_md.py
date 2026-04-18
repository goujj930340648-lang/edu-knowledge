"""
《课程介绍.md》确定性解析：``##`` 系列、``- **字段**:`` 键值、``### 课程`` 下模块列表。

与 PLAN 一致：不调用 LLM；无法识别的行跳过并记入 ``warnings``。
"""

from __future__ import annotations

import re
from typing import Any

_FIELD_LINE = re.compile(r"^-\s*\*\*([^*]+)\*\*\s*[：:]\s*(.*)$")
_MODULE_HEAD = re.compile(r"^-\s*\*\*([^*]+)\*\*\s*$")
_INDENT_KV = re.compile(r"^\s+-\s*\*\*([^*]+)\*\*\s*[：:]\s*(.*)$")


def parse_course_catalog(text: str) -> dict[str, Any]:
    """
    返回 ``{"series": [...], "modules": [...], "warnings": [...]}``。

    每个 series 含: series_code, title, description, category_path,
    audience, goal_tags, grade_tags（数组字段为 list[str]）。
    每个 module 含: module_code, series_code, module_title, lesson_count,
    study_hours, module_desc, sort_order。
    """
    lines = text.splitlines()
    series: list[dict[str, Any]] = []
    modules: list[dict[str, Any]] = []
    warnings: list[str] = []

    current_series: dict[str, Any] | None = None
    in_course_section = False
    pending_module: dict[str, Any] | None = None
    series_idx = -1

    def flush_module() -> None:
        nonlocal pending_module, current_series
        if pending_module is None or current_series is None:
            return
        sc = current_series.get("series_code")
        if not sc:
            warnings.append(f"模块缺少所属 series_code，已跳过: {pending_module}")
            pending_module = None
            return
        pending_module["series_code"] = sc
        if not pending_module.get("module_code"):
            warnings.append(f"模块无编码，已跳过: {pending_module.get('module_title')}")
            pending_module = None
            return
        modules.append(pending_module)
        pending_module = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        if line.startswith("## ") and not line.startswith("### "):
            flush_module()
            in_course_section = False
            title = line[3:].strip()
            current_series = {
                "title": title,
                "series_code": "",
                "description": "",
                "category_path": "",
                "audience": [],
                "goal_tags": [],
                "grade_tags": [],
            }
            series.append(current_series)
            series_idx += 1
            continue

        if current_series is None:
            if line.strip():
                warnings.append(f"在首个 ## 系列前忽略行: {line[:80]}")
            continue

        if line.startswith("### 课程") or line.startswith("### 课程 "):
            flush_module()
            in_course_section = True
            continue

        m = _FIELD_LINE.match(line)
        if m and not in_course_section:
            key, val = m.group(1).strip(), m.group(2).strip()
            nk = _normalize_series_key(key)
            if nk == "series_code":
                current_series["series_code"] = val
            elif nk == "description":
                current_series["description"] = val
            elif nk == "category_path" or nk == "课程分类":
                current_series["category_path"] = val
            elif nk in ("audience", "适合人群"):
                current_series["audience"] = _split_list(val)
            elif nk in ("goal_tags", "学习目标"):
                current_series["goal_tags"] = _split_list(val)
            elif nk in ("grade_tags", "适合年级"):
                current_series["grade_tags"] = _split_list(val)
            else:
                warnings.append(f"未知系列字段「{key}」")
            continue

        if in_course_section:
            mh = _MODULE_HEAD.match(line)
            if mh:
                flush_module()
                pending_module = {
                    "module_title": mh.group(1).strip(),
                    "module_code": "",
                    "lesson_count": 0,
                    "study_hours": 0.0,
                    "module_desc": "",
                    "sort_order": len(
                        [x for x in modules if x.get("series_code") == current_series.get("series_code")]
                    ),
                }
                continue
            im = _INDENT_KV.match(line)
            if im and pending_module is not None:
                k, v = im.group(1).strip(), im.group(2).strip()
                nk = _normalize_module_key(k)
                if nk in ("module_code", "编码"):
                    pending_module["module_code"] = v
                elif nk in ("lesson_count", "课时"):
                    pending_module["lesson_count"] = _parse_int(v, warnings, k)
                elif nk in ("study_hours", "学时"):
                    pending_module["study_hours"] = _parse_float(v, warnings, k)
                elif nk in ("module_desc", "描述"):
                    pending_module["module_desc"] = v
                else:
                    warnings.append(f"未知模块子字段「{k}」")
                continue
            if line.strip().startswith("-") and pending_module is None:
                warnings.append(f"模块小节内未识别的行: {line[:120]}")
            continue

        if m and in_course_section:
            warnings.append(f"课程小节内出现顶层字段行，已忽略: {line[:80]}")

    flush_module()

    for s in series:
        if not s.get("series_code"):
            warnings.append(f"系列缺少 series_code，标题={s.get('title')!r}")

    return {"series": series, "modules": modules, "warnings": warnings}


def _normalize_series_key(k: str) -> str:
    k = k.strip()
    mapping = {
        "系列编码": "series_code",
        "描述": "description",
        "课程描述": "description",
        "课程分类": "category_path",
        "适合人群": "audience",
        "学习目标": "goal_tags",
        "适合年级": "grade_tags",
    }
    return mapping.get(k, k)


def _normalize_module_key(k: str) -> str:
    k = k.strip()
    mapping = {
        "模块名": "module_title",
        "编码": "module_code",
        "模块编码": "module_code",
        "课时": "lesson_count",
        "学时": "study_hours",
        "描述": "module_desc",
        "模块描述": "module_desc",
    }
    return mapping.get(k, k)


def _split_list(val: str) -> list[str]:
    val = val.strip()
    if not val:
        return []
    for sep in ("、", "，", ",", ";", "；"):
        if sep in val:
            return [x.strip() for x in val.split(sep) if x.strip()]
    return [val]


def _parse_int(v: str, warnings: list[str], key: str) -> int:
    try:
        return int(float(v.strip()))
    except ValueError:
        warnings.append(f'「{key}」非整数: {v!r}，记为 0')
        return 0


def _parse_float(v: str, warnings: list[str], key: str) -> float:
    try:
        return float(v.strip())
    except ValueError:
        warnings.append(f'「{key}」非数字: {v!r}，记为 0')
        return 0.0
