"""
《题目资料.md》确定性解析：``##`` 题库、``###`` 题目块、字段抽取与 ``quality_flags``。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_OPEN_TYPES = frozenset(
    {
        "简答题",
        "编程题",
        "阅读理解",
        "材料分析",
        "案例分析",
        "情境分析",
        "计算题",
        "填空题",
        "综合题",
    }
)

_BANK_META = re.compile(r"^-\s*\*\*([^*]+)\*\*\s*[：:]\s*(.*)$")


def _slug(s: str) -> str:
    raw = re.sub(r"\s+", "_", s.strip().lower())
    raw = re.sub(r"[^\w\u4e00-\u9fff]+", "_", raw, flags=re.UNICODE)
    return re.sub(r"_+", "_", raw).strip("_")[:120] or "bank"


def parse_question_bank_file(text: str) -> dict[str, Any]:
    """
    返回 ``banks``（题库元列表）、``items``（题目列表）、``warnings``。
    题目字段：question_code, bank_code, question_type, stem, options,
    answer_key, reference_answer, analysis, raw_block, quality_flags。
    """
    warnings: list[str] = []
    banks: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    for part in parts:
        block = part.strip()
        if not block.startswith("## "):
            continue
        lines = block.splitlines()
        first = lines[0]
        bank_name = first[3:].strip()
        bank_meta: dict[str, str] = {}
        body_lines: list[str] = []
        i = 1
        while i < len(lines):
            ln = lines[i]
            m = _BANK_META.match(ln.strip())
            if m:
                bank_meta[m.group(1).strip()] = m.group(2).strip()
                i += 1
                continue
            body_lines = lines[i:]
            break
        bank_code = (bank_meta.get("题库编码") or bank_meta.get("编码") or "").strip()
        if not bank_code:
            bank_code = _slug(bank_name) + "_bank"
        banks.append(
            {
                "bank_code": bank_code,
                "bank_name": bank_name,
                "domain_tags": [],
                "question_count": 0,
            }
        )

        body = "\n".join(body_lines)
        q_chunks = re.split(r"(?=^### )", body, flags=re.MULTILINE)
        for qc in q_chunks:
            qc = qc.strip()
            if not qc.startswith("### "):
                if qc:
                    warnings.append(f"题库「{bank_name}」下非 ### 块已跳过: {qc[:60]}")
                continue
            ql = qc.splitlines()
            head = ql[0]
            qcode = head[4:].strip()
            qbody = "\n".join(ql[1:]).strip()
            item = _parse_question_block(qcode, bank_code, qbody, warnings)
            if item:
                items.append(item)

    for b in banks:
        b["question_count"] = sum(1 for it in items if it.get("bank_code") == b["bank_code"])

    return {"banks": banks, "items": items, "warnings": warnings}


def _parse_question_block(
    question_code: str,
    bank_code: str,
    body: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    raw_block = f"### {question_code}\n{body}"
    qtype = ""
    stem_lines: list[str] = []
    options: list[dict[str, str]] = []
    answer_raw = ""
    analysis = ""
    flags: list[str] = []

    mode = "stem"
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("【题型】"):
            qtype = s.replace("【题型】", "").strip()
            continue
        if s.startswith("【答案】") or s.startswith("- **答案"):
            mode = "answer"
            rest = s
            if "】" in rest:
                rest = rest.split("】", 1)[-1].strip()
            elif ":" in rest or "：" in rest:
                rest = re.split(r"[：:]", rest, 1)[-1].strip()
            if rest:
                answer_raw = rest
            continue
        if s.startswith("【解析】") or s.startswith("- **解析"):
            mode = "analysis"
            rest = s
            if "】" in rest:
                rest = rest.split("】", 1)[-1].strip()
            elif ":" in rest or "：" in rest:
                rest = re.split(r"[：:]", rest, 1)[-1].strip()
            analysis = rest
            continue
        if mode == "answer":
            if s:
                answer_raw = (answer_raw + "\n" + s).strip() if answer_raw else s
            continue
        if mode == "analysis":
            analysis = (analysis + "\n" + line).strip() if analysis else line
            continue
        opt_m = re.match(r"^([A-Z])[\.\)、]\s*(.*)$", s)
        if opt_m:
            label, content = opt_m.group(1), opt_m.group(2).strip()
            options.append({"label": label, "content": content})
            if not content:
                flags.append(f"empty_option_{label}")
            continue
        stem_lines.append(line)

    stem = "\n".join(stem_lines).strip()
    if not stem and not options:
        warnings.append(f"题目 {question_code} 无题干，跳过")
        return None

    answer_key, reference_answer = _split_answer_fields(answer_raw, qtype)
    _normalize_answer_separators(answer_key, flags)

    qt = (qtype or "").strip()
    choice_like = any(
        x in qt for x in ("单选题", "多选题", "判断题", "选择题", "不定项")
    )
    if choice_like and not options:
        flags.append("missing_options")
    if choice_like and not (answer_key or "").strip() and not (reference_answer or "").strip():
        flags.append("missing_answer_key")
    if len(stem) > 0 and len(stem) < 8:
        flags.append("terse_stem")
    labels = [o.get("label") for o in options if isinstance(o, dict)]
    if len(labels) != len(set(labels)):
        flags.append("duplicate_option_label")
    if labels and (answer_key or "").strip() and not any(
        open_t in qt for open_t in _OPEN_TYPES
    ) and "判断" not in qt:
        labs = {str(l).strip().upper() for l in labels if l is not None}
        for ch in re.findall(r"[A-Z]", (answer_key or "").upper()):
            if ch not in labs:
                flags.append("answer_key_not_in_options")
                break

    return {
        "question_code": question_code.strip(),
        "bank_code": bank_code,
        "question_type": qtype or "未知",
        "stem": stem,
        "options": options,
        "answer_key": answer_key,
        "reference_answer": reference_answer,
        "analysis": analysis,
        "raw_block": raw_block,
        "quality_flags": sorted(set(flags)),
    }


def _split_answer_fields(answer_raw: str, qtype: str) -> tuple[str, str]:
    if not answer_raw:
        return "", ""
    qt = (qtype or "").strip()
    for open_t in _OPEN_TYPES:
        if open_t in qt or qt.startswith(open_t[:2]):
            return "", answer_raw.strip()
    if "材料" in qt or "案例" in qt or "情境" in qt:
        return "", answer_raw.strip()
    return answer_raw.strip(), ""


def _normalize_answer_separators(answer_key: str, flags: list[str]) -> None:
    if not answer_key:
        return
    if re.search(r"[、，,]", answer_key) and re.search(r"[A-Za-z]", answer_key):
        seps = len(re.findall(r"[、，,]", answer_key))
        if seps and "," in answer_key and "、" in answer_key:
            flags.append("mixed_answer_separator")


def fingerprint_raw_block(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
