"""
Question_Extractor：按《题目资料》式锚点（【题型】等）切分块，经 LLM 抽取为 ``EduContent``，写入 ``chunks``。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from schema.edu_content import ContentMetadata, ContentType, EduContent, QuestionStructure
from schema.metadata import DocumentClass as DocumentClassEnum

from processor.import_state import ChunkDraft, ImportGraphState
from utils.client import get_llm_client

QUESTION_EXTRACT_PROMPT = """你是一个专业的教育内容解析专家。请将给定题目文本块解析为结构化 JSON。

### 约束（必须遵守）：
1. **严禁修改 LaTeX 公式**：保留所有 $ 或 $$ 及其中的内容（如 $P(A|B)$），不要转义或改写。
2. **严禁改写 Markdown 代码块**：保留 ``` 围栏及语言标签与内部代码原文。
3. **题干与选项分离**：``content`` 仅含题干（不含 A/B/C/D 选项行）；选择题选项全部放入 ``options``。
4. **答案**：``answer`` 仅填选项字母、判断结论或简短标准表述，勿夹带解析。
5. **解析**：``analysis`` 对应文档中的 [解析] 或「解析」段落全文（含代码/LaTeX 时保持原样）。

### 待处理文本块：
{block_text}

### 输出要求（仅输出一个 JSON 对象，不要其它文字）：
- question_type: 题目类型（如：单选题、多选题、判断题），与文档【题型】一致优先
- question_id: 题目编码；若文档无明确编码则填 null（由系统再生成）
- content: 题干正文
- options: 选项列表（选择题），如 ["A. xxx", "B. xxx"]；非选择题可为 null
- answer: 标准答案
- analysis: 解析全文，无则 null
- question_bank_name: 题库名称；仅从本块或块首上下文明示处提取，无法确定则 null
"""

_QUESTION_TYPE_LINE = re.compile(r"【题型】\s*([^\n\r]*)")
_SPLIT_BEFORE_TYPE = re.compile(r"(?=【题型】)")


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_json(raw: str) -> dict[str, Any]:
    return json.loads(_strip_json_fence(raw))


def _split_question_blocks(text: str) -> list[str]:
    """按下一个「【题型】」锚点切分；仅保留含【题型】的块。"""
    parts = _SPLIT_BEFORE_TYPE.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p or "【题型】" not in p:
            continue
        out.append(p)
    return out


def _fallback_type_from_block(block: str) -> str:
    m = _QUESTION_TYPE_LINE.search(block)
    if m:
        return m.group(1).strip() or "未知"
    return "未知"


def _build_edu_content(
    data: dict[str, Any],
    state: ImportGraphState,
    fallback_type: str,
) -> EduContent:
    qmeta = state.get("question_bank_metadata") or {}
    lec = state.get("lecture_metadata") or {}

    bank_name = data.get("question_bank_name")
    if bank_name is None or (isinstance(bank_name, str) and not bank_name.strip()):
        bank_name = qmeta.get("bank_name")

    course_name = lec.get("course_name")

    qid = data.get("question_id")
    if qid is not None and isinstance(qid, str) and qid.strip():
        question_id: str | None = qid.strip()
    else:
        question_id = str(uuid.uuid4())

    qtype = data.get("question_type")
    if not (isinstance(qtype, str) and qtype.strip()):
        qtype = fallback_type

    content = data.get("content")
    if not isinstance(content, str):
        content = ""
    content = content.strip()
    if not content:
        raise ValueError("LLM 未返回有效题干（content 为空）")

    options = data.get("options")
    if options is not None and not isinstance(options, list):
        options = None

    answer = data.get("answer")
    if answer is not None and not isinstance(answer, str):
        answer = str(answer)

    analysis = data.get("analysis")
    if analysis is not None and not isinstance(analysis, str):
        analysis = str(analysis) if analysis is not None else None

    meta = ContentMetadata(
        content_type=ContentType.QUESTION,
        course_name=course_name,
        bank_name=bank_name if isinstance(bank_name, str) else None,
        source_file=state["original_filename"],
        document_class=DocumentClassEnum.QUESTION_BANK,
    )

    return EduContent(
        metadata=meta,
        content=content,
        question=QuestionStructure(
            question_id=question_id,
            question_type=qtype,
            options=options,
            answer=answer,
            analysis=analysis,
        ),
    )


def question_extractor_node(state: ImportGraphState) -> dict[str, Any]:
    """
    将 ``normalized_text`` 中按【题型】切分的单元经 LLM 转为 ``EduContent``，
    以 ``ChunkDraft``（含 ``edu_content``）列表写入 ``chunks``。
    """
    patch: dict[str, Any] = {}
    errors = list(state.get("errors") or [])
    warnings = list(state.get("warnings") or [])

    dc = state.get("document_class")
    if dc is not None and dc != "question_bank":
        warnings.append(
            f"Question_Extractor：document_class 为 {dc!r}，非 question_bank，已跳过抽取。"
        )
        patch["warnings"] = warnings
        patch["chunks"] = []
        return patch

    text = (state.get("normalized_text") or "").strip()
    if not text:
        errors.append("Question_Extractor：缺少 normalized_text，无法抽取。")
        patch["errors"] = errors
        patch["chunks"] = []
        return patch

    blocks = _split_question_blocks(text)
    if not blocks:
        warnings.append(
            "Question_Extractor：未找到「【题型】」锚点，未生成题目块；请确认文档格式与《题目资料》一致。"
        )
        patch["warnings"] = warnings
        patch["chunks"] = []
        return patch

    llm = get_llm_client()
    chunks_out: list[ChunkDraft] = []

    for idx, block in enumerate(blocks):
        fb_type = _fallback_type_from_block(block)
        try:
            response = llm.chat(
                prompt=QUESTION_EXTRACT_PROMPT.format(block_text=block),
                json_mode=True,
            )
            data = _parse_llm_json(response)
            edu = _build_edu_content(data, state, fb_type)

            file_name = str(state.get("original_filename", "unknown_doc"))
            combined_str = f"{file_name}_{idx}_{edu.content}"
            chunk_id = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

            chunks_out.append(
                {
                    "chunk_id": chunk_id,
                    "text": edu.content,
                    "order": idx,
                    "source_span": {
                        "extractor": "question_extractor",
                        "block_index": idx,
                    },
                    "edu_content": edu.model_dump(mode="json"),
                }
            )
        except Exception as e:
            errors.append(f"Question_Extractor：第 {idx + 1} 块解析失败: {e!s}")

    patch["chunks"] = chunks_out
    if errors:
        patch["errors"] = errors
    if warnings:
        patch["warnings"] = warnings
    return patch


__all__ = ["QUESTION_EXTRACT_PROMPT", "question_extractor_node"]
