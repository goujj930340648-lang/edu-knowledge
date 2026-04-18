"""
Content_Classifier：对规范化 Markdown 头部采样，调用 LLM 判定文档类别。
"""

from __future__ import annotations

import json
from typing import Any

from schema.metadata import DocumentClass as DocumentClassEnum

from processor.import_state import DocumentClass, ImportGraphState
from utils.client import get_llm_client

# 与 ``schema.metadata.DocumentClass`` / ``ImportGraphState.document_class`` 取值一致（含 project）。
CLASSIFIER_PROMPT = """你是一个教育知识库管理助手。请根据提供的文档片段，判断该文档所属的类别。

### 类别定义：
1. **syllabus（课程大纲/介绍）**：以课程目标、先修要求、章节大纲、适合人群等描述性内容为主；常见「课程介绍」类排版。
2. **lecture（教学讲义）**：以章节知识点讲解、概念与例题说明为主，偏正式授课材料（与纯大纲介绍相比，正文讲解更多）。
3. **question_bank（题库）**：包含大量题目、题型（单选/多选等）、选项（A/B/C/D）、答案或解析。
4. **project（项目文档）**：包含实战步骤、环境配置、命令行、代码片段或项目背景，偏实验/项目指导书。

### 样例文档特征：
- 课程介绍/大纲通常有：# 课程名称、适合人群、先修要求、章节列表。
- 题库通常有：【题型】、[答案]、[解析] 或选项标记。
- 项目文档常见：环境依赖、git clone、运行步骤、代码块与目录结构说明。

### 待处理文档片段：
{text_sample}

### 输出要求：
请仅返回 JSON 格式，包含以下字段：
- document_class: 必须是以下之一（小写英文）："syllabus" | "lecture" | "question_bank" | "project"
- confidence: 0.0 到 1.0 之间的置信度
- reasoning: 简短的判断理由
"""

_SAMPLE_LEN = 2000

# LLM 偶发使用大写或旧版标签时的归一化
_ALIAS_TO_CANONICAL: dict[str, DocumentClass] = {
    "syllabus": "syllabus",
    "lecture": "lecture",
    "question_bank": "question_bank",
    "project": "project",
    "LECTURE": "lecture",
    "QUESTION_BANK": "question_bank",
    "PROJECT": "project",
    "SYLLABUS": "syllabus",
    "QUESTIONBANK": "question_bank",
}


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


def _parse_classifier_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(raw)
    return json.loads(cleaned)


def _coerce_document_class(value: Any) -> DocumentClass:
    if value is None:
        raise ValueError("document_class 为空")
    s = str(value).strip()
    if s in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[s]
    lower = s.lower()
    if lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[lower]
    raise ValueError(f"无法识别的 document_class: {value!r}")


def _clamp_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def content_classifier_node(state: ImportGraphState) -> dict[str, Any]:
    """
    内容分类节点：利用 LLM 识别文档类型，写入 ``document_class``、``classifier_confidence``、``classifier_raw``。

    仅返回需合并入状态的片段（与 ``file_router_node`` 一致）；失败时向 ``errors`` 追加说明。
    """
    patch: dict[str, Any] = {}
    errors = list(state.get("errors") or [])

    api_doc = (state.get("api_doc_type") or "").strip().lower()
    if api_doc == "course_doc":
        patch["document_class"] = "lecture"
        patch["classifier_confidence"] = 1.0
        patch["classifier_raw"] = '{"forced_by":"api_doc_type","value":"course_doc"}'
        return patch
    if api_doc == "project_doc":
        patch["document_class"] = "project"
        patch["classifier_confidence"] = 1.0
        patch["classifier_raw"] = '{"forced_by":"api_doc_type","value":"project_doc"}'
        return patch

    text = (state.get("normalized_text") or "").strip()
    if not text:
        errors.append("没有可分类的内容")
        patch["errors"] = errors
        return patch

    sample_text = text[:_SAMPLE_LEN]
    prompt = CLASSIFIER_PROMPT.format(text_sample=sample_text)

    try:
        llm = get_llm_client()
        response = llm.chat(prompt=prompt, json_mode=True)
        res_data = _parse_classifier_json(response)

        doc_cls = _coerce_document_class(res_data.get("document_class"))
        # 校验与 Enum 一致（含新增 project）
        DocumentClassEnum(doc_cls)

        patch["document_class"] = doc_cls
        patch["classifier_confidence"] = _clamp_confidence(res_data.get("confidence"))
        patch["classifier_raw"] = response
    except Exception as e:  # pragma: no cover - 网络/解析
        errors.append(f"分类器执行失败: {e}")
        patch["errors"] = errors

    return patch


__all__ = ["CLASSIFIER_PROMPT", "content_classifier_node"]
