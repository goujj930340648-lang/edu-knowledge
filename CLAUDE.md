# CLAUDE.md

本文件是项目开发的"宪法"，所有 AI 助手和开发者必须严格遵守。

---

## 项目全景图

```
edu-knowledge/
├── scripts/                 # 运行脚本（唯一入口）
│   ├── server.py           # FastAPI 服务器
│   ├── import_document.py  # 单文件导入测试
│   ├── query_once.py       # 单次查询测试
│   ├── init_milvus.py      # Milvus 初始化
│   └── ping_milvus.py      # Milvus 连通性测试
│
├── config/                  # 配置管理
│   ├── env.example         # 环境变量模板
│   ├── settings.py         # Pydantic 配置类
│   ├── query_config.py     # 查询管线配置
│   └── vector_config.py    # 向量索引配置
│
├── processor/               # 核心处理逻辑（扁平结构）
│   ├── adapters/           # 文件解析适配器
│   │   ├── pdf_adapter.py
│   │   ├── docx_adapter.py
│   │   ├── catalog_md.py
│   │   └── questions_md.py
│   ├── extractors/         # 内容提取器
│   │   ├── content_classifier.py
│   │   ├── lecture_extractor.py
│   │   ├── question_extractor.py
│   │   └── syllabus_extractor.py
│   ├── nodes/              # 查询处理节点（无 _node 后缀）
│   │   ├── answer.py
│   │   ├── course_catalog.py
│   │   ├── dense_search.py
│   │   ├── hybrid_search.py
│   │   ├── hyde_search.py
│   │   ├── query_rewrite.py
│   │   ├── reranker.py
│   │   ├── rrf_merge.py
│   │   └── sparse_search.py
│   ├── utils/              # 工具模块
│   │   ├── base.py
│   │   ├── embed_util.py
│   │   └── exceptions.py
│   ├── vector_indexer/     # 向量索引（策略模式）
│   │   ├── indexer.py      # BaseIndexer, LegacyIndexer, V2Indexer
│   │   ├── embedding_service.py  # EmbeddingService Protocol
│   │   ├── node.py         # LangGraph 节点包装
│   │   └── utils.py
│   ├── import_graph.py     # 导入管线图
│   ├── query_graph.py      # 查询管线图
│   ├── import_state.py     # 导入状态 TypedDict
│   └── query_state.py      # 查询状态 TypedDict
│
├── app/                     # FastAPI 应用
│   ├── main.py             # 应用入口
│   ├── deps.py             # 依赖注入
│   ├── streaming.py        # SSE 工具
│   └── routers/            # 路由
│       ├── ingest.py
│       ├── search.py
│       ├── chat.py
│       └── health.py
│
├── schema/                  # Pydantic 模型
│   ├── edu_content.py      # 教育内容模型
│   └── metadata.py         # 元数据模型
│
├── services/                # 业务逻辑编排
│   ├── ingest_runner.py    # 导入任务编排
│   ├── vector_doc_search.py # 向量搜索服务
│   ├── chat_engine.py      # 对话引擎
│   └── ...
│
├── storage/                 # 存储客户端
│   ├── mongo_db.py
│   ├── minio_client.py
│   └── repository.py
│
├── utils/                   # 全局工具
│   ├── client.py           # 客户端单例
│   ├── document_split.py   # 文档切分
│   └── ...
│
├── tests/                   # 测试（镜像源码结构）
└── web/                     # 前端静态文件
```

---

## 核心命令（固化）

### 环境准备
```bash
pip install -r requirements.txt
cp config/env.example .env  # 编辑 .env 填写密钥
```

### 运行服务
```bash
python scripts/server.py              # 启动 FastAPI 服务器
python scripts/import_document.py 文档.md  # 测试导入
python scripts/query_once.py "问题"   # 测试查询
python scripts/init_milvus.py         # 初始化 Milvus
python scripts/ping_milvus.py         # 测试 Milvus 连通性
```

### 测试
```bash
pytest                                    # 全量测试
pytest tests/processor/vector_indexer/   # 特定目录
pytest -v                                 # 详细输出
pytest --tb=short                         # 简洁错误信息
```

---

## 开发准则（强制遵守）

### 1. 文件命名（严禁违反）

**禁止使用的后缀：**
- ❌ `*_node.py`（已禁止，使用 `*.py` 代替）
- ❌ `*_magic.py`（已禁止，使用 `*_adapter.py`）
- ❌ `*_media.py`（已禁止，使用 `*_adapter.py`）

**正确示例：**
- ✓ `course_catalog.py`（不是 `course_catalog_node.py`）
- ✓ `pdf_adapter.py`（不是 `pdf_magic.py`）
- ✓ `content_classifier.py`（不是 `content_classifier_node.py`）

### 2. 目录结构原则

**扁平优于嵌套：**
- ✓ `processor/nodes/answer.py`（3层）
- ❌ `processor/query_process/nodes/answer_output_node.py`（5层）

**规则：**
- 新建目录前先问：能否合并到现有目录？
- 目录嵌套不超过 3 层
- 相关文件放在一起，而非按技术分层

### 3. 代码组织原则

**组合��于继承：**
```python
# ✓ 组合：使用 Protocol 和 Strategy 模式
@runtime_checkable
class EmbeddingService(Protocol):
    def embed_documents(self, texts: list[str]) -> EmbeddingResult: ...

class LegacyIndexer:
    def __init__(self, embed_service: EmbeddingService): ...

# ❌ 继承：深层继承树
class AbstractEmbeddingService(ABC):
    ...

class OpenAIEmbeddingService(AbstractEmbeddingService):
    ...
```

**参考实现：** `processor/vector_indexer/indexer.py`、`processor/vector_indexer/embedding_service.py`

**单一职责：**
- 一个文件只做一件事
- 超过 300 行考虑拆分
- 相关函数放在一起，不相关功能分文件

### 4. 修改前必读（上下文保护）

**修改向量索引相关代码前，必须参考：**
```
processor/vector_indexer/indexer.py       # 策略模式实现
processor/vector_indexer/embedding_service.py  # Protocol 设计
processor/vector_indexer/utils.py         # 工具函数
```

**关键模式：**
- 策略模式：`BaseIndexer` + `LegacyIndexer` + `V2Indexer`
- 工厂函数：`create_indexer(config)` 返回具体策略
- Protocol：`EmbeddingService` 定义接口契约
- 错误处理：`EmbeddingResult` / `EmbeddingError` 联合类型

---

## 风格约束（严格检查）

### 1. Type Hints（必须）

**所有函数必须有类型注解：**
```python
# ✓ 正确
def process_document(
    file_path: str,
    options: dict[str, Any] | None = None,
) -> list[EduContent]:
    ...

# ❌ 错误
def process_document(file_path, options=None):
    ...
```

**使用现代类型：**
```python
from typing import Any  # 仅在无法确定类型时使用

# ✓ 优先使用具体类型
def fetch_items(limit: int) -> list[EduContent]: ...

# ✓ 联合类型
def parse_input(data: str | bytes) -> dict[str, Any]: ...

# ✓ Optional（而不是 None | T）
def get_config(key: str) -> str | None: ...
```

### 2. Google 风格 Docstrings（必须）

**模块级 Docstring：**
```python
"""
向量索引策略实现。

本模块实现向量索引的策略模式，包括：
- IndexerResult: 索引结果数据类
- BaseIndexer: 抽象基类，提供通用功能
- LegacyIndexer: 单集合 + 稠密向量（OpenAI/legacy 模式）
- V2Indexer: 双集合 + 混合向量（BGE-M3/v2 模式）
- create_indexer: 索引器实例化工厂函数

关键设计决策：
1. V2Indexer 仅调用一次 embed_documents(mode="hybrid")
2. 去重统一在 BaseIndexer._deduplicate_content_hashes()
3. 顺序执行（无并行化）以保持简单
4. 每个索引器类处理自己的逻辑，无条件分支
"""
```

**函数级 Docstring：**
```python
def create_indexer(config: VectorIndexerConfig) -> BaseIndexer:
    """根据配置创建向量索引器实例。

    Args:
        config: 向量索引器配置，决定使用 legacy 还是 v2 模式

    Returns:
        BaseIndexer: 具体的索引器实例（LegacyIndexer 或 V2Indexer）

    Raises:
        ValueError: 如果 config.mode 既不是 "legacy" 也不是 "v2"

    Example:
        >>> config = VectorIndexerConfig(mode="legacy")
        >>> indexer = create_indexer(config)
        >>> isinstance(indexer, LegacyIndexer)
        True
    """
```

**类级 Docstring：**
```python
class V2Indexer(BaseIndexer):
    """V2 模式向量索引器（双集合 + 混合向量）。

    使用 BGE-M3 生成稠密和稀疏向量，分别写入：
    - edu_knowledge_item_names_v1: 项目名称表
    - edu_knowledge_chunks_v1: 文档块表

    Attributes:
        config: V2 模式配置，包含 BGE-M3 路径和设备设置
        embed_service: BGE-M3 嵌入服务（单例）

    Note:
        V2Indexer 调用 embed_documents() 时使用 mode="hybrid"，
        一次性获取稠密和稀疏向量。
    """
```

### 3. 代码格式

**必须启用：**
```python
from __future__ import annotations  # 所有文件第一行
```

**导入顺序：**
```python
# 1. 标准库
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 2. 第三方库
from pydantic import BaseModel

# 3. 本地模块
from config.vector_config import VectorIndexerConfig
from processor.vector_indexer.utils import content_fingerprint
```

**模块级 `__all__`：**
```python
__all__ = [
    "create_indexer",
    "BaseIndexer",
    "LegacyIndexer",
    "V2Indexer",
    "IndexerResult",
]
```

---

## 禁止行为（严格执行）

1. **禁止"顺便优化"**：只改任务相关的代码
2. **禁止过度抽象**：一次用不到的逻辑不要抽函数
3. **禁止 speculative error handling**：不可能发生的场景不处理
4. **禁止删除他人遗留的 dead code**：除非是你的改动导致的孤儿代码
5. **禁止修改前不读现有模式**：向量索引代码必须先读 `processor/vector_indexer/`
6. **禁止使用 `_node` 后缀**：文件命名必须简洁明确
7. **禁止深层嵌套目录**：目录嵌套不超过 3 层
8. **禁止无 Type Hints**：所有函数必须有类型注解
9. **禁止无 Docstring**：公共 API 必须有 Google 风格文档

---

## 修改原则（触手可及）

**修改代码时：**
1. 只改必需的代码，不碰无关部分
2. 匹配现有风格，即使你不习惯
3. 你的改动造成的孤儿代码必须删除
4. 发现无关问题：提出来，不要顺手修

**测试驱动：**
1. 先写失败测试
2. 实现最小代码使测试通过
3. 运行 `pytest` 验证
4. 提交代码

---

## 快速参考

### LangGraph 状态管理
- 导入状态：`processor/import_state.py`（TypedDict）
- 查询状态：`processor/query_state.py`（TypedDict）

### 策略模式示例
- 索引器：`processor/vector_indexer/indexer.py`
- 嵌入服务：`processor/vector_indexer/embedding_service.py`

### 测试命令
```bash
pytest                                    # 全量
pytest -v                                 # 详细
pytest --tb=short                         # 简洁
pytest tests/processor/vector_indexer/   # 特定目录
```

### 环境变量
```bash
ACTIVE_API_KEY=sk-xxx                     # LLM 密钥
MILVUS_URI=http://127.0.0.1:19530         # Milvus 地址
MILVUS_RAG_MODE=legacy                    # legacy 或 v2
EMBEDDING_BACKEND=openai                  # openai 或 local_bge_m3
```

---

**违反本文件的修改将被拒绝。**