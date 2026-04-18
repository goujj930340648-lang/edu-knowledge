# 教育知识库项目知识地图

> 生成时间：2026-04-18
> 项目规模：中型（81个Python源码文件）
> 技术栈：Python, FastAPI, LangGraph, Milvus, MongoDB, BGE-M3

---

## 1. 项目知识地图总览

### 核心领域
**教育培训场景的智能知识库系统**：支持课程与文档检索、题库查询、带引用的知识问答（多轮、流式、历史记录）。

### 技术栈

**后端框架：**
- FastAPI 0.115+（Web服务）
- LangGraph 0.2+（状态图编排）
- Pydantic 2.0+（数据验证）

**存储：**
- Milvus 2.4+（向量数据库）
- MongoDB 4.6+（结构化数据）
- MinIO 7.2+（对象存储）

**AI/ML：**
- OpenAI Embeddings（稠密向量）
- BGE-M3（本地混合向量：稠密+稀疏）
- LangChain Text Splitters（文档切分）

**开发工具：**
- pytest（测试）
- python-dotenv（环境变量管理）

### 目录结构（Top 3层）

```
edu-knowledge/
├── app/                          # FastAPI应用层
│   ├── main.py                   # 应用入口，路由注册
│   ├── deps.py                   # 依赖注入
│   ├── streaming.py              # SSE流式输出工具
│   ├── schemas.py                # API请求/响应模型
│   └── routers/                  # 路由模块
│       ├── ingest.py             # 文档导入API
│       ├── search.py             # 向量搜索API
│       ├── chat.py               # 对话API
│       └── health.py             # 健康检查
│
├── processor/                    # 核心处理逻辑
│   ├── adapters/                 # 文件解析适配器
│   │   ├── pdf_adapter.py        # PDF解析
│   │   ├── docx_adapter.py       # DOCX解析
│   │   ├── catalog_md.py         # 课程目录Markdown解析
│   │   └── questions_md.py       # 题库Markdown解析
│   ├── extractors/               # 内容提取器
│   │   ├── content_classifier.py # 文档分类器
│   │   ├── lecture_extractor.py  # 讲义提取
│   │   ├── question_extractor.py # 题目提取
│   │   └── syllabus_extractor.py # 大纲提取
│   ├── nodes/                    # 查询处理节点（LangGraph）
│   │   ├── answer.py             # 答案生成节点
│   │   ├── course_catalog.py     # 课程目录检索节点
│   │   ├── dense_search.py       # 稠密向量检索
│   │   ├── hybrid_search.py      # 混合向量检索
│   │   ├── hyde_search.py        # HyDE假设文档检索
│   │   ├── query_rewrite.py      # 查询改写节点
│   │   ├── reranker.py           # 重排序节点
│   │   ├── rrf_merge.py          # RRF融合节点
│   │   └── sparse_search.py      # 稀疏向量检索
│   ├── utils/                    # 工具模块
│   │   ├── base.py               # 基础节点类
│   │   ├── embed_util.py         # 向量化工具
│   │   └── exceptions.py         # 异常定义
│   ├── vector_indexer/           # 向量索引（策略模式）
│   │   ├── indexer.py            # 策略实现（Legacy/V2）
│   │   ├── embedding_service.py  # Embedding服务Protocol
│   │   ├── node.py               # LangGraph节点包装
│   │   └── utils.py              # 索引工具函数
│   ├── import_graph.py           # 导入管线状态图
│   ├── query_graph.py            # 查询管线状态图
│   ├── import_state.py           # 导入状态TypedDict
│   ├── query_state.py            # 查询状态TypedDict
│   └── file_router.py            # 文件路由节点
│
├── config/                       # 配置管理
│   ├── settings.py               # Pydantic配置类
│   ├── query_config.py           # 查询管线配置
│   └── vector_config.py          # 向量索引配置
│
├── schema/                       # Pydantic模型
│   ├── edu_content.py            # 教育内容模型
│   └── metadata.py               # 元数据模型
│
├── services/                     # 业务逻辑编排
│   ├── ingest_runner.py          # 导入任务编排
│   ├── vector_doc_search.py      # 向量搜索服务
│   ├── chat_engine.py            # 对话引擎
│   ├── chat_stream_runner.py     # 流式对话编排
│   ├── docx_image_upload.py      # DOCX图片上传
│   ├── citation_assets.py        # 引用资源处理
│   └── source_mapping_rules.py   # 来源映射规则
│
├── storage/                      # 存储客户端
│   ├── mongo_db.py               # MongoDB客户端
│   ├── minio_client.py           # MinIO客户端
│   └── repository.py             # 数据仓库层
│
├── utils/                        # 全局工具
│   ├── client.py                 # LLM/Milvus客户端单例
│   ├── document_split.py         # 文档切分
│   ├── local_bge_client.py       # 本地BGE-M3客户端
│   ├── local_bge_reranker.py     # 本地BGE重排序
│   ├── milvus_search_edu.py      # Milvus教育搜索封装
│   ├── retrieval_rrf.py          # RRF融合算法
│   ├── converter.py              # 格式转换工具
│   └── markdown_table_linearizer.py # Markdown表格处理
│
├── scripts/                      # 运行脚本（唯一入口）
│   ├── server.py                 # FastAPI服务器
│   ├── import_document.py        # 单文件导入测试
│   ├── query_once.py             # 单次查询测试
│   ├── init_milvus.py            # Milvus初始化
│   └── ping_milvus.py            # Milvus连通性测试
│
├── tests/                        # 测试（镜像源码结构）
├── web/                          # 前端静态文件
└── docs/                         # 文档
```

---

## 2. 关键文件功能清单

| 文件路径 | 类型 | 主要职责 | 导出内容数量 | 依赖外部模块 |
|---------|------|----------|-------------|-------------|
| **应用层** |
| app/main.py | 应用入口 | FastAPI应用初始化、路由注册 | 4 | fastapi, app.routers |
| app/routers/ingest.py | 路由 | 文档导入API（课程/题库/文档） | 5 | fastapi, services |
| app/routers/chat.py | 路由 | 对话API（同步+流式） | 3 | fastapi, services |
| app/routers/search.py | 路由 | 向量搜索API | 2 | fastapi, services |
| **核心处理** |
| processor/import_graph.py | 状态图 | 导入管线编排（LangGraph） | 3 | langgraph, processor.* |
| processor/query_graph.py | 状态图 | 查询管线编排（LangGraph） | 3 | langgraph, processor.nodes |
| processor/vector_indexer/indexer.py | 策略实现 | 向量索引策略模式 | 5 | pymilvus, schema |
| processor/nodes/hybrid_search.py | 节点 | 混合向量检索节点 | 1 | pymilvus, utils |
| processor/nodes/dense_search.py | 节点 | 稠密向量检索节点 | 1 | pymilvus, utils |
| processor/nodes/sparse_search.py | 节点 | 稀疏向量检索节点 | 1 | pymilvus, utils |
| processor/nodes/reranker.py | 节点 | 重排序节点 | 1 | utils |
| **业务服务** |
| services/ingest_runner.py | 服务编排 | 导入任务后台执行 | 3 | mongo_db, processor |
| services/chat_engine.py | 业务逻辑 | 意图分类+问答编排 | 2 | langgraph, storage |
| services/vector_doc_search.py | 服务 | 向量搜索服务 | 2 | pymilvus, storage |
| **配置** |
| config/settings.py | 配置 | 环境变量配置管理 | 2 | pydantic_settings |
| config/vector_config.py | 配置 | 向量索引配置 | 3 | pydantic |
| **数据模型** |
| schema/edu_content.py | 模型 | 教育内容Pydantic模型 | 8 | pydantic |
| schema/metadata.py | 模型 | 元数据枚举和模型 | 5 | pydantic |
| **存储** |
| storage/repository.py | 数据仓库 | MongoDB读写操作 | 10+ | pymongo |
| storage/mongo_db.py | 客户端 | MongoDB连接管理 | 1 | pymongo |
| storage/minio_client.py | 客户端 | MinIO连接管理 | 1 | minio |

---

## 3. 核心函数/类一览表

### 按重要性排序（被调用次数估算）

| 文件 | 函数/类名 | 输入/输出 | 功能一句话描述 | 被调用次数 | 备注 |
|------|----------|----------|---------------|-----------|------|
| **策略模式核心** |
| processor/vector_indexer/indexer.py | `create_indexer()` | config → BaseIndexer | 向量索引器工厂函数 | 20+ | 核心工厂 |
| processor/vector_indexer/indexer.py | `BaseIndexer` | 抽象基类 | 向量索引器抽象基类 | 15+ | 策略基类 |
| processor/vector_indexer/indexer.py | `LegacyIndexer` | 继承BaseIndexer | 单集合+稠密向量索引器 | 10+ | legacy模式 |
| processor/vector_indexer/indexer.py | `V2Indexer` | 继承BaseIndexer | 双集合+混合向量索引器 | 10+ | v2模式 |
| processor/vector_indexer/embedding_service.py | `EmbeddingService` | Protocol | 向量化服务接口契约 | 20+ | Protocol定义 |
| **状态图核心** |
| processor/import_graph.py | `build_import_graph()` | → CompiledStateGraph | 构建导入管线状态图 | 10+ | 导入主入口 |
| processor/query_graph.py | `create_query_graph()` | → CompiledStateGraph | 构建查询管线状态图 | 15+ | 查询主入口 |
| processor/query_graph.py | `route_after_catalog()` | state → bool | 课程目录后路由决策 | 10+ | 条件路由 |
| **查询节点** |
| processor/nodes/hybrid_search.py | `HybridVectorSearchNode` | state → dict | 混合向量检索（稠密+稀疏） | 15+ | 核心检索 |
| processor/nodes/dense_search.py | `DenseVectorSearchNode` | state → dict | 稠密向量检索 | 10+ | v2模式 |
| processor/nodes/sparse_search.py | `SparseVectorSearchNode` | state → dict | 稀疏向量检索 | 10+ | v2模式 |
| processor/nodes/hyde_search.py | `HyDeVectorSearchNode` | state → dict | HyDE假设文档检索 | 10+ | 查询扩展 |
| processor/nodes/reranker.py | `RerankerNode` | state → dict | 重排序节点 | 10+ | 结果优化 |
| processor/nodes/rrf_merge.py | `RrfMergeNode` | state → dict | RRF融合节点 | 10+ | 多路融合 |
| processor/nodes/course_catalog.py | `CourseCatalogNode` | state → dict | 课程目录检索节点 | 10+ | 元数据检索 |
| processor/nodes/query_rewrite.py | `QueryRewriteNode` | state → dict | 查询改写节点 | 10+ | 查询优化 |
| processor/nodes/answer.py | `AnswerOutputNode` | state → dict | 答案生成节点 | 15+ | 最终输出 |
| **业务服务** |
| services/chat_engine.py | `classify_intent()` | query → intent | 意图分类（规则+LLM） | 10+ | 4类意图 |
| services/chat_engine.py | `run_chat_sync()` | query → result | 同步问答主流程 | 15+ | 对话入口 |
| services/ingest_runner.py | `run_catalog_import()` | task_id, path → None | 课程目录导入 | 5+ | 后台任务 |
| services/ingest_runner.py | `run_questions_import()` | task_id, path → None | 题库导入 | 5+ | 后台任务 |
| services/ingest_runner.py | `run_documents_import()` | task_id, paths → None | 文档批量导入 | 10+ | 后台任务 |
| services/vector_doc_search.py | `search_documents_with_hydrate()` | query → hits | Milvus搜索+Mongo回表 | 10+ | 向量搜索 |
| **数据模型** |
| schema/edu_content.py | `EduContent` | Pydantic模型 | 统一教育内容模型 | 20+ | 核心模型 |
| schema/edu_content.py | `ContentType` | Enum | 内容类型枚举（3种） | 15+ | 类型标记 |
| schema/metadata.py | `DocumentClass` | Enum | 文档分类枚举（4种） | 10+ | 导入分类 |
| **配置管理** |
| config/settings.py | `get_settings()` | → Settings | 配置单例（LRU缓存） | 30+ | 配置入口 |
| config/vector_config.py | `VectorIndexerConfig` | Pydantic模型 | 向量索引配置类 | 15+ | 索引配置 |
| **存储层** |
| storage/repository.py | `upsert_course_catalog()` | db, data → stats | 课程目录写入 | 5+ | 批量写入 |
| storage/repository.py | `upsert_question_data()` | db, data → stats | 题库数据写入 | 5+ | 批量写入 |
| storage/repository.py | `search_courses()` | db, kw → rows | 课程搜索 | 10+ | 全文搜索 |
| storage/repository.py | `search_questions()` | db, kw → rows | 题目搜索 | 10+ | 全文搜索 |
| storage/repository.py | `search_document_chunks_by_text()` | db, kw → rows | 文档块搜索 | 10+ | 全文搜索 |
| storage/mongo_db.py | `get_mongo_db()` | uri, name → Database | MongoDB连接获取 | 20+ | 连接管理 |
| **工具函数** |
| utils/milvus_search_edu.py | `rag_mode()` | → "legacy" | 获取RAG模式 | 20+ | 模式检测 |
| utils/milvus_search_edu.py | `hybrid_search_chunks_v2()` | vectors → hits | v2混合搜索 | 10+ | Milvus封装 |
| utils/milvus_search_edu.py | `dense_search_chunks_v2()` | vector → hits | v2稠密搜索 | 10+ | Milvus封装 |
| utils/milvus_search_edu.py | `dense_search_legacy_chunks()` | vector → hits | legacy稠密搜索 | 10+ | Milvus封装 |
| utils/retrieval_rrf.py | `rrf_merge()` | lists → merged | RRF融合算法 | 10+ | 融合算法 |
| utils/document_split.py | `split_markdown_by_headers()` | md → chunks | Markdown标题切分 | 10+ | 文档切分 |
| utils/client.py | `get_milvus_client()` | → MilvusClient | Milvus客户端单例 | 20+ | 客户端管理 |
| utils/client.py | `get_llm_client()` | → LLMClient | LLM客户端单例 | 15+ | 客户端管理 |

---

## 4. 跨文件依赖关系图

### 主要调用链

#### 4.1 导入流程

```
app/routers/ingest.py (HTTP请求)
  └─> services/ingest_runner.py (任务编排)
        ├─> processor/import_graph.py (LangGraph状态图)
        │     ├─> processor/file_router.py (文件路由)
        │     ├─> processor/extractors/content_classifier.py (文档分类)
        │     ├─> processor/extractors/lecture_extractor.py (讲义提取)
        │     ├─> processor/extractors/question_extractor.py (题目提取)
        │     ├─> processor/extractors/syllabus_extractor.py (大纲提取)
        │     └─> processor/vector_indexer/indexer.py (向量索引)
        │           ├─> processor/vector_indexer/embedding_service.py (向量化)
        │           └─> utils/client.py (Milvus客户端)
        └─> storage/repository.py (MongoDB写入)
```

#### 4.2 查询流程

```
app/routers/chat.py (HTTP请求)
  └─> services/chat_engine.py (意图分类+编排)
        ├─> storage/repository.py (结构化检索：课程/题库)
        └─> processor/query_graph.py (LangGraph状态图)
              ├─> processor/nodes/course_catalog.py (课程目录)
              ├─> processor/nodes/query_rewrite.py (查询改写)
              ├─> processor/nodes/dense_search.py (稠密检索)
              ├─> processor/nodes/sparse_search.py (稀疏检索)
              ├─> processor/nodes/hybrid_search.py (混合检索)
              ├─> processor/nodes/hyde_search.py (HyDE检索)
              ├─> processor/nodes/rrf_merge.py (RRF融合)
              ├─> processor/nodes/reranker.py (重排序)
              └─> processor/nodes/answer.py (答案生成)
                    └─> utils/client.py (LLM客户端)
```

#### 4.3 向量索引策略模式

```
config/vector_config.py (配置)
  └─> processor/vector_indexer/indexer.py
        ├─> create_indexer() (工厂函数)
        ├─> BaseIndexer (抽象基类)
        ├─> LegacyIndexer (legacy模式策略)
        │     └─> EmbeddingService (Protocol)
        │           └─> utils/local_bge_client.py 或 OpenAI
        └─> V2Indexer (v2模式策略)
              └─> EmbeddingService (Protocol)
                    └─> utils/local_bge_client.py (BGE-M3)
```

#### 4.4 配置管理

```
config/settings.py (Pydantic Settings)
  ├─> app/main.py (应用启动)
  ├─> services/ingest_runner.py (导入任务)
  ├─> services/chat_engine.py (对话引擎)
  ├─> storage/mongo_db.py (MongoDB连接)
  └─> storage/minio_client.py (MinIO连接)

config/vector_config.py (向量索引配置)
  ├─> processor/vector_indexer/indexer.py (索引器)
  └─> processor/nodes/* (检索节点)
```

---

## 5. 架构决策记录

### 关键设计选择

#### 5.1 向量索引策略模式
**决策：** 使用策略模式实现向量索引，支持 `legacy` 和 `v2` 两种模式

**原因：**
- `legacy` 模式：单集合 + OpenAI稠密向量（向后兼容）
- `v2` 模式：双集合 + BGE-M3混合向量（稠密+稀疏，提升召回）
- 运行时通过 `MILVUS_RAG_MODE` 环境变量切换

**实现：**
- 抽象基类：`BaseIndexer`
- 具体策略：`LegacyIndexer`, `V2Indexer`
- 工厂函数：`create_indexer(config)`
- Protocol接口：`EmbeddingService`

**参考文件：** `processor/vector_indexer/indexer.py`, `processor/vector_indexer/embedding_service.py`

#### 5.2 LangGraph状态图编排
**决策：** 使用 LangGraph 构建导入和查询管线

**原因：**
- 可视化管线流程（调试友好）
- 支持条件路由和并行执行
- 状态类型安全（TypedDict）
- 易于插入检查点和中间件

**实现：**
- 导入管线：`processor/import_graph.py`（6个节点）
- 查询管线：`processor/query_graph.py`（8个节点）
- 状态定义：`processor/import_state.py`, `processor/query_state.py`

#### 5.3 扁平化目录结构
**决策：** 限制目录嵌套不超过3层，优先扁平组织

**原因：**
- 降低认知负担（开发者快速定位代码）
- 避免深层嵌套导致的路径复杂度
- 相关文件就近放置，而非按技术分层

**示例：**
- ✅ `processor/nodes/hybrid_search.py`（3层）
- ❌ `processor/query_process/nodes/answer_output_node.py`（5层）

**强制规则：** 禁止使用 `*_node.py` 后缀，直接使用 `*.py`

#### 5.4 组合优于继承
**决策：** 优先使用 Protocol 和组合模式，而非深层继承树

**原因：**
- Python 的 Protocol 支持鸭子类型
- 避免继承导致的紧耦合
- 更容易测试和mock

**示例：**
```python
@runtime_checkable
class EmbeddingService(Protocol):
    def embed_documents(self, texts: list[str]) -> EmbeddingResult: ...

class LegacyIndexer:
    def __init__(self, embed_service: EmbeddingService): ...
```

#### 5.5 意图分类（规则+LLM回退）
**决策：** 先用规则分类，歧义时调用LLM

**原因：**
- 规则快（常见场景90%+命中率）
- LLM准（处理歧义和复杂查询）
- 降低成本和延迟

**4类意图：**
- `course_intro`：课程介绍查询
- `question_search`：题库搜索
- `doc_search`：文档检索
- `knowledge_qa`：综合知识问答

**实现：** `services/chat_engine.py::classify_intent()`

#### 5.6 混合检索（Dense + Sparse + HyDE）
**决策：** v2模式使用3路并行检索 + RRF融合

**原因：**
- Dense：语义相似度（关键词不足时补充）
- Sparse：精确关键词匹配（专有名词）
- HyDE：查询扩展（假设文档提升召回）
- RRF：多路融合（避免单路失败）

**实现：** `processor/query_graph.py` 并行节点 + `processor/nodes/rrf_merge.py`

#### 5.7 Pydantic配置管理
**决策：** 使用 Pydantic Settings 管理环境变量

**原因：**
- 类型安全（自动验证）
- 默认值支持（降低配置负担）
- 单例模式（LRU缓存）
- 项目根目录固定（避免CWD问题）

**实现：** `config/settings.py::get_settings()`

---

## 6. 可复用组件/工具函数推荐

### 建议抽成公共库的模块

#### 6.1 向量索引框架 ⭐⭐⭐
**路径：** `processor/vector_indexer/`

**价值：** 策略模式的向量索引框架，可用于其他RAG项目

**核心特性：**
- 策略模式支持多种索引器
- Protocol接口定义清晰
- 支持去重、批量写入、错误处理
- Legacy和V2模式切换

**复用建议：**
```python
from processor.vector_indexer import create_indexer
from config.vector_config import VectorIndexerConfig

config = VectorIndexerConfig(mode="v2")
indexer = create_indexer(config)
result = indexer.index_chunks(chunks)
```

#### 6.2 文档切分工具 ⭐⭐
**路径：** `utils/document_split.py`

**价值：** 标题感知的智能Markdown切分

**核心特性：**
- 按标题层级切分（h1 > h2 > h3）
- 保持上下文完整性
- 支持自定义切分粒度

**复用建议：**
```python
from utils.document_split import split_markdown_by_headers

chunks = split_markdown_by_headers(
    markdown_text,
    max_chunk_size=1000,
    min_chunk_size=200
)
```

#### 6.3 RRF融合算法 ⭐⭐
**路径：** `utils/retrieval_rrf.py`

**价值：** 通用多路检索融合算法

**核心特性：**
- Reciprocal Rank Fusion
- 可配置权重（dense_weight, sparse_weight）
- 支持任意数量的检索路

**复用建议：**
```python
from utils.retrieval_rrf import rrf_merge

merged = rrf_merge(
    [dense_results, sparse_results, hyde_results],
    weights=[0.5, 0.3, 0.2],
    k=60
)
```

#### 6.4 通用向量搜索服务 ⭐
**路径：** `services/vector_doc_search.py`

**价值：** 封装Milvus搜索+MongoDB回表

**核心特性：**
- Milvus向量搜索
- MongoDB回表补充元数据
- 支持HyDE和Rerank

**复用建议：**
```python
from services.vector_doc_search import search_documents_with_hydrate

hits = search_documents_with_hydrate(
    db,
    query="机器学习算法",
    limit=10,
    use_hyde=True,
    use_rerank=True
)
```

#### 6.5 LangGraph节点基类 ⭐
**路径：** `processor/utils/base.py`

**价值：** 统一的LangGraph节点基类

**核心特性：**
- 标准化节点接口
- 日志记录
- 配置管理

**复用建议：**
```python
from processor.utils.base import BaseNode

class MyCustomNode(BaseNode):
    name = "my_custom_node"
    def process(self, state: MyState) -> dict:
        # 节点逻辑
        return {"key": "value"}
```

---

## 7. 环境变量配置清单

### 必需配置

```bash
# MongoDB
MONGODB_URI=mongodb://localhost:27017  # 或 MONGO_URL（兼容）
MONGODB_DATABASE=edu_knowledge

# Milvus
MILVUS_URI=http://127.0.0.1:19530
MILVUS_RAG_MODE=legacy  # legacy 或 v2

# LLM
ACTIVE_API_KEY=sk-xxx  # OpenAI或兼容API
```

### 可选配置

```bash
# MinIO（对象存储）
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=education-knowledge
MINIO_PUBLIC_BASE_URL=http://127.0.0.1:9000/education-knowledge

# Embedding后端
EMBEDDING_BACKEND=openai  # openai 或 local_bge_m3

# 意图分类
INTENT_USE_LLM=false  # true时使用LLM分类

# API服务
API_HOST=0.0.0.0
API_PORT=8000
API_KEYS=key1,key2  # 逗号分隔，非空时要求X-API-Key请求头

# 高级特性
CHAT_DOC_SEARCH_HYDE=false  # 文档搜索是否使用HyDE
CHAT_DOC_SEARCH_RERANK=false  # 文档搜索是否使用Rerank
```

---

## 8. 核心命令速查

### 服务启动
```bash
python scripts/server.py              # 启动FastAPI服务器（默认8000端口）
```

### 测试脚本
```bash
python scripts/import_document.py 课程目录.md  # 测试单文件导入
python scripts/query_once.py "机器学习算法有哪些"  # 测试单次查询
python scripts/init_milvus.py         # 初始化Milvus集合
python scripts/ping_milvus.py         # 测试Milvus连通性
```

### 测试
```bash
pytest                                    # 全量测试
pytest tests/processor/vector_indexer/   # 特定目录测试
pytest -v                                 # 详细输出
pytest --tb=short                         # 简洁错误信息
```

---

## 9. 开发准则摘要

### 文件命名（强制）
- ❌ 禁止：`*_node.py`, `*_magic.py`, `*_media.py`
- ✅ 正确：`hybrid_search.py`, `pdf_adapter.py`, `content_classifier.py`

### 目录结构（强制）
- 扁平优于嵌套：最多3层
- 相关文件就近放置，不按技术分层

### 代码风格（强制）
- 所有函数必须有 Type Hints
- 公共API必须有 Google 风格 Docstring
- 必须启用 `from __future__ import annotations`
- 优先使用现代类型：`list[str]` 而非 `List[str]`

### 修改原则（强制）
- 禁止"顺便优化"：只改任务相关代码
- 禁止过度抽象：一次用不到的逻辑不抽函数
- 禁止 speculative error handling：不可能发生的场景不处理
- 修改向量索引代码前必须先读 `processor/vector_indexer/`

### 测试驱动（推荐）
1. 先写失败测试
2. 实现最小代码使测试通过
3. 运行 `pytest` 验证
4. 提交代码

---

## 10. 常见问题排查

### Milvus连接失败
```bash
# 测试连通性
python scripts/ping_milvus.py

# 检查环境变量
echo $MILVUS_URI
```

### 向量化失败
```bash
# 检查模式配置
echo $MILVUS_RAG_MODE  # legacy 或 v2
echo $EMBEDDING_BACKEND  # openai 或 local_bge_m3

# v2模式需要BGE-M3
# legacy模式需要OpenAI API Key
```

### MongoDB写入失败
```bash
# 检查连接
echo $MONGODB_URI

# 测试写入
python -c "from storage.mongo_db import get_mongo_db; print(get_mongo_db())"
```

### 导入任务卡住
```bash
# 检查后台任务状态
# 1. 查询MongoDB的 ingest_tasks 集合
# 2. 查看status字段：running/completed/failed
# 3. 查看logs字段了解进度
```

---

**文档结束**

如需更新本文档，请在重大架构变更后重新生成。
建议保存到项目根目录，便于团队成员查阅。
