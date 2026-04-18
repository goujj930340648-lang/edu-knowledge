# CLAUDE.md

本文件仅包含核心开发命令和代码规范。无需解释"为什么"，照做即可。

## 核心命令

```bash
# 环境准备
pip install -r requirements.txt
cp config/env.example .env  # 编辑 .env 填写密钥

# 启动服务
python scripts/server.py

# 测试导入/查询管线
python scripts/import_document.py 文档.md
python scripts/query_once.py "问题" --answer-only

# 测试
pytest
pytest tests/processor/vector_indexer/test_vector_indexer_integration.py -v

# Milvus 初始化
python scripts/init_milvus.py
python scripts/ping_milvus.py
```

## 代码规范

### 必须遵守

1. **所有模块必须** `from __future__ import annotations`
2. **State 用 TypedDict，API 用 Pydantic**
3. **模块级 `__all__` 显式导出**
4. **中文注释可接受**（教育领域项目）
5. **错误累积而非快速失败**：errors/warnings 列表

### 禁止行为

1. **禁止"顺便优化"**：只改必需的代码
2. **禁止过度抽象**：一次用不到的逻辑不要抽函数
3. **禁止 speculative error handling**：不可能发生的场景不处理
4. **禁止删除他人遗留的 dead code**：除非是你的改动导致的孤儿代码

### 架构约束

- **导入管线**：File_Router → Content_Classifier → Extractors → Vector_Indexer
- **PDF 路径**：`PDF_IMPORT_ENABLED=false` 时不实现，快速失败返回明确错误
- **向量模式**：legacy（单表）vs v2（双表 BGE-M3），通过 `MILVUS_RAG_MODE` 切换
- **环境变量**：所有配置走 .env，`load_dotenv(override=True)` 确保覆盖系统环境

### 测试原则

- TDD：先写失败测试，再实现
- 每个测试独立运行
- 用 fixture 管理测试数据，禁止硬编码
- 集成测试测真实 Milvus 操作，单元测试用 mock

### 修改原则

- 触手可及：只改任务相关的代码
- 匹配现有风格：即使你不习惯
- 你的改动造成的孤儿代码必须删除
- 发现无关问题：提出来，不要顺手修

## 目录结构速查

```
processor/       # LangGraph 管线节点
  - nodes/       # 查询处理节点（扁平化）
  - adapters/    # 文件解析适配器
  - importer/    # 导入管线逻辑
app/routers/     # FastAPI 路由
config/          # 配置文件（query_config, vector_config, settings）
schema/          # Pydantic 模型
services/        # 业务逻辑编排
scripts/         # 命令行工具（server, import_document, query_once等）
utils/           # 客户端单例（Milvus/Mongo/LLM）
tests/           # 测试，镜像源码结构
```