# 架构重塑方案

> **重构原则**：扁平化、语义化命名、职责分离、最小改��

**目标**：减少目录层级、统一命名规范、分离脚本与业务代码

---

## 重构前后对比表

### 1. 根目录脚本移动

| 当前路径 | 重构后路径 | 说明 |
|---------|-----------|------|
| `run_import_once.py` | `scripts/import_document.py` | 重命名，更语义化 |
| `run_query_once.py` | `scripts/query_once.py` | 重命名 |
| `run_server.py` | `scripts/server.py` | 重命名 |
| `init_milvus.py` | `scripts/init_milvus.py` | 移动 |
| `milvus_ping.py` | `scripts/ping_milvus.py` | 重命名 |
| `main_graph.py` | `processor/import_graph.py` | 移动到 processor/，重命名 |

### 2. 文件重命名（语义化）

| 当前路径 | 重构后路径 | 理由 |
|---------|-----------|------|
| `processor/pdf_magic.py` | `processor/pdf_adapter.py` | "adapter" 比 "magic" 更明确 |
| `processor/docx_media.py` | `processor/docx_adapter.py` | 统一 adapter 命名 |
| `parsers/catalog_md.py` | `processor/adapters/catalog_md.py` | 合并零碎目录 |
| `parsers/questions_md.py` | `processor/adapters/questions_md.py` | 合并零碎目录 |
| `app/main.py` | `app/main.py` | **保持不变**（FastAPI 生态惯例） |

### 3. 目录扁平化

| 当前结构 | 重构后结构 | 改进 |
|---------|-----------|------|
| `processor/query_process/nodes/*.py` | `processor/nodes/*.py` | 减少1层嵌套，更简洁 |
| `processor/vector_indexer/*.py` | `processor/vector_indexer/*.py` | 保持（已合理） |
| `parsers/*.py` | `processor/adapters/*.py` | 合并零碎目录 |

**具体文件移动：**
```
processor/query_process/nodes/hybrid_vector_search_node.py  → processor/nodes/hybrid_search.py
processor/query_process/nodes/hyde_vector_search_node.py    → processor/nodes/hyde_search.py
processor/query_process/nodes/dense_vector_search_node.py   → processor/nodes/dense_search.py
processor/query_process/nodes/sparse_vector_search_node.py  → processor/nodes/sparse_search.py
processor/query_process/nodes/rrf_merge_node.py             → processor/nodes/rrf_merge.py
processor/query_process/nodes/reranker_node.py              → processor/nodes/reranker.py
processor/query_process/nodes/answer_output_node.py         → processor/nodes/answer.py
processor/query_process/nodes/query_rewrite_node.py         → processor/nodes/query_rewrite.py
processor/query_process/nodes/course_catalog_node.py        → processor/nodes/course_catalog.py
parsers/catalog_md.py                                       → processor/adapters/catalog_md.py
parsers/questions_md.py                                      → processor/adapters/questions_md.py
```

### 4. 配置文件统一

| 当前路径 | 重构后路径 | 说明 |
|---------|-----------|------|
| `.env.example` | `config/env.example` | 统一配置目录 |
| `processor/query_process/config.py` | `config/query_config.py` | 统一配置管理 |
| `processor/vector_indexer/config.py` | `config/vector_config.py` | 统一配置管理 |

### 5. 目录结构总览

**重构前：**
```
.
├── run_*.py (5个脚本)
├── main_graph.py
├── init_milvus.py
├── milvus_ping.py
├── .env.example
├── processor/
│   ├── pdf_magic.py
│   ├── docx_media.py
│   └── query_process/
│       └── nodes/ (嵌套2层)
└── app/
    └── main.py
```

**重构后：**
```
.
├── scripts/              # 所有运行脚本
│   ├── import_document.py
│   ├── query_once.py
│   ├── server.py
│   ├── init_milvus.py
│   └── ping_milvus.py
├── config/               # 统一配置管理
│   ├── env.example
│   ├── query_config.py
│   └── vector_config.py
├── processor/            # 扁平化
│   ├── import_graph.py
│   ├── pdf_adapter.py
│   ├── docx_adapter.py
│   ├── adapters/         # 合并后的解析器
│   │   ├── catalog_md.py
│   │   └── questions_md.py
│   ├── nodes/            # 扁平化的节点
│   └── vector_indexer/
├── app/
│   └── main.py           # 保持不变
└── parsers/              # 已删除
```

---

## 重构影响评估

### Import 路径修复清单

**需要更新的文件：**
1. `scripts/*.py` - 更新对 processor/、app/ 的引用
2. `processor/query_process/main_graph.py` - 更新对 nodes 的引用
3. `processor/query_process/base.py` - 更新对 nodes 的引用
4. `tests/` - 所有测试文件的 import 路径
5. `app/deps.py` - 更新配置路径
6. 所有引用 `parsers.` 的文件改为 `processor.adapters.`

**预估修复数量：** 约 15-20 个文件

### 风险点

1. **外部调用**：如果用户有直接调用 `python run_server.py` 的脚本/文档，需更新
2. **CI/CD**：检查是否有路径写死在流水线中
3. **文档**：README.md、CLAUDE.md 中的命令示例需同步更新

---

## 执行计划（待确认后执行）

### 阶段1：创建新目录结构（5分钟）
```bash
mkdir -p scripts
mkdir -p processor/nodes
mkdir -p processor/adapters
```

### 阶段2：移动脚本文件（2分钟）
```bash
git mv run_import_once.py scripts/import_document.py
git mv run_query_once.py scripts/query_once.py
git mv run_server.py scripts/server.py
git mv init_milvus.py scripts/init_milvus.py
git mv milvus_ping.py scripts/ping_milvus.py
git mv main_graph.py processor/import_graph.py
```

### 阶段3：重命名语义化文件（2分钟）
```bash
git mv processor/pdf_magic.py processor/pdf_adapter.py
git mv processor/docx_media.py processor/docx_adapter.py
# app/main.py 保持不变（FastAPI 生态惯例）
```

### 阶段4：扁平化 nodes + 合并 parsers（3分钟）
```bash
# 扁平化查询节点
git mv processor/query_process/nodes/*.py processor/nodes/
# 合并 parsers 目录
git mv parsers/catalog_md.py processor/adapters/
git mv parsers/questions_md.py processor/adapters/
git mv parsers/__init__.py processor/adapters/
```

### 阶段5：统一配置文件（2分钟）
```bash
git mv .env.example config/env.example
git mv processor/query_process/config.py config/query_config.py
git mv processor/vector_indexer/config.py config/vector_config.py
```

### 阶段6：自动修复 import 路径（10分钟）
使用 subagent 批量修复所有 import 语句

### 阶段7：更新文档（2分钟）
- 更新 CLAUDE.md 中的命令
- 更新 README.md 中的示例
- 更新 .env.example 引用

### 阶段8：验证（5分钟）
```bash
# 测试导入
python scripts/import_document.py tests/fixtures/sample.md

# 测试查询
python scripts/query_once.py "测试问题" --answer-only

# 运行测试套件
pytest

# 启动服务器
python scripts/server.py
```

---

## 人工确认项

请确认以下问题后，我将开始执行重构：

1. **目录命名**：`scripts/`、`config/`、`processor/query_nodes/` 是否符合预期？
2. **文件重命名**：
   - `pdf_magic.py` → `pdf_adapter.py` ✓
   - `docx_media.py` → `docx_adapter.py` ✓
   - `main.py` → `server.py` ✓
   - 还有其他建议吗？
3. **保留目录**：`parsers/`、`prompts/`、`temp_data/`、`web/` 是否需要调整？
4. **风险接受**：import 路径自动修复可能有遗漏，是否接受手动测试修复？

**确认方式**：回复"确认执行"或提出修改建议。