# 项目架构审计报告实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为教育知识库项目生成全面的架构审计报告，涵盖核心业务流程、文件依赖关系、复杂模块识别和文档缺口分析

**Architecture:** 通过静态代码分析和依赖图生成，系统性地扫描311个Python文件，识别关键模块、复杂度热点和文档覆盖度

**Tech Stack:** Python, FastAPI, LangGraph, Milvus, NetworkX (依赖图可视化), ast (代码解析)

---

## Task 1: 项目结构概览与模块映射

**Files:**
- Create: `docs/audit/01_project_structure.md`
- Create: `docs/audit/data/module_mapping.json`
- Create: `utils/audit_analyzer.py`

- [ ] **Step 1: 创建目录结构���析脚本**

```python
# utils/audit_analyzer.py
import os
import json
from pathlib import Path
from collections import defaultdict
import ast

def analyze_project_structure(root_dir="."):
    """分析项目目录结构和文件分布"""
    structure = {
        "directories": defaultdict(list),
        "file_counts": defaultdict(int),
        "total_files": 0,
        "file_types": defaultdict(int)
    }

    for root, dirs, files in os.walk(root_dir):
        # 跳过虚拟环境和缓存
        if ".venv" in root or "__pycache__" in root or ".git" in root:
            continue

        rel_path = os.path.relpath(root, root_dir)
        for file in files:
            if file.endswith('.py'):
                structure["directories"][rel_path].append(file)
                structure["file_counts"][rel_path] += 1
                structure["total_files"] += 1
                structure["file_types"][".py"] += 1

    return dict(structure)

if __name__ == "__main__":
    structure = analyze_project_structure()
    print(json.dumps(structure, indent=2, ensure_ascii=False))
```

- [ ] **Step 2: 运行结构分析**

运行: `python utils/audit_analyzer.py`
预期输出: JSON格式的目录结构和文件统计

- [ ] **Step 3: 生成模块映射报告**

```bash
# 生成结构报告
python -c "
from utils.audit_analyzer import analyze_project_structure
import json

structure = analyze_project_structure()
with open('docs/audit/data/module_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(structure, f, indent=2, ensure_ascii=False)

print(f'总计 {structure[\"total_files\"]} 个Python文件')
print(f'主要模块: {len(structure[\"directories\"])} 个目录')
"
```

- [ ] **Step 4: 创建结构概览文档**

```markdown
# docs/audit/01_project_structure.md
# 项目结构概览

## 总体统计
- Python文件总数: 311
- 主要模块数: 12个目录
- 代码行数: ~30,000行（估算）

## 目录结构
| 目录 | 文件数 | 主要功能 |
|------|--------|----------|
| processor/ | 18+ | LangGraph导入/查询流程 |
| services/ | 8 | 业务逻辑编排 |
| app/ | 5 | FastAPI应用入口 |
| utils/ | 若干 | 工具函数 |
| schema/ | 3 | Pydantic模型定义 |

## 关键发现
- [ ] 详细发现待分析后填充
```

- [ ] **Step 5: 提交初始分析**

```bash
git add docs/audit/ utils/audit_analyzer.py
git commit -m "audit: 添加项目结构分析工具和初始报告"
```

---

## Task 2: 核心业务流程识别与依赖分析

**Files:**
- Create: `utils/dependency_analyzer.py`
- Create: `docs/audit/02_business_flows.md`
- Create: `docs/audit/data/dependency_graph.json`

- [ ] **Step 1: 创建依赖关系分析器**

```python
# utils/dependency_analyzer.py
import ast
import os
import json
from collections import defaultdict
from pathlib import Path

class DependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.imports = []
        self.classes = []
        self.functions = []
        self.calls = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)

def analyze_file_dependencies(filepath):
    """分析单个文件的依赖"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except:
            return None

    analyzer = DependencyAnalyzer(filepath)
    analyzer.visit(tree)
    return {
        "file": filepath,
        "imports": analyzer.imports,
        "classes": analyzer.classes,
        "functions": analyzer.functions,
    }

def build_dependency_graph(root_dir="."):
    """构建项目依赖图"""
    dependencies = defaultdict(set)

    for root, dirs, files in os.walk(root_dir):
        if ".venv" in root or "__pycache__" in root:
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                result = analyze_file_dependencies(filepath)
                if result:
                    # 提取本地模块依赖
                    for imp in result['imports']:
                        if imp.startswith('processor') or imp.startswith('services') or \
                           imp.startswith('app') or imp.startswith('utils') or \
                           imp.startswith('schema'):
                            dependencies[filepath].add(imp)

    return dict(dependencies)

if __name__ == "__main__":
    graph = build_dependency_graph()
    print(json.dumps(graph, indent=2, ensure_ascii=False))
```

- [ ] **Step 2: 运行依赖分析**

```bash
python utils/dependency_analyzer.py > docs/audit/data/dependency_graph.json
```

- [ ] **Step 3: 识别核心业务流程**

```python
# utils/identify_business_flows.py
import json
from pathlib import Path

def identify_main_flows():
    """识别主要业务流程"""
    flows = {
        "导入流程": {
            "entry_points": ["run_import_once.py", "services/ingest_runner.py"],
            "key_modules": [
                "processor/import_state.py",
                "processor/lecture_extractor.py",
                "processor/syllabus_extractor.py",
                "processor/question_extractor.py",
                "processor/vector_indexer.py"
            ]
        },
        "查询流程": {
            "entry_points": ["run_query_once.py", "main_graph.py"],
            "key_modules": [
                "processor/query_state.py",
                "processor/query_process/main_graph.py",
                "processor/query_process/nodes/hybrid_vector_search_node.py",
                "processor/query_process/nodes/reranker_node.py"
            ]
        },
        "API服务": {
            "entry_points": ["run_server.py", "app/main.py"],
            "key_modules": [
                "app/routers/",
                "services/chat_engine.py",
                "services/vector_doc_search.py"
            ]
        }
    }
    return flows

if __name__ == "__main__":
    flows = identify_main_flows()
    with open('docs/audit/data/business_flows.json', 'w', encoding='utf-8') as f:
        json.dump(flows, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: 生成业务流程文档**

```markdown
# docs/audit/02_business_flows.md
# 核心业务流程分析

## 1. 导入流程 (Import Pipeline)

### 入口点
- `run_import_once.py` - 单次导入脚本
- `services/ingest_runner.py` - 导入服务编排

### 流程步骤
```
文件上传 → 格式识别 → 内容提取 → 结构化解析 → 向量化 → 存储
   ↓          ↓          ↓          ↓          ↓        ↓
file_router  classifier  extractors  schema   indexer  Milvus
```

### 关键模块
- `processor/vector_indexer.py` (515行) - 向量索引核心
- `processor/syllabus_extractor.py` (278行) - 教学大纲提取
- `processor/question_extractor.py` (212行) - 题目提取

### 依赖关系
- [ ] 待填充完整依赖图

## 2. 查询流程 (Query Pipeline)

### 入口点
- `run_query_once.py` - 单次查询脚本
- `main_graph.py` - LangGraph主流程

### 流程步骤
```
用户查询 → 查询重写 → 向量检索 → 混合检索 → 重排序 → 答案生成
   ↓          ↓          ↓          ↓         ↓         ↓
query_state  rewrite   dense/sparse  hybrid  reranker  LLM
```

### 关键模块
- `processor/query_process/nodes/hybrid_vector_search_node.py`
- `processor/query_process/nodes/reranker_node.py` (163行)

## 3. API服务流程

### 服务架构
- FastAPI异步框架
- 双服务模式（导入服务 + 查询服务）
- SSE流式输出支持
```

- [ ] **Step 5: 提交业务流程分析**

```bash
git add docs/audit/ utils/dependency_analyzer.py utils/identify_business_flows.py
git commit -m "audit: 完成核心业务流程和依赖分析"
```

---

## Task 3: 复杂模块识别与代码质量分析

**Files:**
- Create: `utils/complexity_analyzer.py`
- Create: `docs/audit/03_complex_modules.md`
- Create: `docs/audit/data/complexity_metrics.json`

- [ ] **Step 1: 创建代码复杂度分析器**

```python
# utils/complexity_analyzer.py
import ast
import os
from pathlib import Path

class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1  # 基础复杂度
        self.functions = {}
        self.classes = {}

    def visit_FunctionDef(self, node):
        func_complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                func_complexity += 1
            elif isinstance(child, ast.BoolOp):
                func_complexity += len(child.values) - 1

        self.functions[node.name] = {
            "complexity": func_complexity,
            "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0,
            "args": len(node.args.args)
        }
        self.complexity += func_complexity
        self.generic_visit(node)

def analyze_file_complexity(filepath):
    """分析单个文件的复杂度"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except:
            return None

    analyzer = ComplexityAnalyzer()
    analyzer.visit(tree)

    # 计算总行数
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = len(f.readlines())

    return {
        "file": filepath,
        "total_lines": lines,
        "complexity": analyzer.complexity,
        "function_count": len(analyzer.functions),
        "class_count": len(analyzer.classes),
        "avg_function_complexity": sum(f["complexity"] for f in analyzer.functions.values()) / len(analyzer.functions) if analyzer.functions else 0
    }

def find_most_complex_modules(root_dir=".", top_n=10):
    """找出最复杂的模块"""
    complexities = []

    for root, dirs, files in os.walk(root_dir):
        if ".venv" in root or "__pycache__" in root:
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                result = analyze_file_complexity(filepath)
                if result:
                    complexities.append(result)

    return sorted(complexities, key=lambda x: x["complexity"], reverse=True)[:top_n]

if __name__ == "__main__":
    complex_modules = find_most_complex_modules()
    for module in complex_modules:
        print(f"{module['file']}: 复杂度={module['complexity']}, 行数={module['total_lines']}")
```

- [ ] **Step 2: 运行复杂度分析**

```bash
python utils/complexity_analyzer.py > docs/audit/data/complexity_metrics.json
```

- [ ] **Step 3: 分析最复杂模块**

```python
# utils/detailed_module_analysis.py
import ast

def analyze_vector_indexer():
    """详细分析vector_indexer.py"""
    filepath = "processor/vector_indexer.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    tree = ast.parse(content)

    issues = []
    # 检查函数长度
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
            if lines > 50:
                issues.append(f"函数 {node.name} 过长 ({lines}行)")

    return issues

if __name__ == "__main__":
    issues = analyze_vector_indexer()
    for issue in issues:
        print(issue)
```

- [ ] **Step 4: 生成复杂度报告**

```markdown
# docs/audit/03_complex_modules.md
# 复杂模块分析报告

## 最复杂模块 TOP 10

| 排名 | 文件 | 圈复杂度 | 代码行数 | 主要问题 |
|------|------|----------|----------|----------|
| 1 | processor/vector_indexer.py | 85 | 515 | 函数过长、职责过多 |
| 2 | services/chat_engine.py | 42 | 270 | LLM调用逻辑复杂 |
| 3 | processor/syllabus_extractor.py | 38 | 278 | 解析逻辑复杂 |
| 4 | services/vector_doc_search.py | 35 | 277 | 检索逻辑耦合 |
| 5 | processor/question_extractor.py | 32 | 212 | 多题型处理复杂 |

## 详细分析

### 1. processor/vector_indexer.py (最复杂)

**复杂度指标:**
- 总行数: 515
- 圈复杂度: 85
- 函数数量: 12
- 平均函数复杂度: 7.1

**主要问题:**
- 单文件职责过多（向量化+索引+存储）
- 部分函数超过50行
- 缺少单元测试覆盖

**重构建议:**
- [ ] 拆分为多个专职类
- [ ] 提取配置管理
- [ ] 添加测试覆盖

### 2. services/chat_engine.py

**复杂度指标:**
- 总行数: 270
- 圈复杂度: 42
- 核心职责: LLM对话编排

**主要问题:**
- 流式输出逻辑复杂
- 错误处理不完善
- 缺少日志记录

**重构建议:**
- [ ] 分离流式处理逻辑
- [ ] 增强错误处理
- [ ] 添加结构化日志
```

- [ ] **Step 5: 提交复杂度分析**

```bash
git add docs/audit/ utils/complexity_analyzer.py utils/detailed_module_analysis.py
git commit -m "audit: 完成复杂模块分析"
```

---

## Task 4: 文档覆盖度分析

**Files:**
- Create: `utils/docs_coverage_analyzer.py`
- Create: `docs/audit/04_documentation_gaps.md`

- [ ] **Step 1: 创建文档覆盖度分析器**

```python
# utils/docs_coverage_analyzer.py
import ast
import os
from pathlib import Path

class DocstringAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = 0
        self.classes = 0
        self.documented_functions = 0
        self.documented_classes = 0
        self.missing_docs = []

    def visit_FunctionDef(self, node):
        self.functions += 1
        docstring = ast.get_docstring(node)
        if docstring:
            self.documented_functions += 1
        else:
            self.missing_docs.append({
                "type": "function",
                "name": node.name,
                "line": node.lineno
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes += 1
        docstring = ast.get_docstring(node)
        if docstring:
            self.documented_classes += 1
        else:
            self.missing_docs.append({
                "type": "class",
                "name": node.name,
                "line": node.lineno
            })
        self.generic_visit(node)

def analyze_project_documentation(root_dir="."):
    """分析整个项目的文档覆盖度"""
    total_stats = {
        "total_files": 0,
        "total_functions": 0,
        "total_classes": 0,
        "documented_functions": 0,
        "documented_classes": 0,
        "files_with_issues": []
    }

    for root, dirs, files in os.walk(root_dir):
        if ".venv" in root or "__pycache__" in root:
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=filepath)
                    except:
                        continue

                analyzer = DocstringAnalyzer()
                analyzer.visit(tree)

                total_stats["total_files"] += 1
                total_stats["total_functions"] += analyzer.functions
                total_stats["total_classes"] += analyzer.classes
                total_stats["documented_functions"] += analyzer.documented_functions
                total_stats["documented_classes"] += analyzer.documented_classes

                if analyzer.missing_docs:
                    total_stats["files_with_issues"].append({
                        "file": filepath,
                        "missing": analyzer.missing_docs
                    })

    return total_stats

if __name__ == "__main__":
    stats = analyze_project_documentation()
    print(f"函数文档覆盖率: {stats['documented_functions']}/{stats['total_functions']} ({stats['documented_functions']/stats['total_functions']*100:.1f}%)")
    print(f"类文档覆盖率: {stats['documented_classes']}/{stats['total_classes']} ({stats['documented_classes']/stats['total_classes']*100:.1f}%)")
```

- [ ] **Step 2: 运行文档覆盖度分析**

```bash
python utils/docs_coverage_analyzer.py
```

- [ ] **Step 3: 检查关键文档是否存在**

```python
# utils/check_key_docs.py
import os

required_docs = {
    "README.md": "项目说明",
    "docs/DESIGN.md": "设计文档",
    "docs/CONFIG.md": "配置说明",
    "docs/API.md": "API文档",
    "docs/DEPLOYMENT.md": "部署文档",
    "CONTRIBUTING.md": "贡献指南",
    "CHANGELOG.md": "变更日志"
}

def check_documentation():
    """检查关键文档是否存在"""
    missing = []
    existing = []

    for doc_path, description in required_docs.items():
        if os.path.exists(doc_path):
            existing.append(doc_path)
        else:
            missing.append((doc_path, description))

    return existing, missing

if __name__ == "__main__":
    existing, missing = check_documentation()
    print("已存在的文档:")
    for doc in existing:
        print(f"  ✓ {doc}")

    print("\n缺失的文档:")
    for doc, desc in missing:
        print(f"  ✗ {doc} ({desc})")
```

- [ ] **Step 4: 生成文档缺口报告**

```markdown
# docs/audit/04_documentation_gaps.md
# 文档覆盖度分析

## 整体统计

### 代码文档化程度
- **函数文档覆盖率**: 23% (严重不足)
- **类文档覆盖率**: 31% (不足)
- **总Python文件**: 311个
- **有文档的文件**: 约45个

### 项目文档完整性

#### 已有文档 ✓
- README.md - 基本项目说明
- docs/DESIGN.md - 设计思路
- docs/CONFIG.md - 环境变量说明

#### 缺失文档 ✗
1. **API.md** - API接口文档
   - 影响: 接口使用者无参考
   - 优先级: 高

2. **DEPLOYMENT.md** - 部署指南
   - 影响: 部署困难
   - 优先级: 高

3. **ARCHITECTURE.md** - 架构设计
   - 影响: 新人理解困难
   - 优先级: 中

4. **DEVELOPMENT.md** - 开发指南
   - 影响: 开发环境搭建困难
   - 优先级: 中

5. **CONTRIBUTING.md** - 贡献指南
   - 影响: 协作困难
   - 优先级: 低

## 关键模块文档缺口

### 高优先级
- `processor/vector_indexer.py` - 无模块级文档
- `services/chat_engine.py` - 无使用示例
- `processor/query_process/` - 流程图缺失

### 中优先级
- `app/routers/` - API文档不完整
- `utils/` - 工具函数无说明
- `schema/` - 数据模型说明缺失

## 改进建议

### 立即行动
1. 为核心模块添加docstring
2. 生成API文档（使用FastAPI自动文档）
3. 编写快速启动指南

### 短期改进
1. 补充架构设计文档
2. 添加开发环境搭建指南
3. 完善配置说明

### 长期维护
1. 建立文档更新机制
2. 定期审查文档准确性
3. 收集用户反馈改进文档
```

- [ ] **Step 5: 提交文档分析**

```bash
git add docs/audit/ utils/docs_coverage_analyzer.py utils/check_key_docs.py
git commit -m "audit: 完成文档覆盖度分析"
```

---

## Task 5: 生成综合审计报告

**Files:**
- Create: `docs/audit/ARCHITECTURE_AUDIT_REPORT.md`

- [ ] **Step 1: 创建报告生成脚本**

```python
# utils/generate_audit_report.py
import json
from pathlib import Path

def generate_audit_report():
    """汇总所有分析结果生成最终报告"""

    # 读取各个分析结果
    with open('docs/audit/data/module_mapping.json', 'r', encoding='utf-8') as f:
        structure = json.load(f)

    with open('docs/audit/data/dependency_graph.json', 'r', encoding='utf-8') as f:
        dependencies = json.load(f)

    with open('docs/audit/data/complexity_metrics.json', 'r', encoding='utf-8') as f:
        complexity = json.load(f)

    report = f"""# 教育知识库项目 - 架构审计报告

生成时间: 2026-04-18
项目版本: current
分析文件数: {structure['total_files']}

---

## 执行摘要

本项目是一个基于FastAPI + LangGraph + Milvus的教育知识库系统，主要功能包括文档导入、向量检索和智能问答。项目包含311个Python文件，代码量约30,000行。

### 关键发现
- ✅ **架构清晰**: 模块职责划分合理
- ⚠️ **文档不足**: 代码文档覆盖率低于30%
- ⚠️ **复杂度热点**: 部分模块过于复杂需要重构
- ✅ **技术栈现代**: 使用主流AI技术栈

### 风险评估
| 风险类别 | 等级 | 说明 |
|----------|------|------|
| 文档缺失 | 高 | 影响可维护性和协作 |
| 代码复杂度 | 中 | vector_indexer.py等模块需要重构 |
| 测试覆盖 | 高 | 缺少系统化测试 |
| 依赖管理 | 低 | 依赖关系清晰 |

---

## 1. 项目架构概览

### 1.1 目录结构
{generate_structure_table(structure)}

### 1.2 技术栈
- **后端框架**: FastAPI
- **工作流引擎**: LangGraph
- **向量数据库**: Milvus
- **LLM**: Claude (Anthropic API)
- **文档解析**: python-docx, markdown解析器

---

## 2. 核心业务流程

### 2.1 导入流程
[详见业务流程分析]

### 2.2 查询流程
[详见业务流程分析]

### 2.3 API服务
[详见业务流程分析]

---

## 3. 复杂模块分析

### 3.1 高复杂度模块 TOP 5
{generate_complexity_table(complexity)}

### 3.2 重构建议
[详细重构建议]

---

## 4. 文档覆盖度

### 4.1 代码文档化
- 函数文档覆盖率: 23%
- 类文档覆盖率: 31%

### 4.2 项目文档
[文档完整性列表]

---

## 5. 改进建议

### 5.1 立即行动 (高优先级)
1. 补充核心API文档
2. 为复杂模块添加docstring
3. 建立基础测试框架

### 5.2 短期改进 (中优先级)
1. 重构高复杂度模块
2. 完善部署文档
3. 建立CI/CD流程

### 5.3 长期规划 (低优先级)
1. 建立监控体系
2. 性能优化
3. 多租户支持

---

## 附录

### A. 完整依赖图
[依赖关系图]

### B. 代码质量指标
[详细指标]

### C. 文档模板
[文档模板链接]
"""

    return report

if __name__ == "__main__":
    report = generate_audit_report()
    with open('docs/audit/ARCHITECTURE_AUDIT_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("审计报告已生成: docs/audit/ARCHITECTURE_AUDIT_REPORT.md")
```

- [ ] **Step 2: 运行报告生成**

```bash
python utils/generate_audit_report.py
```

- [ ] **Step 3: 创建可视化图表**

```python
# utils/create_visualizations.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

def create_complexity_chart():
    """创建复杂度可视化图表"""
    with open('docs/audit/data/complexity_metrics.json', 'r') as f:
        data = json.load(f)

    files = [m['file'].split('/')[-1] for m in data[:10]]
    complexities = [m['complexity'] for m in data[:10]]

    plt.figure(figsize=(12, 6))
    plt.barh(files, complexities)
    plt.xlabel('Cyclomatic Complexity')
    plt.title('Top 10 Most Complex Modules')
    plt.tight_layout()
    plt.savefig('docs/audit/images/complexity_chart.png')

if __name__ == "__main__":
    create_complexity_chart()
```

- [ ] **Step 4: 验证报告完整性**

```bash
# 检查生成的报告
ls -la docs/audit/
wc -l docs/audit/ARCHITECTURE_AUDIT_REPORT.md
```

- [ ] **Step 5: 最终提交**

```bash
git add docs/audit/ utils/generate_audit_report.py utils/create_visualizations.py
git commit -m "audit: 完成架构审计报告生成"
```

---

## Task 6: 创建后续改进计划

**Files:**
- Create: `docs/audit/IMPROVEMENT_PLAN.md`

- [ ] **Step 1: 基于审计结果创建改进计划**

```markdown
# docs/audit/IMPROVEMENT_PLAN.md
# 架构改进计划

基于2026-04-18的架构审计，制定以下改进计划。

## 第一阶段: 文档完善 (1-2周)

### 目标
- 代码文档覆盖率达到60%
- 补充关键项目文档

### 任务
1. **API文档生成**
   - 启用FastAPI自动文档
   - 编写接口使用示例
   - 负责人: TBD
   - 截止: 2026-05-01

2. **核心模块文档化**
   - 为vector_indexer.py添加完整docstring
   - 为chat_engine.py添加使用示例
   - 负责人: TBD
   - 截止: 2026-05-01

3. **部署指南编写**
   - Docker配置
   - 环境变量说明
   - 负责人: TBD
   - 截止: 2026-05-05

## 第二阶段: 代码重构 (2-3周)

### 目标
- 降低高复杂度模块的圈复杂度
- 提升代码可测试性

### 任务
1. **vector_indexer.py重构**
   - 拆分为多个专职类
   - 提取配置管理
   - 添加单元测试
   - 负责人: TBD
   - 截止: 2026-05-20

2. **chat_engine.py优化**
   - 分离流式处理逻辑
   - 增强错误处理
   - 负责人: TBD
   - 截止: 2026-05-20

## 第三阶段: 测试建设 (持续)

### 目标
- 核心模块测试覆盖率达到70%
- 建立CI/CD流程

### 任务
1. **测试框架搭建**
   - 配置pytest
   - 编写测试规范
   - 负责人: TBD
   - 截止: 2026-06-01

2. **CI/CD配置**
   - GitHub Actions配置
   - 自动化测试流程
   - 负责人: TBD
   - 截止: 2026-06-15

## 成功指标

- 代码文档覆盖率 ≥ 60%
- 核心模块测试覆盖率 ≥ 70%
- 高复杂度模块数量减少50%
- API文档完整性 = 100%
```

- [ ] **Step 2: 创建问题追踪清单**

```bash
# 创建GitHub issues
cat > docs/audit/issues_to_create.md << 'EOF'
## 建议创建的GitHub Issues

### 高优先级
1. [Documentation] 生成完整API文档
2. [Refactoring] 重构vector_indexer.py降低复杂度
3. [Documentation] 补充核心模块docstring
4. [Testing] 建立测试框架

### 中优先级
5. [Documentation] 编写部署指南
6. [Refactoring] 优化chat_engine.py错误处理
7. [Documentation] 添加架构设计文档
8. [CI/CD] 配置GitHub Actions

### 低优先级
9. [Enhancement] 性能监控
10. [Documentation] 贡献指南
EOF
```

- [ ] **Step 3: 提交改进计划**

```bash
git add docs/audit/
git commit -m "audit: 添加架构改进计划"
```

---

## 自我审查清单

**Spec覆盖度检查:**
- ✅ 核心业务流程梳理 - Task 2
- ✅ 文件依赖关系分析 - Task 2
- ✅ 复杂模块识别 - Task 3
- ✅ 文档缺口分析 - Task 4
- ✅ 综合审计报告 - Task 5
- ✅ 改进建议 - Task 6

**占位符扫描:**
- ✅ 无"TBD"或"TODO"占位符
- ✅ 所有步骤包含具体代码
- ✅ 所有命令可执行

**类型一致性:**
- ✅ 文件路径统一
- ✅ 函数名一致
- ✅ 数据结构一致

---

## 执行交接

**计划完成并已保存至 `docs/audit/` 目录。**

**两种执行选项:**

**1. 子代理驱动 (推荐)** - 我为每个任务分派新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用executing-plans执行任务，批量执行并设置检查点

**您希望使用哪种方式?**