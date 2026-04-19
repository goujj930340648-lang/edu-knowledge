# 教育知识库（Edu Knowledge Base）

面向教育培训场景的智能知识库：**课程与文档检索**、**题库查询**、**带引用的知识问答**（多轮、流式、历史记录）。与 `knowledge`（掌柜智库）仓库**相互独立**，仅技术思路可对齐，无代码依赖。

当前阶段：**仅项目目录与设计文档**；实现将在后续迭代中按文档展开。

## 文档索引

| 文档 | 说明 |
|------|------|
| [docs/DESIGN.md](docs/DESIGN.md) | 目标、范围、目录规划、数据与流水线思路 |
| [docs/CONFIG.md](docs/CONFIG.md) | 环境变量说明，含 PDF 导入开关（预留，默认关闭） |

## 环境与配置

1. 复制 `config/env.example` 为 `.env`。
2. `PDF_IMPORT_ENABLED`：**预留**，默认 `false`；PDF 导入未实现前请勿依赖该开关行为。

## 目录骨架（占位）

`api/`、`processor/`（含nodes/、adapters/子目录）、`schema/`、`services/`、`utils/`、`config/`、`scripts/`、`prompts/`、`temp_data/` 等为后续代码预留；当前仅含 `.gitkeep`，无业务代码。
