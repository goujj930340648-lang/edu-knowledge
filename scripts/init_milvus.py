"""
初始化 Milvus 集合（教育 RAG），**幂等**：

- 某集合 **已存在** 且未设 ``MILVUS_INIT_DROP`` → **跳过创建**（不删数据）。
- **不存在** → 创建 Schema、索引并 load。
- 数据写入由导入图完成；本脚本只负责 **表结构**。

- **legacy**（``MILVUS_RAG_MODE`` 非 ``v2``）：单表 ``MILVUS_COLLECTION``（默认 ``edu_knowledge_vectors_v1``）。
- **v2 双表**（始终检查/创建）：默认 ``edu_knowledge_item_names_v1``、``edu_knowledge_chunks_v1``（与本仓库一致，避免与其它项目 ``kb_*`` 混用）。

``run_import_once.py`` 会在跑图前自动调用 ``ensure_milvus_collections()``；也可手动执行 ``python init_milvus.py``。

使用前请启动 Milvus 2.4+，并安装 ``pymilvus``。

环境变量：

- ``MILVUS_URI``：默认 ``http://127.0.0.1:19530``
- ``MILVUS_TOKEN``：可选
- ``MILVUS_RAG_MODE``：非 ``v2`` 时会建 legacy 表；为 ``v2`` 时跳过 legacy 表
- ``MILVUS_COLLECTION``：legacy 表名，默认 ``edu_knowledge_vectors_v1``
- ``MILVUS_LEGACY_VECTOR_DIM``：仅在不使用本地 BGE 时生效；**本地 BGE-M3** 时以 ``BGE_M3_DIM``（默认 1024）建表，与 ``vector_indexer`` 一致
- ``MILVUS_NAMES_COLLECTION``：名称表，默认 ``edu_knowledge_item_names_v1``
- ``MILVUS_CHUNKS_COLLECTION``：切片表，默认 ``edu_knowledge_chunks_v1``
- ``BGE_M3_DIM``：v2 稠密向量维度，默认 ``1024``（``BAAI/bge-m3``）
- ``MILVUS_INIT_DROP``：设为 ``1`` 时先删后建（慎用）
- ``MILVUS_DISABLE_PROXY``：设为 ``1`` 时本进程清除 HTTP(S)/ALL_PROXY，避免 gRPC 连局域网 Milvus 被代理干扰

运行::

    python init_milvus.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _resolve_legacy_vector_dim() -> int:
    """
    legacy 单表 ``vector`` 维度须与 ``vector_indexer`` 实际 Embedding 一致：
    本地 BGE-M3 → ``BGE_M3_DIM``；OpenAI 兼容 API → ``MILVUS_LEGACY_VECTOR_DIM`` 或 1536。
    """
    from utils.local_bge_client import should_use_local_bge_embedding

    if should_use_local_bge_embedding():
        return int(os.environ.get("BGE_M3_DIM", "1024"))
    raw = (os.environ.get("MILVUS_LEGACY_VECTOR_DIM") or "1536").strip()
    try:
        return int(raw)
    except ValueError:
        return 1536


def _connect() -> None:
    from pymilvus import connections

    uri = (
        os.environ.get("MILVUS_URI") or os.environ.get("MILVUS_URL") or "http://127.0.0.1:19530"
    ).strip()
    token = (os.environ.get("MILVUS_TOKEN") or "").strip() or None
    kwargs: dict = {"uri": uri}
    if token:
        kwargs["token"] = token
    connections.connect("default", **kwargs)


def _create_names_collection(
    name: str, dim: int, drop: bool
) -> None:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    if utility.has_collection(name):
        if not drop:
            print(f"[skip] 集合已存在，跳过创建: {name}")
            return
        utility.drop_collection(name)
        print(f"[drop] {name}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="item_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="item_fingerprint", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(
        fields,
        description="课程/项目/题库名称索引（意图预搜索）",
        enable_dynamic_field=True,
    )
    col = Collection(name=name, schema=schema)
    col.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.create_index(
        field_name="item_name",
        index_params={"index_type": "INVERTED"},
    )
    col.load()
    print(f"[ok] 已创建并加载: {name}")


def _create_chunks_collection(
    name: str, dim: int, drop: bool
) -> None:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    if utility.has_collection(name):
        if not drop:
            print(f"[skip] 集合已存在，跳过创建: {name}")
            return
        utility.drop_collection(name)
        print(f"[drop] {name}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector_dense", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="vector_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="course_name", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="document_class", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="question_type", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(
        fields,
        description="正文切片（稠密+稀疏混合检索）",
        enable_dynamic_field=True,
    )
    col = Collection(name=name, schema=schema)
    col.create_index(
        field_name="vector_dense",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.create_index(
        field_name="vector_sparse",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2},
        },
    )
    for scalar in ("course_name", "content_type", "content_hash"):
        col.create_index(
            field_name=scalar,
            index_params={"index_type": "INVERTED"},
        )
    col.load()
    print(f"[ok] 已创建并加载: {name}")


def _create_legacy_collection(name: str, dim: int, drop: bool) -> None:
    """``MILVUS_RAG_MODE=legacy`` 使用的单表；含 ``vector`` 与 ``content_hash`` 去重字段，其余走动态列。"""
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    if utility.has_collection(name):
        if not drop:
            print(f"[skip] 集合已存在，跳过创建: {name}")
            return
        utility.drop_collection(name)
        print(f"[drop] {name}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(
        fields,
        description="legacy 单表入库（OpenAI 兼容 Embedding + vector）",
        enable_dynamic_field=True,
    )
    col = Collection(name=name, schema=schema)
    col.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.create_index(
        field_name="content_hash",
        index_params={"index_type": "INVERTED"},
    )
    col.load()
    print(f"[ok] 已创建并加载: {name}")


def ensure_milvus_collections() -> tuple[bool, str]:
    """
    连接 Milvus 并按环境变量 **确保** 所需集合：无则建表，有则跳过（除非 ``MILVUS_INIT_DROP``）。
    会先 ``load_dotenv``、再 ``maybe_strip_proxy_for_milvus``。
    返回 ``(True, "")`` 或 ``(False, 错误信息)``。
    """
    _load_dotenv()
    from utils.client import maybe_strip_proxy_for_milvus

    maybe_strip_proxy_for_milvus()

    drop = os.environ.get("MILVUS_INIT_DROP", "").strip() in ("1", "true", "yes")
    dim = int(os.environ.get("BGE_M3_DIM", "1024"))
    rag_mode = os.environ.get("MILVUS_RAG_MODE", "legacy").strip().lower()
    legacy_name = (
        os.environ.get("MILVUS_COLLECTION") or "edu_knowledge_vectors_v1"
    ).strip() or "edu_knowledge_vectors_v1"
    legacy_dim = _resolve_legacy_vector_dim()
    names = (
        os.environ.get("MILVUS_NAMES_COLLECTION")
        or os.environ.get("ITEM_NAME_COLLECTION")
        or "edu_knowledge_item_names_v1"
    ).strip()
    chunks = (
        os.environ.get("MILVUS_CHUNKS_COLLECTION")
        or os.environ.get("CHUNKS_COLLECTION")
        or "edu_knowledge_chunks_v1"
    ).strip()

    try:
        _connect()
    except Exception as e:
        return False, f"无法连接 Milvus: {e}"

    try:
        if rag_mode != "v2":
            print(
                f"[info] legacy 单表「{legacy_name}」向量维度: {legacy_dim} "
                "（须与当前 Embedding 输出一致；换模型后请删集合或换新 MILVUS_COLLECTION）"
            )
            _create_legacy_collection(legacy_name, legacy_dim, drop)
        _create_names_collection(names, dim, drop)
        _create_chunks_collection(chunks, dim, drop)
    except Exception as e:
        return False, str(e)

    return True, ""


def main() -> int:
    ok, err = ensure_milvus_collections()
    if not ok:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    print("[done] Milvus 集合初始化完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
