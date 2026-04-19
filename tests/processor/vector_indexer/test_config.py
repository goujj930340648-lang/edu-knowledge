"""Tests for vector_indexer configuration module."""

import os
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from config.vector_config import VectorIndexerConfig


class TestVectorIndexerConfig:
    """Tests for VectorIndexerConfig."""

    def test_default_values(self, monkeypatch):
        """Test default configuration values."""
        # Clear all relevant env vars
        for key in [
            "MILVUS_RAG_MODE",
            "MILVUS_COLLECTION",
            "MILVUS_NAMES_COLLECTION",
            "MILVUS_CHUNKS_COLLECTION",
            "ITEM_NAME_COLLECTION",
            "CHUNKS_COLLECTION",
            "MILVUS_SKIP_DEDUP",
            "EMBEDDING_BACKEND",
            "BGE_M3_PATH",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = VectorIndexerConfig.from_env()

        assert config.rag_mode == "legacy"
        assert config.legacy_collection == "edu_knowledge_vectors_v1"
        assert config.v2_names_collection == "edu_knowledge_item_names_v1"
        assert config.v2_chunks_collection == "edu_knowledge_chunks_v1"
        assert config.skip_dedup is False
        assert config.embedding_backend == "openai"

    def test_auto_local_bge_when_bge_m3_path_is_dir(self, monkeypatch, tmp_path):
        """未设置 EMBEDDING_BACKEND 且 BGE_M3_PATH 为已存在目录时默认本地 BGE-M3。"""
        for key in [
            "MILVUS_RAG_MODE",
            "MILVUS_COLLECTION",
            "MILVUS_NAMES_COLLECTION",
            "MILVUS_CHUNKS_COLLECTION",
            "ITEM_NAME_COLLECTION",
            "CHUNKS_COLLECTION",
            "MILVUS_SKIP_DEDUP",
            "EMBEDDING_BACKEND",
        ]:
            monkeypatch.delenv(key, raising=False)
        fake_model_dir = tmp_path / "bge-m3"
        fake_model_dir.mkdir()
        monkeypatch.setenv("BGE_M3_PATH", str(fake_model_dir))

        config = VectorIndexerConfig.from_env()

        assert config.embedding_backend == "local_bge_m3"

    def test_explicit_openai_overrides_local_path(self, monkeypatch, tmp_path):
        """显式 EMBEDDING_BACKEND=openai 时不因 BGE_M3_PATH 改成本地。"""
        for key in [
            "MILVUS_RAG_MODE",
            "MILVUS_COLLECTION",
            "MILVUS_NAMES_COLLECTION",
            "MILVUS_CHUNKS_COLLECTION",
            "ITEM_NAME_COLLECTION",
            "CHUNKS_COLLECTION",
            "MILVUS_SKIP_DEDUP",
        ]:
            monkeypatch.delenv(key, raising=False)
        fake_model_dir = tmp_path / "bge-m3"
        fake_model_dir.mkdir()
        monkeypatch.setenv("BGE_M3_PATH", str(fake_model_dir))
        monkeypatch.setenv("EMBEDDING_BACKEND", "openai")

        config = VectorIndexerConfig.from_env()

        assert config.embedding_backend == "openai"

    def test_legacy_mode_config(self, monkeypatch):
        """Test legacy mode configuration."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")
        monkeypatch.setenv("MILVUS_COLLECTION", "my_vectors")
        monkeypatch.setenv("MILVUS_SKIP_DEDUP", "1")
        monkeypatch.setenv("EMBEDDING_BACKEND", "local_bge_m3")

        config = VectorIndexerConfig.from_env()

        assert config.rag_mode == "legacy"
        assert config.legacy_collection == "my_vectors"
        assert config.skip_dedup is True
        assert config.embedding_backend == "local_bge_m3"

    def test_v2_mode_config(self, monkeypatch):
        """Test v2 mode configuration."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")
        monkeypatch.setenv("MILVUS_NAMES_COLLECTION", "my_names")
        monkeypatch.setenv("MILVUS_CHUNKS_COLLECTION", "my_chunks")
        monkeypatch.setenv("MILVUS_SKIP_DEDUP", "true")

        config = VectorIndexerConfig.from_env()

        assert config.rag_mode == "v2"
        assert config.v2_names_collection == "my_names"
        assert config.v2_chunks_collection == "my_chunks"
        assert config.skip_dedup is True

    def test_skip_dedup_various_values(self, monkeypatch):
        """Test MILVUS_SKIP_DEDUP with various truthy and falsy values."""
        truthy_values = ["1", "true", "yes", "on", "TRUE", "YES", "ON"]
        falsy_values = ["0", "false", "no", "off", "", "FALSE", "NO", "OFF"]

        for val in truthy_values:
            monkeypatch.setenv("MILVUS_SKIP_DEDUP", val)
            config = VectorIndexerConfig.from_env()
            assert config.skip_dedup is True, f"Expected True for {val!r}"

        for val in falsy_values:
            monkeypatch.setenv("MILVUS_SKIP_DEDUP", val)
            config = VectorIndexerConfig.from_env()
            assert config.skip_dedup is False, f"Expected False for {val!r}"

    def test_invalid_rag_mode(self, monkeypatch):
        """Test that invalid rag_mode raises ValueError."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "invalid")

        with pytest.raises(ValueError, match="Invalid MILVUS_RAG_MODE"):
            VectorIndexerConfig.from_env()

    def test_whitespace_handling(self, monkeypatch):
        """Test that whitespace is properly stripped from values."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "  v2  ")
        monkeypatch.setenv("MILVUS_NAMES_COLLECTION", "  my_names  ")
        monkeypatch.setenv("MILVUS_SKIP_DEDUP", "  1  ")

        config = VectorIndexerConfig.from_env()

        assert config.rag_mode == "v2"
        assert config.v2_names_collection == "my_names"
        assert config.skip_dedup is True

    def test_empty_collection_names_use_defaults(self, monkeypatch):
        """Test that empty collection names fall back to defaults."""
        monkeypatch.setenv("MILVUS_COLLECTION", "")
        monkeypatch.setenv("MILVUS_NAMES_COLLECTION", "")
        monkeypatch.setenv("MILVUS_CHUNKS_COLLECTION", "")

        config = VectorIndexerConfig.from_env()

        assert config.legacy_collection == "edu_knowledge_vectors_v1"
        assert config.v2_names_collection == "edu_knowledge_item_names_v1"
        assert config.v2_chunks_collection == "edu_knowledge_chunks_v1"

    def test_legacy_env_var_names(self, monkeypatch):
        """Test that legacy env var names (ITEM_NAME_COLLECTION, CHUNKS_COLLECTION) still work."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "legacy_names")
        monkeypatch.setenv("CHUNKS_COLLECTION", "legacy_chunks")

        config = VectorIndexerConfig.from_env()

        assert config.v2_names_collection == "legacy_names"
        assert config.v2_chunks_collection == "legacy_chunks"

    def test_new_env_vars_take_precedence_over_legacy(self, monkeypatch):
        """Test that new env var names take precedence over legacy ones."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")
        monkeypatch.setenv("MILVUS_NAMES_COLLECTION", "new_names")
        monkeypatch.setenv("ITEM_NAME_COLLECTION", "legacy_names")
        monkeypatch.setenv("MILVUS_CHUNKS_COLLECTION", "new_chunks")
        monkeypatch.setenv("CHUNKS_COLLECTION", "legacy_chunks")

        config = VectorIndexerConfig.from_env()

        assert config.v2_names_collection == "new_names"
        assert config.v2_chunks_collection == "new_chunks"

    def test_get_collection_name_legacy_mode(self, monkeypatch):
        """Test get_collection_name in legacy mode."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "legacy")
        monkeypatch.setenv("MILVUS_COLLECTION", "my_vectors")

        config = VectorIndexerConfig.from_env()
        assert config.get_collection_name() == "my_vectors"

    def test_get_collection_name_v2_mode_raises(self, monkeypatch):
        """Test that get_collection_name raises error in v2 mode."""
        monkeypatch.setenv("MILVUS_RAG_MODE", "v2")

        config = VectorIndexerConfig.from_env()

        with pytest.raises(ValueError, match="v2 mode uses two collections"):
            config.get_collection_name()

    def test_frozen_dataclass(self):
        """Test that VectorIndexerConfig is frozen (immutable)."""
        config = VectorIndexerConfig(
            rag_mode="legacy",
            legacy_collection="test",
            v2_names_collection="test_names",
            v2_chunks_collection="test_chunks",
            skip_dedup=False,
            embedding_backend="openai",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            config.rag_mode = "v2"
