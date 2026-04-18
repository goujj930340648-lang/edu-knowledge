"""Unified embedding service with error handling.

This module provides a unified interface for embedding generation, supporting
both OpenAI-compatible APIs and local BGE-M3 models. It uses VectorIndexerConfig
for configuration and implements proper error handling with EmbeddingError returns.

Architecture:
- EmbeddingBackend: Enum for backend types
- EmbeddingResult/EmbeddingError: Result types with error handling
- EmbeddingService: Protocol defining the service interface
- OpenAIEmbeddingService: OpenAI-compatible API implementation
- BGEM3EmbeddingService: Local BGE-M3 implementation (singleton)
- get_embedding_service: Factory function using VectorIndexerConfig
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from processor.vector_indexer.config import VectorIndexerConfig


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""

    OPENAI = "openai"
    BGE_M3 = "local_bge_m3"


@dataclass(frozen=True)
class EmbeddingResult:
    """Successful embedding generation result.

    Attributes:
        embeddings: List of embedding vectors (each is list of floats)
        backend: Which backend generated the embeddings
        processing_time_ms: Time taken in milliseconds

    Example:
        >>> result = EmbeddingResult(
        ...     embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ...     backend=EmbeddingBackend.OPENAI,
        ...     processing_time_ms=150
        ... )
    """

    embeddings: list[list[float]]
    backend: EmbeddingBackend
    processing_time_ms: float


@dataclass(frozen=True)
class EmbeddingError:
    """Embedding generation error.

    Attributes:
        error_type: Type of error (e.g., "config_error", "api_error", "model_error")
        message: Human-readable error message
        backend: Which backend was being used
        processing_time_ms: Time taken before failure

    Example:
        >>> error = EmbeddingError(
        ...     error_type="api_error",
        ...     message="OpenAI API request failed",
        ...     backend=EmbeddingBackend.OPENAI,
        ...     processing_time_ms=50
        ... )
    """

    error_type: str
    message: str
    backend: EmbeddingBackend
    processing_time_ms: float


@runtime_checkable
class EmbeddingService(Protocol):
    """Protocol for embedding services.

    All embedding services must implement these methods with proper
    error handling via EmbeddingError returns.
    """

    def embed_documents(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """Generate embeddings for documents.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult on success, EmbeddingError on failure

        Example:
            >>> service = get_embedding_service()
            >>> result = service.embed_documents(["hello", "world"])
            >>> if isinstance(result, EmbeddingResult):
            ...     print(f"Generated {len(result.embeddings)} embeddings")
            >>> else:
            ...     print(f"Error: {result.message}")
        """
        ...

    def embed_dense_only(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """Generate dense-only embeddings (for name tables).

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult on success, EmbeddingError on failure
        """
        ...

    def embed_dense_sparse(
        self, texts: list[str]
    ) -> tuple[EmbeddingResult | EmbeddingError, EmbeddingResult | EmbeddingError]:
        """Generate dense and sparse embeddings (for hybrid search).

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (dense_result, sparse_result). Each can be either
            EmbeddingResult or EmbeddingError.
        """
        ...


class OpenAIEmbeddingService:
    """OpenAI-compatible API embedding service.

    Uses EmbeddingClient from utils/client.py for actual API calls.
    Configuration comes from VectorIndexerConfig.
    """

    def __init__(self, config: VectorIndexerConfig) -> None:
        """Initialize OpenAI embedding service.

        Args:
            config: VectorIndexerConfig instance
        """
        from utils.client import EmbeddingClient

        self._config = config
        self._client = EmbeddingClient()
        self._backend = EmbeddingBackend.OPENAI

    def embed_documents(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """Generate embeddings using OpenAI-compatible API.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings or EmbeddingError on failure
        """
        start_time = time.time()

        # Validate input
        validation_error = validate_embedding_input(texts)
        if validation_error:
            return EmbeddingError(
                error_type="validation_error",
                message=validation_error,
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            embeddings = self._client.embed_documents(texts)
            return EmbeddingResult(
                embeddings=embeddings,
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return EmbeddingError(
                error_type="api_error",
                message=f"OpenAI API request failed: {e}",
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def embed_dense_only(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """OpenAI only supports dense embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with dense embeddings or EmbeddingError on failure
        """
        return self.embed_documents(texts)

    def embed_dense_sparse(
        self, texts: list[str]
    ) -> tuple[EmbeddingResult | EmbeddingError, EmbeddingResult | EmbeddingError]:
        """OpenAI does not support sparse embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (EmbeddingResult, EmbeddingError) - sparse is not supported
        """
        start_time = time.time()
        dense_result = self.embed_documents(texts)

        sparse_error = EmbeddingError(
            error_type="not_supported",
            message="OpenAI backend does not support sparse embeddings",
            backend=self._backend,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        return (dense_result, sparse_error)


class BGEM3EmbeddingService:
    """Local BGE-M3 embedding service (singleton pattern).

    Uses LocalBGEClient from utils/local_bge_client.py for actual model calls.
    Configuration comes from VectorIndexerConfig.
    """

    def __init__(self, config: VectorIndexerConfig) -> None:
        """Initialize BGE-M3 embedding service.

        Args:
            config: VectorIndexerConfig instance
        """
        from utils.local_bge_client import LocalBGEClient

        self._config = config
        self._client = LocalBGEClient()
        self._backend = EmbeddingBackend.BGE_M3

    def embed_documents(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """Generate dense-only embeddings for legacy mode.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with dense embeddings or EmbeddingError on failure
        """
        return self.embed_dense_only(texts)

    def embed_dense_only(self, texts: list[str]) -> EmbeddingResult | EmbeddingError:
        """Generate dense-only embeddings (for name tables or legacy mode).

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with dense embeddings or EmbeddingError on failure
        """
        start_time = time.time()

        # Validate input
        validation_error = validate_embedding_input(texts)
        if validation_error:
            return EmbeddingError(
                error_type="validation_error",
                message=validation_error,
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            embeddings = self._client.embed_items_dense(texts)
            return EmbeddingResult(
                embeddings=embeddings,
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return EmbeddingError(
                error_type="model_error",
                message=f"BGE-M3 dense embedding failed: {e}",
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def embed_dense_sparse(
        self, texts: list[str]
    ) -> tuple[EmbeddingResult | EmbeddingError, EmbeddingResult | EmbeddingError]:
        """Generate dense and sparse embeddings for hybrid search.

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (dense_result, sparse_result). Each can be either
            EmbeddingResult or EmbeddingError.
        """
        start_time = time.time()

        # Validate input
        validation_error = validate_embedding_input(texts)
        if validation_error:
            error = EmbeddingError(
                error_type="validation_error",
                message=validation_error,
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            return (error, error)

        try:
            dense_vecs, sparse_vecs = self._client.embed_documents_dense_sparse(texts)
            elapsed_ms = (time.time() - start_time) * 1000

            dense_result = EmbeddingResult(
                embeddings=dense_vecs,
                backend=self._backend,
                processing_time_ms=elapsed_ms,
            )

            # Convert sparse dicts to lists for EmbeddingResult
            # For sparse, we store the dicts as-is in a wrapper
            sparse_result = EmbeddingResult(
                embeddings=sparse_vecs,  # type: ignore[assignment]
                backend=self._backend,
                processing_time_ms=elapsed_ms,
            )

            return (dense_result, sparse_result)

        except Exception as e:
            error = EmbeddingError(
                error_type="model_error",
                message=f"BGE-M3 dense+sparse embedding failed: {e}",
                backend=self._backend,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            return (error, error)


# Singleton instance cache
_embedding_service_instance: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance based on VectorIndexerConfig.

    This is a factory function that:
    1. Reads VectorIndexerConfig.from_env()
    2. Returns the appropriate service instance
    3. Caches the instance for reuse (singleton pattern)

    Returns:
        EmbeddingService: Either OpenAIEmbeddingService or BGEM3EmbeddingService

    Example:
        >>> service = get_embedding_service()
        >>> result = service.embed_documents(["hello world"])
        >>> if isinstance(result, EmbeddingResult):
        ...     print(f"Generated embedding with {len(result.embeddings[0])} dimensions")
    """
    global _embedding_service_instance

    if _embedding_service_instance is not None:
        return _embedding_service_instance

    config = VectorIndexerConfig.from_env()

    if config.embedding_backend.lower() in ("local_bge_m3", "local", "offline"):
        _embedding_service_instance = BGEM3EmbeddingService(config)
    else:
        # Default to OpenAI
        _embedding_service_instance = OpenAIEmbeddingService(config)

    return _embedding_service_instance


def reset_embedding_service_cache() -> None:
    """Reset the embedding service cache.

    This is primarily used for testing to ensure fresh instances.
    """
    global _embedding_service_instance
    _embedding_service_instance = None


def validate_embedding_input(texts: list[str]) -> str | None:
    """Validate embedding input.

    Args:
        texts: List of text strings to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not isinstance(texts, list):
        return "Input must be a list of strings"

    if not texts:
        return "Input list cannot be empty"

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            return f"Input at index {i} is not a string: {type(text).__name__}"

        if not text.strip():
            return f"Input at index {i} is an empty or whitespace-only string"

    return None


__all__ = [
    "EmbeddingBackend",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingService",
    "OpenAIEmbeddingService",
    "BGEM3EmbeddingService",
    "get_embedding_service",
    "reset_embedding_service_cache",
    "validate_embedding_input",
]
