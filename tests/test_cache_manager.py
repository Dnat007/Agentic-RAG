from unittest.mock import MagicMock

import pytest

from app.cache.key_builder import CacheKeyBuilder
from app.cache.manager import CacheManager
from app.cache.memory import InMemoryCacheBackend


def create_manager() -> CacheManager:
    backend = InMemoryCacheBackend()

    key_builder = CacheKeyBuilder(
        namespace="test-agentic-rag",
    )

    return CacheManager(
        backend=backend,
        key_builder=key_builder,
        default_ttl=3600,
    )


def test_cache_manager_set_and_get():
    manager = create_manager()

    assert manager.set(
        "What is RAG?",
        "RAG combines retrieval and generation.",
        user_id="user-1",
    )

    result = manager.get(
        "what is rag?",
        user_id="user-1",
    )

    assert result == "RAG combines retrieval and generation."


def test_cache_manager_miss():
    manager = create_manager()

    result = manager.get(
        "unknown query",
        user_id="user-1",
    )

    assert result is None

    stats = manager.stats()

    assert stats["misses"] == 1


def test_cache_manager_hit_statistics():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "cached answer",
        user_id="user-1",
    )

    manager.get(
        "What is RAG?",
        user_id="user-1",
    )

    stats = manager.stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 1.0


def test_user_isolation():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "user-1 answer",
        user_id="user-1",
    )

    result = manager.get(
        "What is RAG?",
        user_id="user-2",
    )

    assert result is None


def test_tenant_isolation():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "tenant-a answer",
        tenant_id="tenant-a",
    )

    result = manager.get(
        "What is RAG?",
        tenant_id="tenant-b",
    )

    assert result is None


def test_model_version_isolation():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "model-v1 answer",
        model="model-v1",
    )

    result = manager.get(
        "What is RAG?",
        model="model-v2",
    )

    assert result is None


def test_delete():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "cached answer",
    )

    assert manager.delete(
        "What is RAG?",
    )

    assert manager.get(
        "What is RAG?",
    ) is None


def test_exists():
    manager = create_manager()

    manager.set(
        "What is RAG?",
        "cached answer",
    )

    assert manager.exists(
        "What is RAG?",
    ) is True


def test_custom_ttl_validation():
    manager = create_manager()

    with pytest.raises(ValueError):
        manager.set(
            "What is RAG?",
            "answer",
            ttl=0,
        )


def test_backend_failure_isolated():
    backend = MagicMock()

    backend.get.side_effect = RuntimeError(
        "Redis unavailable"
    )

    manager = CacheManager(
        backend=backend,
        key_builder=CacheKeyBuilder(),
    )

    result = manager.get(
        "What is RAG?",
    )

    assert result is None

    stats = manager.stats()

    assert stats["errors"] == 1
    assert stats["misses"] == 1