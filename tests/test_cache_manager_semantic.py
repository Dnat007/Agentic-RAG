from app.cache.key_builder import CacheKeyBuilder
from app.cache.manager import CacheManager
from app.cache.memory import InMemoryCacheBackend
from app.cache.semantic import SemanticCache


class FakeEmbeddingProvider:
    vectors = {
        "what is rag": [1.0, 0.0, 0.0],
        "explain rag": [0.99, 0.01, 0.0],
        "rag explanation": [0.98, 0.02, 0.0],
        "what is sql": [0.0, 1.0, 0.0],
        "unrelated query": [0.0, 0.0, 1.0],
    }

    def embed_query(self, text: str) -> list[float]:
        """
        Simulate production embedding normalization.

        Real embedding models naturally handle punctuation,
        therefore the fake provider should normalize it too.
        """

        normalized = (
            text.strip()
            .lower()
            .replace("?", "")
            .replace("!", "")
            .replace(".", "")
            .replace(",", "")
        )

        if normalized not in self.vectors:
            raise ValueError(
                f"No fake embedding for query: {text}"
            )

        return self.vectors[normalized]


def create_semantic_manager() -> CacheManager:
    l1_backend = InMemoryCacheBackend(
        max_entries=100,
    )

    l2_backend = InMemoryCacheBackend(
        max_entries=100,
    )

    key_builder = CacheKeyBuilder(
        namespace="test-agentic-rag",
        version="v1",
    )

    semantic_cache = SemanticCache(
        embedding_provider=FakeEmbeddingProvider(),
        similarity_threshold=0.92,
        max_entries=100,
    )

    return CacheManager(
        backend=l2_backend,
        l1_backend=l1_backend,
        key_builder=key_builder,
        default_ttl=3600,
        semantic_cache=semantic_cache,
    )


def clear_exact_caches(manager: CacheManager) -> None:
    """
    Clear only L1 and L2.

    L3 semantic cache remains populated so that
    semantic lookup can be tested independently.
    """

    if manager.l1_backend is not None:
        manager.l1_backend.clear()

    manager.backend.clear()


def test_semantic_cache_hit():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="RAG combines retrieval and generation.",
        user_id="user-1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
    )

    assert result == (
        "RAG combines retrieval and generation."
    )


def test_semantic_cache_hit_statistics():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="RAG answer",
        user_id="user-1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="RAG explanation",
        user_id="user-1",
    )

    assert result == "RAG answer"

    stats = manager.stats()

    assert stats["semantic_hits"] == 1
    assert stats["hits"] == 1


def test_semantic_cache_scope_isolation():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="User 1 private answer",
        user_id="user-1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-2",
    )

    assert result is None


def test_semantic_cache_tenant_isolation():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="Tenant A answer",
        user_id="user-1",
        tenant_id="tenant-a",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
        tenant_id="tenant-b",
    )

    assert result is None


def test_semantic_cache_model_isolation():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="GPT-4.1 answer",
        user_id="user-1",
        model="gpt-4.1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
        model="gpt-4.1-mini",
    )

    assert result is None


def test_semantic_cache_retrieval_version_isolation():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="Version 1 answer",
        user_id="user-1",
        retrieval_version="v1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
        retrieval_version="v2",
    )

    assert result is None


def test_semantic_cache_access_version_isolation():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="Access version 1 answer",
        user_id="user-1",
        access_version="v1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
        access_version="v2",
    )

    assert result is None


def test_semantic_hit_promotes_to_exact_cache():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="RAG answer",
        user_id="user-1",
    )

    clear_exact_caches(manager)

    # First request should be served by L3 semantic cache.
    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
    )

    assert result == "RAG answer"

    stats_after_semantic_hit = manager.stats()

    assert stats_after_semantic_hit["semantic_hits"] == 1
    assert stats_after_semantic_hit["hits"] == 1

    # Second request should be served by exact cache
    # because the semantic hit was promoted to L1/L2.
    result = manager.get(
        query="Explain RAG",
        user_id="user-1",
    )

    assert result == "RAG answer"

    stats_after_exact_hit = manager.stats()

    assert stats_after_exact_hit["hits"] == 2
    assert stats_after_exact_hit["semantic_hits"] == 1


def test_semantic_cache_complete_miss():
    manager = create_semantic_manager()

    assert manager.set(
        query="What is RAG?",
        value="RAG answer",
        user_id="user-1",
    )

    clear_exact_caches(manager)

    result = manager.get(
        query="What is SQL?",
        user_id="user-1",
    )

    assert result is None

    stats = manager.stats()

    assert stats["semantic_hits"] == 0
    assert stats["misses"] == 1
