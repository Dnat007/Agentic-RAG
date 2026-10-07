from datetime import datetime, timedelta, timezone

import pytest

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
        normalized = text.strip().lower()

        if normalized not in self.vectors:
            raise ValueError(f"No fake embedding for query: {text}")

        return self.vectors[normalized]


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


@pytest.fixture
def semantic_cache(embedding_provider):
    return SemanticCache(
        embedding_provider=embedding_provider,
        similarity_threshold=0.92,
        max_entries=100,
    )


def test_set_and_get_exact_query(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG stands for Retrieval-Augmented Generation.",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="what is rag",
        scope="user-1",
    )

    assert result is not None
    assert result.value == "RAG stands for Retrieval-Augmented Generation."


def test_semantic_similarity_hit(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="rag explanation",
        scope="user-1",
    )

    assert result is not None
    assert result.value == "RAG answer"


def test_scope_isolation(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="User 1 answer",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="what is rag",
        scope="user-2",
    )

    assert result is None


def test_semantic_threshold(embedding_provider):
    semantic_cache = SemanticCache(
        embedding_provider=embedding_provider,
        similarity_threshold=0.999,
        max_entries=100,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="what is sql",
        scope="user-1",
    )

    assert result is None


def test_unrelated_query_is_miss(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="unrelated query",
        scope="user-1",
    )

    assert result is None


def test_delete(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    deleted = semantic_cache.delete("key-1")

    assert deleted is True

    result = semantic_cache.get(
        query="what is rag",
        scope="user-1",
    )

    assert result is None


def test_clear(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    semantic_cache.set(
        cache_key="key-2",
        query="what is sql",
        value="SQL answer",
        scope="user-1",
        expires_at=expires_at,
    )

    semantic_cache.clear()

    result_1 = semantic_cache.get(
        query="what is rag",
        scope="user-1",
    )

    result_2 = semantic_cache.get(
        query="what is sql",
        scope="user-1",
    )

    assert result_1 is None
    assert result_2 is None


def test_expired_entry(semantic_cache):
    expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    semantic_cache.set(
        cache_key="expired-key",
        query="what is rag",
        value="Expired answer",
        scope="user-1",
        expires_at=expires_at,
    )

    result = semantic_cache.get(
        query="what is rag",
        scope="user-1",
    )

    assert result is None

    stats = semantic_cache.stats()

    assert stats["size"] == 0


def test_stats(semantic_cache):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    semantic_cache.get(
        query="what is rag",
        scope="user-1",
    )

    semantic_cache.get(
        query="unrelated query",
        scope="user-1",
    )

    stats = semantic_cache.stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_lru_eviction(embedding_provider):
    semantic_cache = SemanticCache(
        embedding_provider=embedding_provider,
        similarity_threshold=0.92,
        max_entries=2,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    semantic_cache.set(
        cache_key="key-1",
        query="what is rag",
        value="RAG answer",
        scope="user-1",
        expires_at=expires_at,
    )

    semantic_cache.set(
        cache_key="key-2",
        query="what is sql",
        value="SQL answer",
        scope="user-1",
        expires_at=expires_at,
    )

    semantic_cache.set(
        cache_key="key-3",
        query="unrelated query",
        value="Other answer",
        scope="user-1",
        expires_at=expires_at,
    )

    stats = semantic_cache.stats()

    assert stats["size"] == 2
    assert stats["evictions"] == 1