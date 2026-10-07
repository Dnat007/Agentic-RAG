import pytest

from app.cache.key_builder import CacheKeyBuilder


def test_query_normalization():
    builder = CacheKeyBuilder()

    assert (
        builder.normalize_query(
            "  What   is   RAG?  "
        )
        == "what is rag?"
    )


def test_unicode_normalization():
    builder = CacheKeyBuilder()

    result = builder.normalize_query(
        "  Café   Search  "
    )

    assert result == "café search"


def test_same_query_produces_same_key():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        user_id="user-1",
        tenant_id="tenant-1",
    )

    key_2 = builder.build(
        "  what   is rag? ",
        user_id="user-1",
        tenant_id="tenant-1",
    )

    assert key_1 == key_2


def test_different_users_produce_different_keys():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        user_id="user-1",
    )

    key_2 = builder.build(
        "What is RAG?",
        user_id="user-2",
    )

    assert key_1 != key_2


def test_different_tenants_produce_different_keys():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        tenant_id="company-a",
    )

    key_2 = builder.build(
        "What is RAG?",
        tenant_id="company-b",
    )

    assert key_1 != key_2


def test_different_models_produce_different_keys():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        model="gpt-4.1-mini",
    )

    key_2 = builder.build(
        "What is RAG?",
        model="gpt-5",
    )

    assert key_1 != key_2


def test_retrieval_version_changes_key():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        retrieval_version="v1",
    )

    key_2 = builder.build(
        "What is RAG?",
        retrieval_version="v2",
    )

    assert key_1 != key_2


def test_access_version_changes_key():
    builder = CacheKeyBuilder()

    key_1 = builder.build(
        "What is RAG?",
        access_version="v1",
    )

    key_2 = builder.build(
        "What is RAG?",
        access_version="v2",
    )

    assert key_1 != key_2


def test_key_does_not_expose_query():
    builder = CacheKeyBuilder()

    query = "What is our confidential HR policy?"

    key = builder.build(query)

    assert query.lower() not in key
    assert "confidential" not in key


def test_key_is_sha256_based():
    builder = CacheKeyBuilder()

    key = builder.build(
        "What is RAG?",
        user_id="user-1",
    )

    digest = key.split(":")[-1]

    assert len(digest) == 64
    assert all(
        character in "0123456789abcdef"
        for character in digest
    )


def test_empty_namespace():
    with pytest.raises(ValueError):
        CacheKeyBuilder(namespace=" ")


def test_empty_version():
    with pytest.raises(ValueError):
        CacheKeyBuilder(version=" ")


def test_invalid_query_type():
    builder = CacheKeyBuilder()

    with pytest.raises(TypeError):
        builder.normalize_query(None)