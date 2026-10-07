from datetime import datetime, timedelta, timezone

import pytest

from app.cache.models import CacheEntry


def test_cache_entry_creation_and_expiration():
    entry = CacheEntry(
        key="agentic-rag:test:user_1",
        value={"answer": "cached answer"},
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )

    assert entry.key == "agentic-rag:test:user_1"
    assert entry.value["answer"] == "cached answer"
    assert entry.is_expired() is False


def test_cache_entry_without_expiration():
    entry = CacheEntry(
        key="agentic-rag:test:user_1",
        value="cached answer",
    )

    assert entry.is_expired() is False


def test_expired_cache_entry():
    entry = CacheEntry(
        key="agentic-rag:test:user_1",
        value="cached answer",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert entry.is_expired() is True


def test_empty_cache_key():
    with pytest.raises(ValueError):
        CacheEntry(
            key="   ",
            value="cached answer",
        )


def test_cache_entry_rejects_unknown_fields():
    with pytest.raises(ValueError):
        CacheEntry(
            key="test",
            value="answer",
            unknown_field="invalid",
        )
