from datetime import datetime, timedelta, timezone

from app.cache.memory import InMemoryCacheBackend
from app.cache.models import CacheEntry


def create_entry(
    key: str,
    value: str,
    expires_at=None,
) -> CacheEntry:
    return CacheEntry(
        key=key,
        value=value,
        expires_at=expires_at,
    )


def test_set_and_get():
    cache = InMemoryCacheBackend()

    entry = create_entry("key-1", "answer")

    cache.set("key-1", entry)

    result = cache.get("key-1")

    assert result is not None
    assert result.value == "answer"


def test_cache_miss():
    cache = InMemoryCacheBackend()

    assert cache.get("missing") is None

    stats = cache.stats()

    assert stats.misses == 1
    assert stats.hits == 0


def test_cache_hit():
    cache = InMemoryCacheBackend()

    cache.set(
        "key-1",
        create_entry("key-1", "answer"),
    )

    cache.get("key-1")
    cache.get("key-1")

    stats = cache.stats()

    assert stats.hits == 2
    assert stats.hit_rate == 1.0


def test_expired_entry():
    cache = InMemoryCacheBackend()

    cache.set(
        "key-1",
        create_entry(
            "key-1",
            "answer",
            expires_at=(
                datetime.now(timezone.utc)
                - timedelta(seconds=1)
            ),
        ),
    )

    assert cache.get("key-1") is None
    assert cache.size() == 0

    stats = cache.stats()

    assert stats.expirations >= 1


def test_delete():
    cache = InMemoryCacheBackend()

    cache.set(
        "key-1",
        create_entry("key-1", "answer"),
    )

    cache.delete("key-1")

    assert cache.get("key-1") is None


def test_exists():
    cache = InMemoryCacheBackend()

    cache.set(
        "key-1",
        create_entry("key-1", "answer"),
    )

    assert cache.exists("key-1") is True
    assert cache.exists("missing") is False


def test_lru_eviction():
    cache = InMemoryCacheBackend(max_entries=2)

    cache.set(
        "key-1",
        create_entry("key-1", "answer-1"),
    )

    cache.set(
        "key-2",
        create_entry("key-2", "answer-2"),
    )

    # key-1 becomes recently used.
    cache.get("key-1")

    cache.set(
        "key-3",
        create_entry("key-3", "answer-3"),
    )

    assert cache.get("key-1") is not None
    assert cache.get("key-2") is None
    assert cache.get("key-3") is not None

    stats = cache.stats()

    assert stats.evictions == 1


def test_clear():
    cache = InMemoryCacheBackend()

    cache.set(
        "key-1",
        create_entry("key-1", "answer"),
    )

    cache.clear()

    assert cache.size() == 0


def test_health_check():
    cache = InMemoryCacheBackend()

    assert cache.health_check() is True


def test_entry_key_mismatch():
    cache = InMemoryCacheBackend()

    entry = create_entry(
        "key-2",
        "answer",
    )

    try:
        cache.set("key-1", entry)
        assert False
    except ValueError:
        assert True