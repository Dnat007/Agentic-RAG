import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import redis

from app.cache.models import CacheEntry
from app.cache.redis import RedisCacheBackend


def create_backend() -> RedisCacheBackend:
    backend = RedisCacheBackend(
        redis_url="redis://localhost:6379/0",
        namespace="test-agentic-rag",
    )

    backend.client = MagicMock()

    return backend


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


def test_key_namespace():
    backend = create_backend()

    assert (
        backend._build_key("query-1")
        == "test-agentic-rag:query-1"
    )


def test_set_serializes_entry():
    backend = create_backend()

    entry = create_entry(
        "query-1",
        "cached answer",
    )

    backend.set("query-1", entry)

    backend.client.set.assert_called_once()

    args, kwargs = backend.client.set.call_args

    assert args[0] == "test-agentic-rag:query-1"

    payload = json.loads(args[1])

    assert payload["key"] == "query-1"
    assert payload["value"] == "cached answer"


def test_get_deserializes_entry():
    backend = create_backend()

    entry = create_entry(
        "query-1",
        "cached answer",
    )

    backend.client.get.return_value = (
        entry.model_dump_json()
    )

    result = backend.get("query-1")

    assert result is not None
    assert result.key == "query-1"
    assert result.value == "cached answer"


def test_get_cache_miss():
    backend = create_backend()

    backend.client.get.return_value = None

    result = backend.get("missing")

    assert result is None


def test_delete():
    backend = create_backend()

    backend.delete("query-1")

    backend.client.delete.assert_called_once_with(
        "test-agentic-rag:query-1"
    )


def test_exists():
    backend = create_backend()

    backend.client.exists.return_value = 1

    assert backend.exists("query-1") is True


def test_ttl():
    backend = create_backend()

    backend.client.ttl.return_value = 120

    assert backend.ttl("query-1") == 120


def test_health_check():
    backend = create_backend()

    backend.client.ping.return_value = True

    assert backend.health_check() is True


def test_redis_failure_is_fail_open():
    backend = create_backend()

    backend.client.get.side_effect = redis.RedisError(
        "Redis unavailable"
    )

    assert backend.get("query-1") is None


def test_expired_entry_is_not_returned():
    backend = create_backend()

    entry = create_entry(
        "query-1",
        "expired answer",
        expires_at=(
            datetime.now(timezone.utc)
            - timedelta(seconds=10)
        ),
    )

    backend.client.get.return_value = (
        entry.model_dump_json()
    )

    assert backend.get("query-1") is None