from __future__ import annotations

import json
from typing import Any

import redis
from redis import Redis

from app.cache.base import CacheBackend
from app.cache.models import CacheEntry


class RedisCacheBackend(CacheBackend):
    """
    Production Redis cache backend.

    Responsibilities:
        - Redis connection management
        - JSON serialization/deserialization
        - Native Redis TTL
        - Namespace isolation
        - Health checks
        - Cache backend abstraction

    Cache failures are intentionally isolated from the application.
    A Redis failure results in a cache miss / no-op instead of
    breaking the main RAG pipeline.
    """

    def __init__(
        self,
        redis_url: str,
        namespace: str = "agentic-rag",
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        max_connections: int = 50,
    ):
        if not redis_url.strip():
            raise ValueError("Redis URL cannot be empty.")

        if not namespace.strip():
            raise ValueError("Redis namespace cannot be empty.")

        if max_connections <= 0:
            raise ValueError(
                "max_connections must be greater than 0."
            )

        self.namespace = namespace.strip().strip(":")

        self.client: Redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
            health_check_interval=30,
        )

    # ============================================================
    # KEY
    # ============================================================

    def _build_key(self, key: str) -> str:
        self._validate_key(key)

        return f"{self.namespace}:{key}"

    # ============================================================
    # GET
    # ============================================================

    def get(self, key: str) -> CacheEntry | None:
        redis_key = self._build_key(key)

        try:
            raw_value = self.client.get(redis_key)

            if raw_value is None:
                return None

            data = json.loads(raw_value)

            entry = CacheEntry.model_validate(data)

            if entry.is_expired():
                self.delete(key)
                return None

            return entry

        except (
            redis.RedisError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):
            # Cache failure must never break the RAG pipeline.
            return None

    # ============================================================
    # SET
    # ============================================================

    def set(
        self,
        key: str,
        entry: CacheEntry,
    ) -> None:
        redis_key = self._build_key(key)

        if entry.key != key:
            raise ValueError(
                "Cache entry key must match the provided key."
            )

        try:
            payload = entry.model_dump(mode="json")

            serialized = json.dumps(
                payload,
                separators=(",", ":"),
            )

            ttl = self._calculate_ttl(entry)

            if ttl is None:
                self.client.set(
                    redis_key,
                    serialized,
                )
            else:
                self.client.set(
                    redis_key,
                    serialized,
                    ex=ttl,
                )

        except redis.RedisError:
            # Fail-open:
            # application continues even when Redis is unavailable.
            return

    # ============================================================
    # DELETE
    # ============================================================

    def delete(self, key: str) -> None:
        redis_key = self._build_key(key)

        try:
            self.client.delete(redis_key)

        except redis.RedisError:
            return

    # ============================================================
    # EXISTS
    # ============================================================

    def exists(self, key: str) -> bool:
        redis_key = self._build_key(key)

        try:
            return bool(self.client.exists(redis_key))

        except redis.RedisError:
            return False

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self) -> None:
        """
        Delete only keys belonging to this namespace.

        SCAN is used instead of KEYS to avoid blocking Redis.
        """

        pattern = f"{self.namespace}:*"

        try:
            cursor = 0

            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=500,
                )

                if keys:
                    self.client.delete(*keys)

                if cursor == 0:
                    break

        except redis.RedisError:
            return

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> bool:
        try:
            return bool(self.client.ping())

        except redis.RedisError:
            return False

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self) -> None:
        """
        Close the Redis connection pool.
        """

        try:
            self.client.close()

        except redis.RedisError:
            return

    # ============================================================
    # TTL
    # ============================================================

    def ttl(self, key: str) -> int:
        """
        Return remaining Redis TTL.

        Redis semantics:
            -1 = key exists without expiration
            -2 = key does not exist
        """

        redis_key = self._build_key(key)

        try:
            return int(self.client.ttl(redis_key))

        except redis.RedisError:
            return -2

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _calculate_ttl(
        entry: CacheEntry,
    ) -> int | None:
        if entry.expires_at is None:
            return None

        from datetime import datetime, timezone

        remaining_seconds = (
            entry.expires_at
            - datetime.now(timezone.utc)
        ).total_seconds()

        if remaining_seconds <= 0:
            return 1

        return max(1, int(remaining_seconds))

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("Cache key must be a string.")

        if not key.strip():
            raise ValueError(
                "Cache key cannot be empty."
            )