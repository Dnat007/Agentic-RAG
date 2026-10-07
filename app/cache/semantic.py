from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol

import numpy as np

from app.cache.models import CacheEntry


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass
class SemanticCacheItem:
    entry: CacheEntry
    embedding: np.ndarray
    scope: str


class SemanticCache:
    """
    In-memory semantic cache.

    Stores cached responses together with their query embeddings and
    returns the most semantically similar cached response within the
    same security scope.

    Security-sensitive data must never be matched across different scopes.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.92,
        max_entries: int = 10_000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0."
            )

        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")

        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        self._entries: OrderedDict[str, SemanticCacheItem] = OrderedDict()
        self._lock = RLock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def set(
        self,
        cache_key: str,
        query: str,
        value: Any,
        scope: str,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a semantic cache entry.

        Args:
            cache_key: Unique deterministic cache key.
            query: Original user query used for embedding.
            value: Cached response/value.
            scope: Security/isolation scope.
            expires_at: Absolute UTC expiration time.
            metadata: Additional cache metadata.
        """

        self._validate_key(cache_key)
        self._validate_query(query)
        self._validate_scope(scope)

        if expires_at is not None:
            if expires_at.tzinfo is None:
                raise ValueError(
                    "expires_at must be timezone-aware."
                )

            if expires_at <= datetime.now(timezone.utc):
                return

        embedding = self._embed(query)

        entry = CacheEntry(
            key=cache_key,
            value=value,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        item = SemanticCacheItem(
            entry=entry,
            embedding=embedding,
            scope=scope,
        )

        with self._lock:
            self._remove_expired_entries()

            # Replace existing entry with the same key.
            if cache_key in self._entries:
                del self._entries[cache_key]

            # LRU eviction.
            while len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

            self._entries[cache_key] = item

    def get(
        self,
        query: str,
        scope: str,
    ) -> CacheEntry | None:
        """
        Find the most semantically similar cached entry within the scope.
        """

        self._validate_query(query)
        self._validate_scope(scope)

        query_embedding = self._embed(query)

        with self._lock:
            self._remove_expired_entries()

            best_key: str | None = None
            best_item: SemanticCacheItem | None = None
            best_similarity = -1.0

            for key, item in self._entries.items():
                # Never cross security scopes.
                if item.scope != scope:
                    continue

                similarity = self._cosine_similarity(
                    query_embedding,
                    item.embedding,
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key
                    best_item = item

            # No candidate or similarity below threshold.
            if (
                best_item is None
                or best_key is None
                or best_similarity < self.similarity_threshold
            ):
                self._misses += 1
                return None

            # Pylance/type checker now knows best_key is definitely str.
            self._entries.move_to_end(best_key)

            self._hits += 1

            return best_item.entry

    def delete(self, cache_key: str) -> bool:
        """
        Delete a cache entry.

        Returns:
            True if an entry existed and was deleted.
            False if the key did not exist.
        """

        self._validate_key(cache_key)

        with self._lock:
            if cache_key not in self._entries:
                return False

            del self._entries[cache_key]
            return True

    def clear(self) -> None:
        """Remove all semantic cache entries."""

        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int | float]:
        """Return semantic cache statistics."""

        with self._lock:
            total_requests = self._hits + self._misses

            hit_rate = (
                self._hits / total_requests
                if total_requests > 0
                else 0.0
            )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "size": len(self._entries),
            }

    def _embed(self, text: str) -> np.ndarray:
        """Generate and normalize an embedding vector."""

        vector = self.embedding_provider.embed_query(text)

        embedding = np.asarray(
            vector,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise ValueError(
                "Embedding must be a one-dimensional vector."
            )

        if embedding.size == 0:
            raise ValueError(
                "Embedding cannot be empty."
            )

        norm = float(np.linalg.norm(embedding))

        if norm == 0.0:
            raise ValueError(
                "Embedding vector cannot have zero magnitude."
            )

        return embedding / norm

    @staticmethod
    def _cosine_similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two normalized vectors.
        """

        if first.shape != second.shape:
            raise ValueError(
                "Embedding dimensions must match."
            )

        return float(np.dot(first, second))

    def _remove_expired_entries(self) -> None:
        """Remove expired entries from the cache."""

        now = datetime.now(timezone.utc)

        expired_keys: list[str] = []

        for key, item in self._entries.items():
            expires_at = item.entry.expires_at

            if expires_at is not None and now >= expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            del self._entries[key]
            self._expirations += 1

    @staticmethod
    def _validate_key(cache_key: str) -> None:
        if not isinstance(cache_key, str):
            raise TypeError("cache_key must be a string.")

        if not cache_key.strip():
            raise ValueError(
                "cache_key cannot be empty."
            )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if not isinstance(scope, str):
            raise TypeError("scope must be a string.")

        if not scope.strip():
            raise ValueError(
                "scope cannot be empty."
            )
