from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from app.cache.base import CacheBackend
from app.cache.models import CacheEntry


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    expirations: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses

        if total == 0:
            return 0.0

        return self.hits / total


class InMemoryCacheBackend(CacheBackend):
    """
    Thread-safe bounded in-memory cache.

    Characteristics:
        - Thread-safe
        - TTL-aware
        - Lazy expiration
        - LRU eviction
        - Bounded memory
        - Runtime statistics
        - Safe/idempotent deletion
    """

    def __init__(self, max_entries: int = 10_000):
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")

        self.max_entries = max_entries

        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats()

    # ============================================================
    # GET
    # ============================================================

    def get(self, key: str) -> CacheEntry | None:
        self._validate_key(key)

        with self._lock:
            entry = self._store.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._store[key]

                self._stats.misses += 1
                self._stats.expirations += 1

                return None

            # Move recently accessed item to the end.
            self._store.move_to_end(key)

            self._stats.hits += 1

            return entry

    # ============================================================
    # SET
    # ============================================================

    def set(
        self,
        key: str,
        entry: CacheEntry,
    ) -> None:
        self._validate_key(key)

        if entry.key != key:
            raise ValueError(
                "Cache entry key must match the provided key."
            )

        with self._lock:
            # Replace existing entry.
            if key in self._store:
                del self._store[key]

            # Remove expired entries before inserting.
            self._remove_expired_entries()

            # Enforce maximum capacity.
            while len(self._store) >= self.max_entries:
                self._store.popitem(last=False)
                self._stats.evictions += 1

            self._store[key] = entry
            self._stats.sets += 1

    # ============================================================
    # DELETE
    # ============================================================

    def delete(self, key: str) -> None:
        self._validate_key(key)

        with self._lock:
            if key in self._store:
                del self._store[key]
                self._stats.deletes += 1

    # ============================================================
    # EXISTS
    # ============================================================

    def exists(self, key: str) -> bool:
        self._validate_key(key)

        with self._lock:
            entry = self._store.get(key)

            if entry is None:
                return False

            if entry.is_expired():
                del self._store[key]
                self._stats.expirations += 1
                return False

            self._store.move_to_end(key)

            return True

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> bool:
        try:
            with self._lock:
                return True
        except Exception:
            return False

    # ============================================================
    # STATS
    # ============================================================

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                expirations=self._stats.expirations,
                evictions=self._stats.evictions,
            )

    # ============================================================
    # SIZE
    # ============================================================

    def size(self) -> int:
        with self._lock:
            self._remove_expired_entries()
            return len(self._store)

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _remove_expired_entries(self) -> None:
        expired_keys = [
            key
            for key, entry in self._store.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._store[key]
            self._stats.expirations += 1

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("Cache key must be a string.")

        if not key.strip():
            raise ValueError("Cache key cannot be empty.")