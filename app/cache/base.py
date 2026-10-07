from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.cache.models import CacheEntry


class CacheBackend(ABC):
    """
    Abstract cache backend.

    The application should depend on this interface rather than
    directly depending on Redis or an in-memory implementation.

    Implementations can include:

        - InMemoryCacheBackend
        - RedisCacheBackend
        - Future distributed cache backends
    """

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """
        Retrieve a cache entry.

        Returns:
            CacheEntry if present and valid.
            None if the key does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        key: str,
        entry: CacheEntry,
    ) -> None:
        """
        Store or replace a cache entry.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete a cache entry.

        Deleting a missing key must be safe and idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check whether a cache key exists.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all entries managed by this backend.

        Production implementations may restrict this operation
        or namespace it to avoid accidental global deletion.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify backend availability.

        Cache health failure must never imply application failure;
        callers can use this to determine whether the backend is usable.
        """
        raise NotImplementedError