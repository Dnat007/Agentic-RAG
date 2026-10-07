from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.cache.base import CacheBackend
from app.cache.key_builder import CacheKeyBuilder
from app.cache.models import CacheEntry
from app.cache.semantic import SemanticCache


class CacheManager:
    """
    Production cache orchestration layer.

    Cache hierarchy:

        L1 -> In-memory exact cache
        L2 -> Redis / persistent exact cache
        L3 -> Semantic cache

    The cache layer is fail-open:
    cache failures must never break the main application.
    """

    def __init__(
        self,
        backend: CacheBackend,
        key_builder: CacheKeyBuilder,
        default_ttl: int = 3600,
        l1_backend: CacheBackend | None = None,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError(
                "default_ttl must be greater than 0."
            )

        self.backend = backend
        self.l1_backend = l1_backend
        self.key_builder = key_builder
        self.default_ttl = default_ttl
        self.semantic_cache = semantic_cache

        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._errors = 0

    def get(
        self,
        query: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        model: str | None = None,
        retrieval_version: str = "v1",
        access_version: str = "v1",
    ) -> Any | None:
        """
        Retrieve a cached value.

        Lookup order:

            L1 exact cache
            ↓
            L2 exact cache
            ↓
            L3 semantic cache
            ↓
            MISS

        Public API returns the cached value, not CacheEntry.
        """

        try:
            key = self._build_key(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                model=model,
                retrieval_version=retrieval_version,
                access_version=access_version,
            )

            # =========================================================
            # L1 - IN-MEMORY EXACT CACHE
            # =========================================================
            if self.l1_backend is not None:
                entry = self.l1_backend.get(key)

                if entry is not None:
                    if entry.is_expired():
                        self.l1_backend.delete(key)
                    else:
                        self._hits += 1
                        return entry.value

            # =========================================================
            # L2 - REDIS / PERSISTENT EXACT CACHE
            # =========================================================
            entry = self.backend.get(key)

            if entry is not None:
                if entry.is_expired():
                    self.backend.delete(key)
                else:
                    # Warm L1 from L2.
                    if self.l1_backend is not None:
                        try:
                            self.l1_backend.set(
                                key,
                                entry,
                            )
                        except Exception:
                            self._errors += 1

                    self._hits += 1

                    return entry.value

            # =========================================================
            # L3 - SEMANTIC CACHE
            # =========================================================
            if self.semantic_cache is not None:
                scope = self._build_semantic_scope(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    model=model,
                    retrieval_version=retrieval_version,
                    access_version=access_version,
                )

                semantic_entry = self.semantic_cache.get(
                    query=query,
                    scope=scope,
                )

                if semantic_entry is not None:
                    self._hits += 1
                    self._semantic_hits += 1

                    # Promote semantic hit to exact cache.
                    self._promote_to_exact_cache(
                        key=key,
                        entry=semantic_entry,
                    )

                    return semantic_entry.value

            # =========================================================
            # COMPLETE MISS
            # =========================================================
            self._misses += 1

            return None

        except Exception:
            # Cache failures must never break the application.
            self._errors += 1
            self._misses += 1

            return None

    def set(
        self,
        query: str,
        value: Any,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        model: str | None = None,
        retrieval_version: str = "v1",
        access_version: str = "v1",
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store a value in all configured cache layers.

        L1 -> In-memory
        L2 -> Redis
        L3 -> Semantic
        """

        # IMPORTANT:
        # Validate TTL OUTSIDE try/except so invalid configuration
        # raises ValueError instead of being silently swallowed.

        if ttl is None:
            ttl = self.default_ttl

        if ttl <= 0:
            raise ValueError(
                "ttl must be greater than 0."
            )

        try:
            key = self._build_key(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                model=model,
                retrieval_version=retrieval_version,
                access_version=access_version,
            )

            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=ttl)
            )

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            success = False

            # =========================================================
            # L1 - IN-MEMORY CACHE
            # =========================================================
            if self.l1_backend is not None:
                try:
                    self.l1_backend.set(
                        key,
                        entry,
                    )

                    success = True

                except Exception:
                    self._errors += 1

            # =========================================================
            # L2 - REDIS / PERSISTENT CACHE
            # =========================================================
            try:
                self.backend.set(
                    key,
                    entry,
                )

                success = True

            except Exception:
                self._errors += 1

            # =========================================================
            # L3 - SEMANTIC CACHE
            # =========================================================
            if self.semantic_cache is not None:
                try:
                    scope = self._build_semantic_scope(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        model=model,
                        retrieval_version=retrieval_version,
                        access_version=access_version,
                    )

                    self.semantic_cache.set(
                        cache_key=key,
                        query=query,
                        value=value,
                        scope=scope,
                        expires_at=expires_at,
                        metadata=metadata,
                    )

                    success = True

                except Exception:
                    self._errors += 1

            return success

        except Exception:
            self._errors += 1

            return False

    def delete(
        self,
        query: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        model: str | None = None,
        retrieval_version: str = "v1",
        access_version: str = "v1",
    ) -> bool:
        """
        Delete the query from all cache layers.
        """

        try:
            key = self._build_key(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                model=model,
                retrieval_version=retrieval_version,
                access_version=access_version,
            )

            deleted = False

            # =========================================================
            # L1
            # =========================================================
            if self.l1_backend is not None:
                try:
                    if self.l1_backend.exists(key):
                        self.l1_backend.delete(key)
                        deleted = True

                except Exception:
                    self._errors += 1

            # =========================================================
            # L2
            # =========================================================
            try:
                if self.backend.exists(key):
                    self.backend.delete(key)
                    deleted = True

            except Exception:
                self._errors += 1

            # =========================================================
            # L3
            # =========================================================
            if self.semantic_cache is not None:
                try:
                    if self.semantic_cache.delete(key):
                        deleted = True

                except Exception:
                    self._errors += 1

            return deleted

        except Exception:
            self._errors += 1

            return False

    def exists(
        self,
        query: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        model: str | None = None,
        retrieval_version: str = "v1",
        access_version: str = "v1",
    ) -> bool:
        """
        Check whether an exact cache entry exists.

        Semantic similarity is intentionally NOT used here.
        """

        try:
            key = self._build_key(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                model=model,
                retrieval_version=retrieval_version,
                access_version=access_version,
            )

            # L1
            if self.l1_backend is not None:
                if self.l1_backend.exists(key):
                    return True

            # L2
            if self.backend.exists(key):
                return True

            return False

        except Exception:
            self._errors += 1

            return False

    def health_check(self) -> bool:
        """
        Check configured cache backend health.
        """

        try:
            # L2 is the primary persistent backend.
            if not self.backend.health_check():
                return False

            # L1 health check if configured.
            if self.l1_backend is not None:
                try:
                    if not self.l1_backend.health_check():
                        return False

                except Exception:
                    return False

            return True

        except Exception:
            self._errors += 1

            return False

    def stats(self) -> dict[str, Any]:
        """
        Return aggregated cache statistics.
        """

        total_requests = self._hits + self._misses

        hit_rate = (
            self._hits / total_requests
            if total_requests > 0
            else 0.0
        )

        semantic_hit_rate = (
            self._semantic_hits / total_requests
            if total_requests > 0
            else 0.0
        )

        result: dict[str, Any] = {
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "errors": self._errors,
            "hit_rate": hit_rate,
            "semantic_hit_rate": semantic_hit_rate,
        }

        # L1 statistics.
        if self.l1_backend is not None:
            result["l1"] = self._backend_stats(
                self.l1_backend
            )

        # L2 statistics.
        result["l2"] = self._backend_stats(
            self.backend
        )

        # L3 statistics.
        if self.semantic_cache is not None:
            result["l3_semantic"] = (
                self.semantic_cache.stats()
            )

        return result

    def _promote_to_exact_cache(
        self,
        key: str,
        entry: CacheEntry,
    ) -> None:
        """
        Promote a semantic-cache hit into exact caches.

        This means:

            Semantic HIT
                 ↓
            L1 + L2 populated
                 ↓
            Next exact query becomes much faster
        """

        if entry.is_expired():
            return

        # =============================================================
        # L1
        # =============================================================
        if self.l1_backend is not None:
            try:
                l1_entry = entry.model_copy(
                    update={
                        "key": key,
                    }
                )

                self.l1_backend.set(
                    key,
                    l1_entry,
                )

            except Exception:
                self._errors += 1

        # =============================================================
        # L2
        # =============================================================
        try:
            l2_entry = entry.model_copy(
                update={
                    "key": key,
                }
            )

            self.backend.set(
                key,
                l2_entry,
            )

        except Exception:
            self._errors += 1

    def _build_key(
        self,
        *,
        query: str,
        user_id: str | None,
        tenant_id: str | None,
        model: str | None,
        retrieval_version: str,
        access_version: str,
    ) -> str:
        """
        Build deterministic exact-cache key.
        """

        return self.key_builder.build(
            query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            model=model,
            retrieval_version=retrieval_version,
            access_version=access_version,
        )

    @staticmethod
    def _build_semantic_scope(
        *,
        user_id: str | None,
        tenant_id: str | None,
        model: str | None,
        retrieval_version: str,
        access_version: str,
    ) -> str:
        """
        Build semantic-cache isolation scope.

        Semantic matches are only allowed inside the same scope.
        """

        return "|".join(
            [
                f"user={user_id or 'anonymous'}",
                f"tenant={tenant_id or 'default'}",
                f"model={model or 'default'}",
                f"retrieval={retrieval_version}",
                f"access={access_version}",
            ]
        )

    @staticmethod
    def _backend_stats(
        backend: CacheBackend,
    ) -> dict[str, Any]:
        """
        Return backend-specific statistics when available.
        """

        stats_method = getattr(
            backend,
            "stats",
            None,
        )

        if callable(stats_method):
            try:
                result = stats_method()

                if isinstance(result, dict):
                    return result

            except Exception:
                return {}

        return {}
