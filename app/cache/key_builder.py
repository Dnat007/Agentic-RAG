from __future__ import annotations

import hashlib
import json
import re
import unicodedata


class CacheKeyBuilder:
    """
    Builds deterministic, isolated cache keys.

    The generated key does not expose the original user query.
    """

    def __init__(
        self,
        namespace: str = "agentic-rag",
        version: str = "v1",
    ):
        if not namespace.strip():
            raise ValueError("Cache namespace cannot be empty.")

        if not version.strip():
            raise ValueError("Cache version cannot be empty.")

        self.namespace = namespace.strip()
        self.version = version.strip()

    def build(
        self,
        query: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        model: str | None = None,
        retrieval_version: str = "v1",
        access_version: str = "v1",
    ) -> str:
        normalized_query = self.normalize_query(query)

        payload = {
            "query": normalized_query,
            "user_id": user_id or "anonymous",
            "tenant_id": tenant_id or "default",
            "model": model or "default",
            "retrieval_version": retrieval_version,
            "access_version": access_version,
        }

        canonical_payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        digest = hashlib.sha256(
            canonical_payload.encode("utf-8")
        ).hexdigest()

        return (
            f"{self.namespace}:"
            f"{self.version}:"
            f"{digest}"
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = unicodedata.normalize(
            "NFKC",
            query,
        )

        query = query.strip().lower()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query
