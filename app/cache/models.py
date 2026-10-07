from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CacheEntry(BaseModel):
    """
    Canonical representation of a cached value.

    The backend is responsible for physically storing this entry.
    CacheManager will be responsible for cache policy and key generation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    key: str = Field(
        ...,
        min_length=1,
        description="Unique cache key.",
    )

    value: Any = Field(
        ...,
        description="Cached value.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    expires_at: datetime | None = Field(
        default=None,
        description="Absolute expiration timestamp.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-sensitive cache metadata.",
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Cache key cannot be empty.")

        return value

    def is_expired(self) -> bool:
        """
        Determine whether this cache entry has expired.
        """

        if self.expires_at is None:
            return False

        return datetime.now(timezone.utc) >= self.expires_at
