from __future__ import annotations

import re
import sqlite3
from typing import Any


class SQLToolError(RuntimeError):
    """Raised when SQL execution fails or violates tool policy."""


class SQLTool:
    """
    Safe SQL execution tool.

    Initial implementation supports read-only SQLite queries.
    """

    _forbidden_keywords = {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "truncate",
        "attach",
        "detach",
        "pragma",
        "vacuum",
    }

    def __init__(
        self,
        database: str = ":memory:",
        *,
        max_rows: int = 100,
        timeout: float = 5.0,
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be greater than zero.")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self.database = database
        self.max_rows = max_rows
        self.timeout = timeout

    def execute(
        self,
        query: str,
        parameters: tuple[Any, ...] | list[Any] | None = None,
    ) -> dict[str, Any]:
        query = self._validate_query(query)

        params = tuple(parameters or ())

        try:
            connection = sqlite3.connect(
                self.database,
                timeout=self.timeout,
            )
            connection.row_factory = sqlite3.Row

            try:
                cursor = connection.execute(query, params)
                rows = cursor.fetchmany(self.max_rows)

                columns = [
                    description[0]
                    for description in cursor.description or []
                ]

                data = [
                    dict(row)
                    for row in rows
                ]

                return {
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "truncated": len(data) == self.max_rows,
                }

            finally:
                connection.close()

        except sqlite3.Error as exc:
            raise SQLToolError(
                f"SQL execution failed: {exc}"
            ) from exc

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise SQLToolError("SQL query must be a string.")

        query = query.strip()

        if not query:
            raise SQLToolError("SQL query cannot be empty.")

        if len(query) > 10_000:
            raise SQLToolError("SQL query is too long.")

        normalized = re.sub(
            r"\s+",
            " ",
            query.lower(),
        ).strip()

        if not normalized.startswith("select"):
            raise SQLToolError(
                "Only SELECT queries are allowed."
            )

        if ";" in normalized.rstrip(";"):
            raise SQLToolError(
                "Multiple SQL statements are not allowed."
            )

        for keyword in self._forbidden_keywords:
            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, normalized):
                raise SQLToolError(
                    f"SQL operation '{keyword}' is not allowed."
                )

        return query
