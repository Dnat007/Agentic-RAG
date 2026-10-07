from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import requests


class WebSearchError(RuntimeError):
    """Raised when web search cannot be completed."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class WebSearchTool:
    """
    Deterministic web-search tool.

    Uses DuckDuckGo Instant Answer API as the initial
    external-search implementation.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_results: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if max_results <= 0:
            raise ValueError("max_results must be greater than zero.")

        self.timeout = timeout
        self.max_results = max_results
        self.session = session or requests.Session()

    def search(self, query: str) -> list[dict[str, str]]:
        query = self._validate_query(query)

        url = (
            "https://api.duckduckgo.com/"
            f"?q={quote_plus(query)}"
            "&format=json"
            "&no_html=1"
            "&skip_disambig=1"
        )

        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Agentic-RAG/1.0",
                },
            )
            response.raise_for_status()
            payload = response.json()

        except requests.RequestException as exc:
            raise WebSearchError(
                f"Web search request failed: {exc}"
            ) from exc

        except ValueError as exc:
            raise WebSearchError(
                "Web search returned invalid JSON."
            ) from exc

        results = self._parse_results(payload)

        return [
            result.to_dict()
            for result in results[: self.max_results]
        ]

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise WebSearchError("Search query must be a string.")

        query = " ".join(query.split()).strip()

        if not query:
            raise WebSearchError("Search query cannot be empty.")

        if len(query) > 500:
            raise WebSearchError("Search query is too long.")

        return query

    @staticmethod
    def _parse_results(payload: dict[str, Any]) -> list[SearchResult]:
        results: list[SearchResult] = []

        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        heading = payload.get("Heading")

        if abstract_text and abstract_url:
            results.append(
                SearchResult(
                    title=str(heading or "Web Result"),
                    url=str(abstract_url),
                    snippet=str(abstract_text),
                )
            )

        for topic in payload.get("RelatedTopics", []):
            if not isinstance(topic, dict):
                continue

            if "Topics" in topic:
                nested_topics = topic.get("Topics", [])

                if not isinstance(nested_topics, list):
                    continue

                for nested in nested_topics:
                    result = WebSearchTool._topic_to_result(nested)

                    if result is not None:
                        results.append(result)

            else:
                result = WebSearchTool._topic_to_result(topic)

                if result is not None:
                    results.append(result)

        return results

    @staticmethod
    def _topic_to_result(
        topic: dict[str, Any],
    ) -> SearchResult | None:
        text = topic.get("Text")
        url = topic.get("FirstURL")

        if not text or not url:
            return None

        text = str(text).strip()
        url = str(url).strip()

        if not text or not url:
            return None

        title = text.split(" - ", 1)[0].strip()

        return SearchResult(
            title=title or "Web Result",
            url=url,
            snippet=text,
        )