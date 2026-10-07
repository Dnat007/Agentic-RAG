import pytest

from app.tools.web_search import (
    SearchResult,
    WebSearchError,
    WebSearchTool,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        status_code=200,
    ):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(
            {
                "url": url,
                "kwargs": kwargs,
            }
        )
        return self.response


def test_search_returns_abstract_result():
    response = FakeResponse(
        {
            "Heading": "Python",
            "AbstractText": "Python is a programming language.",
            "AbstractURL": "https://example.com/python",
            "RelatedTopics": [],
        }
    )

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    results = tool.search("Python")

    assert len(results) == 1
    assert results[0]["title"] == "Python"
    assert results[0]["url"] == "https://example.com/python"


def test_search_returns_related_topics():
    response = FakeResponse(
        {
            "AbstractText": "",
            "AbstractURL": "",
            "RelatedTopics": [
                {
                    "Text": "Python programming language",
                    "FirstURL": "https://example.com/python",
                },
                {
                    "Text": "Python Software Foundation",
                    "FirstURL": "https://example.com/psf",
                },
            ],
        }
    )

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    results = tool.search("Python")

    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/python"
    assert results[1]["url"] == "https://example.com/psf"


def test_search_parses_nested_topics():
    response = FakeResponse(
        {
            "RelatedTopics": [
                {
                    "Name": "Nested",
                    "Topics": [
                        {
                            "Text": "Nested result",
                            "FirstURL": "https://example.com/nested",
                        }
                    ],
                }
            ]
        }
    )

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    results = tool.search("test")

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/nested"


def test_max_results_is_respected():
    response = FakeResponse(
        {
            "RelatedTopics": [
                {
                    "Text": f"Result {i}",
                    "FirstURL": f"https://example.com/{i}",
                }
                for i in range(10)
            ]
        }
    )

    session = FakeSession(response)
    tool = WebSearchTool(
        session=session,
        max_results=3,
    )

    results = tool.search("test")

    assert len(results) == 3


def test_empty_query_is_rejected():
    tool = WebSearchTool()

    with pytest.raises(WebSearchError):
        tool.search("")


def test_whitespace_query_is_rejected():
    tool = WebSearchTool()

    with pytest.raises(WebSearchError):
        tool.search("   ")


def test_non_string_query_is_rejected():
    tool = WebSearchTool()

    with pytest.raises(WebSearchError):
        tool.search(123)


def test_long_query_is_rejected():
    tool = WebSearchTool()

    with pytest.raises(WebSearchError):
        tool.search("a" * 501)


def test_timeout_must_be_positive():
    with pytest.raises(ValueError):
        WebSearchTool(timeout=0)


def test_max_results_must_be_positive():
    with pytest.raises(ValueError):
        WebSearchTool(max_results=0)


def test_search_result_to_dict():
    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="Example snippet",
    )

    assert result.to_dict() == {
        "title": "Test",
        "url": "https://example.com",
        "snippet": "Example snippet",
    }


def test_empty_payload_returns_empty_results():
    response = FakeResponse({})

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    results = tool.search("test")

    assert results == []


def test_missing_topic_url_is_ignored():
    response = FakeResponse(
        {
            "RelatedTopics": [
                {
                    "Text": "No URL result",
                },
                {
                    "Text": "Valid result",
                    "FirstURL": "https://example.com",
                },
            ]
        }
    )

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    results = tool.search("test")

    assert len(results) == 1


def test_user_agent_is_sent():
    response = FakeResponse({})

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    tool.search("test")

    assert (
        session.calls[0]["kwargs"]["headers"]["User-Agent"]
        == "Agentic-RAG/1.0"
    )


def test_query_is_normalized():
    response = FakeResponse({})

    session = FakeSession(response)
    tool = WebSearchTool(session=session)

    tool.search("   latest   AI   news   ")

    assert "latest+AI+news" in session.calls[0]["url"]