import json

import pytest

from app.agent.planner import AgentPlanner
from app.agent.state import AgentState


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [
            FakeChoice(content)
        ]


class FakeCompletions:
    def __init__(self, response):
        self.response = response

    def create(self, **kwargs):
        return self.response


class FakeChat:
    def __init__(self, response):
        self.completions = FakeCompletions(response)


class FakeClient:
    def __init__(self, response):
        self.chat = FakeChat(response)


def test_document_planning():

    response = {
        "intent": "document",
        "capabilities": ["document"],
        "plan": [
            "Search indexed documents",
            "Evaluate retrieved context",
            "Generate grounded answer",
        ],
        "next_action": "rag",
    }

    client = FakeClient(
        FakeResponse(json.dumps(response))
    )

    planner = AgentPlanner(client=client)

    state = AgentState(
        query="What does the PDF say about employee leave?"
    )

    result = planner.plan(state)

    assert result["intent"] == "document"

    assert result["plan"] == [
        "document",
        "Search indexed documents",
        "Evaluate retrieved context",
        "Generate grounded answer",
    ]

    assert result["next_action"] == "rag"


def test_computation_planning():

    response = {
        "intent": "computation",
        "capabilities": ["computation"],
        "plan": [
            "Use calculator",
            "Return verified calculation",
        ],
        "next_action": "calculator",
    }

    client = FakeClient(
        FakeResponse(json.dumps(response))
    )

    planner = AgentPlanner(client=client)

    state = AgentState(
        query=(
            "Calculate simple interest "
            "for P=10000, R=5%, T=2"
        )
    )

    result = planner.plan(state)

    assert result["intent"] == "computation"

    assert result["plan"] == [
        "computation",
        "Use calculator",
        "Return verified calculation",
    ]

    assert result["next_action"] == "calculator"


def test_external_planning():

    response = {
        "intent": "external",
        "capabilities": ["external"],
        "plan": [
            "Retrieve current external information",
            "Validate the information",
            "Generate answer",
        ],
        "next_action": "web",
    }

    client = FakeClient(
        FakeResponse(json.dumps(response))
    )

    planner = AgentPlanner(client=client)

    state = AgentState(
        query="What is today's NVIDIA stock price?"
    )

    result = planner.plan(state)

    assert result["intent"] == "external"

    assert result["plan"] == [
        "external",
        "Retrieve current external information",
        "Validate the information",
        "Generate answer",
    ]

    assert result["next_action"] == "web"


def test_multi_source_planning():

    response = {
        "intent": "multi_source",
        "capabilities": [
            "document",
            "external",
        ],
        "plan": [
            "Retrieve relevant private documents",
            "Retrieve current external information",
            "Combine evidence",
            "Generate grounded answer",
        ],
        "next_action": "multi_source",
    }

    client = FakeClient(
        FakeResponse(json.dumps(response))
    )

    planner = AgentPlanner(client=client)

    state = AgentState(
        query=(
            "According to the PDF, what technology "
            "does the company use and what is the "
            "latest version of that technology?"
        )
    )

    result = planner.plan(state)

    assert result["intent"] == "multi_source"

    assert result["plan"] == [
        "document",
        "external",
        "Retrieve relevant private documents",
        "Retrieve current external information",
        "Combine evidence",
        "Generate grounded answer",
    ]

    assert result["next_action"] == "multi_source"


def test_empty_query():

    client = FakeClient(
        FakeResponse("{}")
    )

    planner = AgentPlanner(client=client)

    state = AgentState(
        query=""
    )

    with pytest.raises(ValueError):
        planner.plan(state)