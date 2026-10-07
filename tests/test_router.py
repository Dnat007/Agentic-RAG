import pytest

from app.agent.router import AgentRouter
from app.agent.state import AgentState


def test_router_sends_document_query_to_planner():

    router = AgentRouter()

    state = AgentState(
        query="What does the PDF say about employee leave?"
    )

    result = router.route(state)

    assert result["query"] == (
        "What does the PDF say about employee leave?"
    )

    assert result["intent"] == "unknown"
    assert result["next_action"] == "planner"


def test_router_sends_external_query_to_planner():

    router = AgentRouter()

    state = AgentState(
        query="What is today's NVIDIA stock price?"
    )

    result = router.route(state)

    assert result["query"] == (
        "What is today's NVIDIA stock price?"
    )

    assert result["intent"] == "unknown"
    assert result["next_action"] == "planner"


def test_router_sends_computation_query_to_planner():

    router = AgentRouter()

    state = AgentState(
        query="Calculate 25 * 4"
    )

    result = router.route(state)

    assert result["query"] == "Calculate 25 * 4"

    assert result["intent"] == "unknown"
    assert result["next_action"] == "planner"


def test_router_sends_multi_source_query_to_planner():

    router = AgentRouter()

    state = AgentState(
        query=(
            "According to the PDF, what technology "
            "does the company use and what is the "
            "latest version of that technology?"
        )
    )

    result = router.route(state)

    assert result["query"] == (
        "According to the PDF, what technology "
        "does the company use and what is the "
        "latest version of that technology?"
    )

    assert result["intent"] == "unknown"
    assert result["next_action"] == "planner"


def test_router_unknown_query_to_planner():

    router = AgentRouter()

    state = AgentState(
        query="Tell me something interesting."
    )

    result = router.route(state)

    assert result["intent"] == "unknown"
    assert result["next_action"] == "planner"


def test_router_empty_query():

    router = AgentRouter()

    state = AgentState(
        query=""
    )

    with pytest.raises(ValueError):
        router.route(state)