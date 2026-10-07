from app.agent.state import AgentState


def test_default_state():
    state = AgentState(
        query="What does the document say about leave?"
    )

    assert state["query"] == (
        "What does the document say about leave?"
    )


def test_document_intent():
    state = AgentState(
        query="What does the PDF say about leave?",
        intent="document",
    )

    assert state["intent"] == "document"


def test_computation_intent():
    state = AgentState(
        query="Calculate simple interest",
        intent="computation",
    )

    assert state["intent"] == "computation"


def test_structured_data_intent():
    state = AgentState(
        query="Show employees from the database",
        intent="structured_data",
    )

    assert state["intent"] == "structured_data"


def test_multi_source_intent():
    state = AgentState(
        query="Compare the PDF with latest web information",
        intent="multi_source",
    )

    assert state["intent"] == "multi_source"


def test_planning_state():
    state = AgentState(
        query="What does the PDF say?"
    )

    state["next_action"] = "planner"

    assert state["next_action"] == "planner"


def test_retrieval_state():
    state = AgentState(
        query="Explain hybrid search"
    )

    state["retrieval_attempts"] = 1
    state["retrieved_documents"] = []
    state["context_relevant"] = True

    assert state["retrieval_attempts"] == 1
    assert state["retrieved_documents"] == []
    assert state["context_relevant"] is True


def test_tool_state():
    state = AgentState(
        query="Calculate 25 * 4"
    )

    state["tool_calls"] = 1
    state["tools_used"] = ["calculator"]
    state["tool_results"] = [100]

    assert state["tool_calls"] == 1
    assert state["tools_used"] == ["calculator"]
    assert state["tool_results"] == [100]


def test_answer_state():
    state = AgentState(
        query="What is RAG?"
    )

    state["answer"] = (
        "RAG stands for "
        "Retrieval-Augmented Generation."
    )
    state["answer_verified"] = True
    state["finished"] = True

    assert state["answer"] == (
        "RAG stands for "
        "Retrieval-Augmented Generation."
    )
    assert state["answer_verified"] is True
    assert state["finished"] is True