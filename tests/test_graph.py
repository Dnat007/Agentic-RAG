from langchain_core.documents import Document

import app.agent.graph as graph_module
from app.agent.graph import (
    agent_graph,
    configure_agent_graph,
)
from app.agent.state import AgentState


class FakePlanner:
    def plan(self, state):
        return {
            **state,
            "intent": "document",
            "next_action": "rag",
            "plan": ["retrieve relevant documents"],
        }


class FakeGenerator:
    def generate(self, state):
        return {
            **state,
            "answer": "Employees are eligible for annual leave according to company policy.",
            "answer_verified": False,
            "finished": False,
        }


def create_planner():
    return FakePlanner()


def create_generator():
    return FakeGenerator()


def test_agent_graph_rag_route():
    documents = [
        Document(
            page_content=(
                "Employees are eligible for annual leave "
                "according to company policy."
            ),
            metadata={
                "document_id": "doc_1",
                "document_name": "leave_policy.pdf",
                "user_id": "user_1",
                "department": "HR",
            },
        ),
        Document(
            page_content=(
                "Employees must submit leave requests "
                "through the HR portal."
            ),
            metadata={
                "document_id": "doc_2",
                "document_name": "hr_process.pdf",
                "user_id": "user_1",
                "department": "HR",
            },
        ),
    ]

    # Build RAG index
    graph_module.rag_tool.build(documents)

    # Inject fake planner and generator
    configure_agent_graph(
        planner_factory=create_planner,
        generator_factory=create_generator,
        rag_tool_factory=lambda: graph_module.rag_tool,
    )

    state: AgentState = {
        "messages": [],
        "query": "What does the PDF say about annual leave?",
        "user_id": "user_1",
        "tenant_id": None,

        "intent": "document",
        "next_action": "rag",
        "plan": [],

        "cache_hit": False,
        "cache_layer": None,
        "cached_answer": None,

        "model_version": "model-v1",
        "retrieval_version": "retrieval-v1",
        "access_version": "access-v1",

        "retrieval_attempts": 0,
        "retrieved_documents": [],
        "retrieval_scores": [],
        "context_relevant": False,
        "context_evaluation": {},

        "tool_calls": 0,
        "tools_used": [],
        "tool_results": [],

        "memory_context": [],
        "context": [],

        "answer": "",
        "answer_verified": False,

        "agent_iterations": 0,
        "finished": False,
        "error": None,
    }

    result = agent_graph.invoke(state)

    # Agent execution
    assert result["agent_iterations"] >= 1

    # Retrieval
    assert result["retrieval_attempts"] == 1
    assert len(result["retrieved_documents"]) > 0
    assert len(result["retrieval_scores"]) > 0
    assert len(result["context"]) > 0

    # Context evaluation
    assert result["context_relevant"] is True

    assert isinstance(
        result["context_evaluation"],
        dict,
    )

    assert result["context_evaluation"]["is_relevant"] is True

    # Generation
    assert result["answer"] != ""
