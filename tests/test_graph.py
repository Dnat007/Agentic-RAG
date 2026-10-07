from langchain_core.documents import Document

from app.agent.graph import agent_graph
from app.agent.state import AgentState


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

    # NOTE:
    # The current graph's rag_node is not yet wired to accept
    # an externally supplied document list.
    #
    # Therefore this test currently validates the graph route/state
    # rather than rebuilding the old global rag_tool architecture.

    result = agent_graph.invoke(state)

    assert result["agent_iterations"] >= 1
    assert result["retrieval_attempts"] >= 1
    assert "rag" in result["tools_used"]
