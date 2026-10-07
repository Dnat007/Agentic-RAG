from __future__ import annotations

from typing import Literal

from langgraph.graph import MessagesState


IntentType = Literal[
    "document",
    "external",
    "computation",
    "structured_data",
    "multi_source",
    "direct_response",
    "unknown",
]


class AgentState(MessagesState, total=False):
    # ---------------------------------------------------------
    # Request
    # ---------------------------------------------------------
    query: str
    user_id: str | None
    tenant_id: str | None

    # ---------------------------------------------------------
    # Agent decision
    # ---------------------------------------------------------
    intent: IntentType
    next_action: str
    plan: list[str]

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------
    cache_hit: bool
    cache_layer: str | None
    cached_answer: str | None

    # Cache versioning / isolation
    model_version: str
    retrieval_version: str
    access_version: str

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    retrieval_attempts: int
    retrieved_documents: list
    retrieval_scores: list[float]
    context_relevant: bool
    context_evaluation: dict

    # ---------------------------------------------------------
    # Tools
    # ---------------------------------------------------------
    tool_calls: int
    tools_used: list[str]
    tool_results: list

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------
    memory_context: list

    # ---------------------------------------------------------
    # Context
    # ---------------------------------------------------------
    context: list

    # ---------------------------------------------------------
    # Answer
    # ---------------------------------------------------------
    answer: str
    answer_verified: bool

    # ---------------------------------------------------------
    # Agent execution
    # ---------------------------------------------------------
    agent_iterations: int
    finished: bool

    # ---------------------------------------------------------
    # Error handling
    # ---------------------------------------------------------
    error: str | None
