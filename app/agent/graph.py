from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.agent.generator import AnswerGenerator
from app.agent.planner import AgentPlanner
from app.agent.router import AgentRouter
from app.agent.state import AgentState
from app.cache.manager import CacheManager
from app.retrieval.evaluator import ContextEvaluator
from app.tools.rag import RAGTool


# ============================================================
# Runtime Dependencies
# ============================================================

_cache_manager: CacheManager | None = None
_planner_factory: Callable[[], AgentPlanner] | None = None
_generator_factory: Callable[[], AnswerGenerator] | None = None
_rag_tool_factory: Callable[[], RAGTool] | None = None

# Default RAG tool.
# Tests can replace this through configure_agent_graph().
rag_tool = RAGTool()


def configure_agent_graph(
    *,
    cache_manager: CacheManager | None = None,
    planner_factory: Callable[[], AgentPlanner] | None = None,
    generator_factory: Callable[[], AnswerGenerator] | None = None,
    rag_tool_factory: Callable[[], RAGTool] | None = None,
) -> None:
    """
    Configure runtime dependencies.

    Dependency injection keeps the graph testable and prevents
    tests from requiring real OpenAI/Redis infrastructure.
    """

    global _cache_manager
    global _planner_factory
    global _generator_factory
    global _rag_tool_factory

    _cache_manager = cache_manager
    _planner_factory = planner_factory
    _generator_factory = generator_factory
    _rag_tool_factory = rag_tool_factory


# ============================================================
# Cache Check
# ============================================================

def cache_check_node(state: AgentState) -> AgentState:
    query = state.get("query", "").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if _cache_manager is None:
        return {
            **state,
            "cache_hit": False,
            "cache_layer": None,
            "cached_answer": None,
        }

    try:
        cached_value = _cache_manager.get(
            query=query,
            user_id=state.get("user_id"),
            tenant_id=state.get("tenant_id"),
            model=state.get("model_version", "default"),
            retrieval_version=state.get(
                "retrieval_version",
                "v1",
            ),
            access_version=state.get(
                "access_version",
                "v1",
            ),
        )

        if cached_value is None:
            return {
                **state,
                "cache_hit": False,
                "cache_layer": None,
                "cached_answer": None,
            }

        answer = str(cached_value)

        return {
            **state,
            "cache_hit": True,
            "cache_layer": "cache",
            "cached_answer": answer,
            "answer": answer,
            "answer_verified": True,
            "finished": True,
        }

    except Exception as exc:
        return {
            **state,
            "cache_hit": False,
            "cache_layer": None,
            "cached_answer": None,
            "error": f"Cache check failed: {exc}",
        }


def route_after_cache(state: AgentState) -> str:
    if state.get("cache_hit", False):
        return "end"

    return "router"


# ============================================================
# Router
# ============================================================

def router_node(state: AgentState) -> AgentState:
    router = AgentRouter()
    return router.route(state)


# ============================================================
# Planner
# ============================================================

def planner_node(state: AgentState) -> AgentState:
    if _planner_factory is not None:
        planner = _planner_factory()
    else:
        planner = AgentPlanner()

    return planner.plan(state)


def route_after_planner(state: AgentState) -> str:
    next_action = state.get(
        "next_action",
        "rag",
    )

    routes = {
        "rag": "rag",
        "web": "web",
        "calculator": "calculator",
        "sql": "sql",
        "multi_source": "multi_source",
        "direct_response": "direct_response",
        "planner": "rag",
    }

    return routes.get(
        next_action,
        "rag",
    )


# ============================================================
# RAG
# ============================================================

def _get_rag_tool() -> RAGTool:
    if _rag_tool_factory is not None:
        return _rag_tool_factory()

    return rag_tool


def _build_context(
    results: list[tuple[Document, float]],
) -> list[Document]:
    return [document for document, _ in results]


def _build_scores(
    results: list[tuple[Document, float]],
) -> list[float]:
    return [float(score) for _, score in results]


def rag_node(state: AgentState) -> AgentState:
    """
    Execute the real RAG retrieval pipeline.

    RAGTool internally performs:

        Query
            ↓
        Metadata / ACL filtering
            ↓
        BM25 + Vector Search
            ↓
        RRF
            ↓
        Cross Encoder Reranking
            ↓
        Final Documents
    """

    query = state.get("query", "").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    attempts = state.get(
        "retrieval_attempts",
        0,
    )

    rag = _get_rag_tool()

    if not rag.is_built:
        raise RuntimeError(
            "RAG tool has not been built. "
            "Build the RAG index before invoking the agent graph."
        )

    results = rag.search(
        query=query,
        top_k=None,
        user_id=state.get("user_id"),
        allowed_departments=state.get(
            "allowed_departments"
        ),
        allowed_document_ids=state.get(
            "allowed_document_ids"
        ),
    )

    retrieved_documents = _build_context(results)
    retrieval_scores = _build_scores(results)

    context = list(retrieved_documents)

    tools_used = [
        *state.get("tools_used", []),
    ]

    if "rag" not in tools_used:
        tools_used.append("rag")

    return {
        **state,
        "retrieval_attempts": attempts + 1,
        "agent_iterations": (
            state.get(
                "agent_iterations",
                0,
            )
            + 1
        ),
        "retrieved_documents": retrieved_documents,
        "retrieval_scores": retrieval_scores,
        "context": context,
        "tools_used": tools_used,
    }


# ============================================================
# Context Evaluation
# ============================================================

def context_evaluation_node(
    state: AgentState,
) -> AgentState:
    query = state.get(
        "query",
        "",
    ).strip()

    documents = state.get(
        "retrieved_documents",
        [],
    )

    scores = state.get(
        "retrieval_scores",
        [],
    )

    results = list(
        zip(
            documents,
            scores,
        )
    )

    evaluator = ContextEvaluator(
        min_documents=1,
        min_score=-10.0,
    )

    evaluation = evaluator.evaluate(
        query=query,
        results=results,
    )

    return {
        **state,
        "context_relevant": bool(
            evaluation["is_relevant"]
        ),
        "context_evaluation": evaluation,
    }


def route_after_context(state: AgentState) -> str:
    if state.get(
        "context_relevant",
        False,
    ):
        return "generate"

    attempts = state.get(
        "retrieval_attempts",
        0,
    )

    if attempts < 2:
        return "retry"

    return "generate"


# ============================================================
# Retrieval Retry
# ============================================================

def retrieval_retry_node(
    state: AgentState,
) -> AgentState:
    return {
        **state,
        "agent_iterations": (
            state.get(
                "agent_iterations",
                0,
            )
            + 1
        ),
    }


# ============================================================
# Answer Generation
# ============================================================

def generation_node(
    state: AgentState,
) -> AgentState:
    if _generator_factory is not None:
        generator = _generator_factory()
    else:
        generator = AnswerGenerator()

    return generator.generate(state)


# ============================================================
# Cache SET
# ============================================================

def cache_set_node(
    state: AgentState,
) -> AgentState:

    answer = state.get(
        "answer",
        "",
    ).strip()

    if not answer:
        return {
            **state,
            "finished": True,
        }

    if _cache_manager is None:
        return {
            **state,
            "finished": True,
        }

    try:
        _cache_manager.set(
            query=state.get(
                "query",
                "",
            ),
            value=answer,
            user_id=state.get(
                "user_id"
            ),
            tenant_id=state.get(
                "tenant_id"
            ),
            model=state.get(
                "model_version",
                "default",
            ),
            retrieval_version=state.get(
                "retrieval_version",
                "v1",
            ),
            access_version=state.get(
                "access_version",
                "v1",
            ),
        )

    except Exception as exc:
        return {
            **state,
            "finished": True,
            "error": f"Cache set failed: {exc}",
        }

    return {
        **state,
        "finished": True,
    }


# ============================================================
# Tool Nodes
# ============================================================

def web_node(state: AgentState) -> AgentState:
    return {
        **state,
        "tools_used": [
            *state.get(
                "tools_used",
                [],
            ),
            "web",
        ],
        "tool_calls": (
            state.get(
                "tool_calls",
                0,
            )
            + 1
        ),
    }


def calculator_node(
    state: AgentState,
) -> AgentState:
    return {
        **state,
        "tools_used": [
            *state.get(
                "tools_used",
                [],
            ),
            "calculator",
        ],
        "tool_calls": (
            state.get(
                "tool_calls",
                0,
            )
            + 1
        ),
    }


def sql_node(state: AgentState) -> AgentState:
    return {
        **state,
        "tools_used": [
            *state.get(
                "tools_used",
                [],
            ),
            "sql",
        ],
        "tool_calls": (
            state.get(
                "tool_calls",
                0,
            )
            + 1
        ),
    }


def multi_source_node(
    state: AgentState,
) -> AgentState:
    return {
        **state,
        "tools_used": [
            *state.get(
                "tools_used",
                [],
            ),
            "multi_source",
        ],
        "tool_calls": (
            state.get(
                "tool_calls",
                0,
            )
            + 1
        ),
    }


def direct_response_node(
    state: AgentState,
) -> AgentState:
    return {
        **state,
        "finished": True,
    }


# ============================================================
# Graph Construction
# ============================================================

def build_agent_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node(
        "cache_check",
        cache_check_node,
    )

    graph.add_node(
        "router",
        router_node,
    )

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "rag",
        rag_node,
    )

    graph.add_node(
        "context_evaluation",
        context_evaluation_node,
    )

    graph.add_node(
        "retrieval_retry",
        retrieval_retry_node,
    )

    graph.add_node(
        "generation",
        generation_node,
    )

    graph.add_node(
        "cache_set",
        cache_set_node,
    )

    graph.add_node(
        "web",
        web_node,
    )

    graph.add_node(
        "calculator",
        calculator_node,
    )

    graph.add_node(
        "sql",
        sql_node,
    )

    graph.add_node(
        "multi_source",
        multi_source_node,
    )

    graph.add_node(
        "direct_response",
        direct_response_node,
    )

    # START -> CACHE
    graph.add_edge(
        START,
        "cache_check",
    )

    # CACHE -> HIT / MISS
    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "end": END,
            "router": "router",
        },
    )

    # ROUTER -> PLANNER
    graph.add_edge(
        "router",
        "planner",
    )

    # PLANNER -> ACTION
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "rag": "rag",
            "web": "web",
            "calculator": "calculator",
            "sql": "sql",
            "multi_source": "multi_source",
            "direct_response": "direct_response",
        },
    )

    # RAG -> CONTEXT EVALUATION
    graph.add_edge(
        "rag",
        "context_evaluation",
    )

    # CONTEXT -> RETRY / GENERATION
    graph.add_conditional_edges(
        "context_evaluation",
        route_after_context,
        {
            "retry": "retrieval_retry",
            "generate": "generation",
        },
    )

    # RETRY -> RAG
    graph.add_edge(
        "retrieval_retry",
        "rag",
    )

    # GENERATION -> CACHE
    graph.add_edge(
        "generation",
        "cache_set",
    )

    # CACHE -> END
    graph.add_edge(
        "cache_set",
        END,
    )

    # Tool routes
    graph.add_edge(
        "web",
        END,
    )

    graph.add_edge(
        "calculator",
        END,
    )

    graph.add_edge(
        "sql",
        END,
    )

    graph.add_edge(
        "multi_source",
        END,
    )

    graph.add_edge(
        "direct_response",
        END,
    )

    return graph.compile()


agent_graph = build_agent_graph()
