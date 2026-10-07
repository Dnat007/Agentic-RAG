from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agent.generator import AnswerGenerator
from app.agent.planner import AgentPlanner
from app.agent.router import AgentRouter
from app.agent.state import AgentState
from app.cache.manager import CacheManager


# ============================================================
# Runtime Dependencies
# ============================================================

_cache_manager: CacheManager | None = None
_planner_factory: Callable[[], AgentPlanner] | None = None
_generator_factory: Callable[[], AnswerGenerator] | None = None


def configure_agent_graph(
    *,
    cache_manager: CacheManager | None = None,
    planner_factory: Callable[[], AgentPlanner] | None = None,
    generator_factory: Callable[
        [],
        AnswerGenerator,
    ]
    | None = None,
) -> None:
    """
    Configure runtime dependencies.

    Dependency injection keeps the graph testable and prevents
    tests from requiring real OpenAI/Redis infrastructure.
    """

    global _cache_manager
    global _planner_factory
    global _generator_factory

    _cache_manager = cache_manager
    _planner_factory = planner_factory
    _generator_factory = generator_factory


# ============================================================
# Cache Check
# ============================================================

def cache_check_node(
    state: AgentState,
) -> AgentState:
    """
    Check cache before planner/RAG/LLM execution.
    """

    query = state.get(
        "query",
        "",
    ).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

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
            "error": (
                f"Cache check failed: {exc}"
            ),
        }


def route_after_cache(
    state: AgentState,
) -> str:
    """
    Cache HIT -> END
    Cache MISS -> Router
    """

    if state.get(
        "cache_hit",
        False,
    ):
        return "end"

    return "router"


# ============================================================
# Router
# ============================================================

def router_node(
    state: AgentState,
) -> AgentState:
    """
    Deterministic routing before planning.
    """

    router = AgentRouter()

    return router.route(state)


# ============================================================
# Planner
# ============================================================

def planner_node(
    state: AgentState,
) -> AgentState:
    """
    Execute planner after cache miss.
    """

    if _planner_factory is not None:
        planner = _planner_factory()
    else:
        planner = AgentPlanner()

    return planner.plan(state)


def route_after_planner(
    state: AgentState,
) -> str:
    """
    Route according to planner decision.
    """

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

def rag_node(
    state: AgentState,
) -> AgentState:
    """
    RAG execution node.

    Retrieval implementation will be connected here.
    """

    attempts = state.get(
        "retrieval_attempts",
        0,
    )

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
        "tools_used": [
            *state.get(
                "tools_used",
                [],
            ),
            "rag",
        ],
    }


# ============================================================
# Context Evaluation
# ============================================================

def context_evaluation_node(
    state: AgentState,
) -> AgentState:
    """
    Evaluate retrieved context.

    For now this preserves the existing evaluation state.
    """

    return {
        **state,
        "context_relevant": state.get(
            "context_relevant",
            False,
        ),
    }


def route_after_context(
    state: AgentState,
) -> str:
    """
    Context routing.

    Relevant:
        Generate answer.

    Not relevant:
        Retry retrieval up to the bounded limit.
    """

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

    # If retry limit is exhausted, generation is still allowed
    # to explicitly state that context is insufficient.
    return "generate"


# ============================================================
# Retrieval Retry
# ============================================================

def retrieval_retry_node(
    state: AgentState,
) -> AgentState:
    """
    Increment agent iteration before retry.
    """

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
    """
    Generate the final grounded answer.
    """

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
    """
    Cache the generated answer after successful generation.

    Cache failures never break answer delivery.
    """

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
        # Cache failure must never remove a valid answer.
        return {
            **state,
            "finished": True,
            "error": (
                f"Cache set failed: {exc}"
            ),
        }

    return {
        **state,
        "finished": True,
    }


# ============================================================
# Tool Nodes
# ============================================================

def web_node(
    state: AgentState,
) -> AgentState:
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


def sql_node(
    state: AgentState,
) -> AgentState:
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
    """
    Build and compile the Agentic RAG graph.
    """

    graph = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # START -> CACHE
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "cache_check",
    )

    # --------------------------------------------------------
    # CACHE -> HIT / MISS
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "end": END,
            "router": "router",
        },
    )

    # --------------------------------------------------------
    # ROUTER -> PLANNER
    # --------------------------------------------------------

    graph.add_edge(
        "router",
        "planner",
    )

    # --------------------------------------------------------
    # PLANNER -> ACTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RAG -> CONTEXT EVALUATION
    # --------------------------------------------------------

    graph.add_edge(
        "rag",
        "context_evaluation",
    )

    # --------------------------------------------------------
    # CONTEXT -> RETRY / GENERATION
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "context_evaluation",
        route_after_context,
        {
            "retry": "retrieval_retry",
            "generate": "generation",
        },
    )

    # --------------------------------------------------------
    # RETRY -> RAG
    # --------------------------------------------------------

    graph.add_edge(
        "retrieval_retry",
        "rag",
    )

    # --------------------------------------------------------
    # GENERATION -> CACHE SET
    # --------------------------------------------------------

    graph.add_edge(
        "generation",
        "cache_set",
    )

    # --------------------------------------------------------
    # CACHE SET -> END
    # --------------------------------------------------------

    graph.add_edge(
        "cache_set",
        END,
    )

    # --------------------------------------------------------
    # Tool routes
    # --------------------------------------------------------

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
