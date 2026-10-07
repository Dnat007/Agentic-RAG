from app.agent.generator import AnswerGenerator
from app.agent.graph import (
    agent_graph,
    configure_agent_graph,
)
from app.cache.key_builder import CacheKeyBuilder
from app.cache.manager import CacheManager
from app.cache.memory import InMemoryCacheBackend


class FakeMessage:
    def __init__(
        self,
        content: str,
    ):
        self.content = content


class FakeChoice:
    def __init__(
        self,
        content: str,
    ):
        self.message = FakeMessage(
            content
        )


class FakeResponse:
    def __init__(
        self,
        content: str,
    ):
        self.choices = [
            FakeChoice(content)
        ]


class FakeCompletions:
    def __init__(
        self,
        answer: str,
    ):
        self.answer = answer
        self.calls = 0

    def create(
        self,
        **kwargs,
    ):
        self.calls += 1

        return FakeResponse(
            self.answer
        )


class FakeLLMClient:
    def __init__(
        self,
        answer: str,
    ):
        self.chat = type(
            "Chat",
            (),
            {
                "completions": FakeCompletions(
                    answer
                )
            },
        )()


class FakePlanner:
    def __init__(self):
        self.calls = 0

    def plan(
        self,
        state,
    ):
        self.calls += 1

        return {
            **state,
            "intent": "document",
            "next_action": "rag",
            "plan": [
                "Search documents",
                "Evaluate context",
                "Generate answer",
            ],
        }


def create_cache_manager() -> CacheManager:
    l1 = InMemoryCacheBackend(
        max_entries=100,
    )

    l2 = InMemoryCacheBackend(
        max_entries=100,
    )

    return CacheManager(
        backend=l2,
        l1_backend=l1,
        key_builder=CacheKeyBuilder(
            namespace="generation-test",
            version="v1",
        ),
        default_ttl=3600,
    )


def create_generator(
    client: FakeLLMClient,
) -> AnswerGenerator:
    return AnswerGenerator(
        client=client,
        model="test-model",
        temperature=0.0,
    )


def create_state():
    return {
        "messages": [],
        "query": "What is RAG?",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "model_version": "test-model",
        "retrieval_version": "v1",
        "access_version": "v1",
        "intent": "document",
        "next_action": "rag",
        "plan": [],
        "retrieval_attempts": 1,
        "retrieved_documents": [],
        "retrieval_scores": [],
        "context_relevant": True,
        "context_evaluation": {
            "is_relevant": True,
            "score": 1.0,
        },
        "tool_calls": 0,
        "tools_used": [
            "rag",
        ],
        "tool_results": [],
        "memory_context": [],
        "context": [
            "RAG combines retrieval with generation."
        ],
        "answer": "",
        "answer_verified": False,
        "agent_iterations": 1,
        "finished": False,
        "error": None,
    }


def test_generator_produces_answer():
    client = FakeLLMClient(
        "RAG combines retrieval and generation."
    )

    generator = create_generator(
        client
    )

    state = create_state()

    result = generator.generate(
        state
    )

    assert result["answer"] == (
        "RAG combines retrieval and generation."
    )

    assert result["answer_verified"] is False

    assert client.chat.completions.calls == 1


def test_generator_rejects_empty_query():
    client = FakeLLMClient(
        "Some answer"
    )

    generator = create_generator(
        client
    )

    state = create_state()

    state["query"] = ""

    try:
        generator.generate(state)
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "Query cannot be empty."
        )


def test_graph_generates_and_caches_answer():
    manager = create_cache_manager()

    planner = FakePlanner()

    client = FakeLLMClient(
        "RAG combines retrieval and generation."
    )

    configure_agent_graph(
        cache_manager=manager,
        planner_factory=lambda: planner,
        generator_factory=lambda: create_generator(
            client
        ),
    )

    result = agent_graph.invoke(
        create_state()
    )

    assert result["cache_hit"] is False

    assert result["answer"] == (
        "RAG combines retrieval and generation."
    )

    assert result["finished"] is True

    assert client.chat.completions.calls == 1

    cached_value = manager.get(
        query="What is RAG?",
        user_id="user-1",
        tenant_id="tenant-1",
        model="test-model",
        retrieval_version="v1",
        access_version="v1",
    )

    assert cached_value == (
        "RAG combines retrieval and generation."
    )


def test_second_request_uses_cache_and_skips_generation():
    manager = create_cache_manager()

    planner = FakePlanner()

    client = FakeLLMClient(
        "RAG combines retrieval and generation."
    )

    configure_agent_graph(
        cache_manager=manager,
        planner_factory=lambda: planner,
        generator_factory=lambda: create_generator(
            client
        ),
    )

    first_result = agent_graph.invoke(
        create_state()
    )

    assert first_result["cache_hit"] is False

    assert client.chat.completions.calls == 1

    second_result = agent_graph.invoke(
        create_state()
    )

    assert second_result["cache_hit"] is True

    assert second_result["answer"] == (
        "RAG combines retrieval and generation."
    )

    assert second_result["finished"] is True

    # LLM must NOT be called again.
    assert client.chat.completions.calls == 1


def test_cache_failure_does_not_remove_generated_answer():
    manager = create_cache_manager()

    planner = FakePlanner()

    client = FakeLLMClient(
        "Generated answer despite cache failure."
    )

    configure_agent_graph(
        cache_manager=manager,
        planner_factory=lambda: planner,
        generator_factory=lambda: create_generator(
            client
        ),
    )

    result = agent_graph.invoke(
        create_state()
    )

    assert result["answer"] == (
        "Generated answer despite cache failure."
    )

    assert result["finished"] is True

    assert client.chat.completions.calls == 1