from app.agent.graph import agent_graph, configure_agent_graph
from app.cache.manager import CacheManager
from app.cache.memory import InMemoryCacheBackend
from app.cache.key_builder import CacheKeyBuilder


class FakePlanner:
    def __init__(self):
        self.called = False

    def plan(self, state):
        self.called = True

        return {
            **state,
            "intent": "direct_response",
            "next_action": "generate",
            "plan": ["generate"],
        }


class FakeGenerator:
    def __init__(self):
        self.called = False

    def generate(self, state):
        self.called = True

        return {
            **state,
            "answer": f"Generated answer for: {state['query']}",
            "answer_verified": False,
            "finished": False,
        }


def create_planner():
    return FakePlanner()


def create_generator():
    return FakeGenerator()


def create_cache_manager():
    l1_backend = InMemoryCacheBackend(
        max_entries=100
    )

    backend = InMemoryCacheBackend(
        max_entries=100
    )

    key_builder = CacheKeyBuilder(
        namespace="test-agentic-rag",
        version="v1",
    )

    return CacheManager(
        backend=backend,
        key_builder=key_builder,
        default_ttl=3600,
        l1_backend=l1_backend,
        semantic_cache=None,
    )


def configure_test_graph(manager):
    configure_agent_graph(
        cache_manager=manager,
        planner_factory=create_planner,
        generator_factory=create_generator,
    )


def test_graph_cache_miss_continues_to_agent():
    manager = create_cache_manager()
    configure_test_graph(manager)

    state = {
        "query": "What is machine learning?",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "model_version": "model-v1",
        "retrieval_version": "retrieval-v1",
        "access_version": "access-v1",
    }

    result = agent_graph.invoke(state)

    assert result["cache_hit"] is False
    assert result["answer"] == "Generated answer for: What is machine learning?"
    assert result["finished"] is True


def test_graph_cache_hit_returns_without_planner():
    manager = create_cache_manager()

    query = "What is machine learning?"

    manager.set(
        query=query,
        value="Cached machine learning answer",
        user_id="user-1",
        tenant_id="tenant-1",
        model="model-v1",
        retrieval_version="retrieval-v1",
        access_version="access-v1",
    )

    configure_test_graph(manager)

    state = {
        "query": query,
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "model_version": "model-v1",
        "retrieval_version": "retrieval-v1",
        "access_version": "access-v1",
    }

    result = agent_graph.invoke(state)

    assert result["cache_hit"] is True
    assert result["cached_answer"] == "Cached machine learning answer"
    assert result["answer"] == "Cached machine learning answer"
    assert result["finished"] is True


def test_graph_cache_miss_does_not_return_cached_answer():
    manager = create_cache_manager()

    manager.set(
        query="What is machine learning?",
        value="Old cached answer",
        user_id="user-1",
        tenant_id="tenant-1",
        model="model-v1",
        retrieval_version="retrieval-v1",
        access_version="access-v1",
    )

    configure_test_graph(manager)

    state = {
        "query": "What is deep learning?",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "model_version": "model-v1",
        "retrieval_version": "retrieval-v1",
        "access_version": "access-v1",
    }

    result = agent_graph.invoke(state)

    assert result["cache_hit"] is False
    assert result["answer"] == "Generated answer for: What is deep learning?"
    assert result["answer"] != "Old cached answer"
    assert result["finished"] is True


def test_graph_cache_user_isolation():
    manager = create_cache_manager()

    manager.set(
        query="What is machine learning?",
        value="User 1 cached answer",
        user_id="user-1",
        tenant_id="tenant-1",
        model="model-v1",
        retrieval_version="retrieval-v1",
        access_version="access-v1",
    )

    configure_test_graph(manager)

    state = {
        "query": "What is machine learning?",
        "user_id": "user-2",
        "tenant_id": "tenant-1",
        "model_version": "model-v1",
        "retrieval_version": "retrieval-v1",
        "access_version": "access-v1",
    }

    result = agent_graph.invoke(state)

    assert result["cache_hit"] is False
    assert result["answer"] == "Generated answer for: What is machine learning?"
    assert result["answer"] != "User 1 cached answer"
    assert result["finished"] is True