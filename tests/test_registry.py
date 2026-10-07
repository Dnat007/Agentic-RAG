import pytest

from app.tools.registry import ToolRegistry, ToolRegistryError


class DummyTool:

    def execute(self, value):
        return value * 2


class SearchTool:

    def search(self, query):
        return [{"query": query}]


class CalculatorTool:

    def calculate(self, expression):
        return 42


def test_register_and_get():
    registry = ToolRegistry()
    tool = DummyTool()

    registry.register("dummy", tool)

    assert registry.get("dummy") is tool


def test_has_tool():
    registry = ToolRegistry()

    registry.register("dummy", DummyTool())

    assert registry.has("dummy") is True
    assert registry.has("missing") is False


def test_list_tools():
    registry = ToolRegistry()

    registry.register("web", SearchTool())
    registry.register("calculator", CalculatorTool())
    registry.register("dummy", DummyTool())

    assert registry.list_tools() == [
        "calculator",
        "dummy",
        "web",
    ]


def test_execute_method():
    registry = ToolRegistry()

    registry.register("dummy", DummyTool())

    assert registry.execute("dummy", 10) == 20


def test_search_method():
    registry = ToolRegistry()

    registry.register("web", SearchTool())

    result = registry.execute(
        "web",
        "latest AI news",
    )

    assert result == [
        {"query": "latest AI news"}
    ]


def test_calculate_method():
    registry = ToolRegistry()

    registry.register("calculator", CalculatorTool())

    assert registry.execute(
        "calculator",
        "10 + 20",
    ) == 42


def test_duplicate_registration_blocked():
    registry = ToolRegistry()

    registry.register("dummy", DummyTool())

    with pytest.raises(ToolRegistryError):
        registry.register("dummy", DummyTool())


def test_overwrite_registration():
    registry = ToolRegistry()

    first = DummyTool()
    second = DummyTool()

    registry.register("dummy", first)
    registry.register(
        "dummy",
        second,
        overwrite=True,
    )

    assert registry.get("dummy") is second


def test_missing_tool():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError):
        registry.get("missing")


def test_unregister():
    registry = ToolRegistry()

    registry.register("dummy", DummyTool())
    registry.unregister("dummy")

    assert registry.has("dummy") is False


def test_unregister_missing_tool():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError):
        registry.unregister("missing")


def test_none_tool_blocked():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError):
        registry.register("dummy", None)


def test_empty_name_blocked():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError):
        registry.register("", DummyTool())


def test_non_string_name_blocked():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError):
        registry.register(123, DummyTool())


def test_tool_without_execution_interface():
    registry = ToolRegistry()

    registry.register("invalid", object())

    with pytest.raises(ToolRegistryError):
        registry.execute("invalid")


def test_clear():
    registry = ToolRegistry()

    registry.register("dummy", DummyTool())
    registry.register("web", SearchTool())

    registry.clear()

    assert registry.list_tools() == []
