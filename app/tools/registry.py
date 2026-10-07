from __future__ import annotations

from typing import Any, Callable


class ToolRegistryError(RuntimeError):
    """Raised when tool registration or execution fails."""


class ToolRegistry:
    """
    Central registry for Agentic RAG tools.

    The agent interacts with tools through this registry
    instead of directly depending on individual implementations.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(
        self,
        name: str,
        tool: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        name = self._validate_name(name)

        if tool is None:
            raise ToolRegistryError(
                f"Tool '{name}' cannot be None."
            )

        if name in self._tools and not overwrite:
            raise ToolRegistryError(
                f"Tool '{name}' is already registered."
            )

        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        name = self._validate_name(name)

        if name not in self._tools:
            raise ToolRegistryError(
                f"Tool '{name}' is not registered."
            )

        del self._tools[name]

    def get(self, name: str) -> Any:
        name = self._validate_name(name)

        tool = self._tools.get(name)

        if tool is None:
            raise ToolRegistryError(
                f"Tool '{name}' is not registered."
            )

        return tool

    def has(self, name: str) -> bool:
        name = self._validate_name(name)
        return name in self._tools

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        tool = self.get(name)

        execute_method = getattr(tool, "execute", None)

        if not callable(execute_method):
            execute_method = getattr(tool, "search", None)

        if not callable(execute_method):
            execute_method = getattr(tool, "calculate", None)

        if not callable(execute_method):
            raise ToolRegistryError(
                f"Tool '{name}' does not expose "
                "execute(), search(), or calculate()."
            )

        try:
            return execute_method(*args, **kwargs)

        except Exception as exc:
            if isinstance(exc, ToolRegistryError):
                raise

            raise ToolRegistryError(
                f"Tool '{name}' execution failed: {exc}"
            ) from exc

    def clear(self) -> None:
        self._tools.clear()

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise ToolRegistryError(
                "Tool name must be a string."
            )

        name = name.strip().lower()

        if not name:
            raise ToolRegistryError(
                "Tool name cannot be empty."
            )

        return name