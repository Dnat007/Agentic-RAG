from __future__ import annotations

from dataclasses import dataclass, field


class ToolPermissionError(PermissionError):
    """Raised when a tool execution is not permitted."""


@dataclass
class ToolPermissionManager:
    """
    Controls which tools an agent is allowed to execute.

    Permissions can be defined globally and optionally
    restricted further by user or tenant.
    """

    allowed_tools: set[str] = field(default_factory=set)
    user_permissions: dict[str, set[str]] = field(default_factory=dict)
    tenant_permissions: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allowed_tools = self._normalize_tools(
            self.allowed_tools
        )

        self.user_permissions = {
            user_id: self._normalize_tools(tools)
            for user_id, tools in self.user_permissions.items()
        }

        self.tenant_permissions = {
            tenant_id: self._normalize_tools(tools)
            for tenant_id, tools in self.tenant_permissions.items()
        }

    def is_allowed(
        self,
        tool_name: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        tool_name = self._normalize_tool_name(tool_name)

        if tool_name not in self.allowed_tools:
            return False

        if tenant_id is not None:
            tenant_tools = self.tenant_permissions.get(tenant_id)

            if tenant_tools is not None:
                if tool_name not in tenant_tools:
                    return False

        if user_id is not None:
            user_tools = self.user_permissions.get(user_id)

            if user_tools is not None:
                if tool_name not in user_tools:
                    return False

        return True

    def check(
        self,
        tool_name: str,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if not self.is_allowed(
            tool_name,
            user_id=user_id,
            tenant_id=tenant_id,
        ):
            raise ToolPermissionError(
                f"Tool '{tool_name}' is not permitted."
            )

    def grant_tool(self, tool_name: str) -> None:
        tool_name = self._normalize_tool_name(tool_name)
        self.allowed_tools.add(tool_name)

    def revoke_tool(self, tool_name: str) -> None:
        tool_name = self._normalize_tool_name(tool_name)
        self.allowed_tools.discard(tool_name)

    def grant_user_tool(
        self,
        user_id: str,
        tool_name: str,
    ) -> None:
        self._validate_identity(user_id)

        tool_name = self._normalize_tool_name(tool_name)

        self.user_permissions.setdefault(
            user_id,
            set(),
        ).add(tool_name)

    def revoke_user_tool(
        self,
        user_id: str,
        tool_name: str,
    ) -> None:
        self._validate_identity(user_id)

        tool_name = self._normalize_tool_name(tool_name)

        tools = self.user_permissions.get(user_id)

        if tools is not None:
            tools.discard(tool_name)

    def grant_tenant_tool(
        self,
        tenant_id: str,
        tool_name: str,
    ) -> None:
        self._validate_identity(tenant_id)

        tool_name = self._normalize_tool_name(tool_name)

        self.tenant_permissions.setdefault(
            tenant_id,
            set(),
        ).add(tool_name)

    def revoke_tenant_tool(
        self,
        tenant_id: str,
        tool_name: str,
    ) -> None:
        self._validate_identity(tenant_id)

        tool_name = self._normalize_tool_name(tool_name)

        tools = self.tenant_permissions.get(tenant_id)

        if tools is not None:
            tools.discard(tool_name)

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:
        if not isinstance(tool_name, str):
            raise ToolPermissionError(
                "Tool name must be a string."
            )

        tool_name = tool_name.strip().lower()

        if not tool_name:
            raise ToolPermissionError(
                "Tool name cannot be empty."
            )

        return tool_name

    @staticmethod
    def _normalize_tools(tools: set[str]) -> set[str]:
        if not isinstance(tools, set):
            tools = set(tools)

        normalized = set()

        for tool in tools:
            if not isinstance(tool, str):
                raise ToolPermissionError(
                    "Tool names must be strings."
                )

            tool = tool.strip().lower()

            if tool:
                normalized.add(tool)

        return normalized

    @staticmethod
    def _validate_identity(identity: str) -> None:
        if not isinstance(identity, str):
            raise ToolPermissionError(
                "User or tenant ID must be a string."
            )

        if not identity.strip():
            raise ToolPermissionError(
                "User or tenant ID cannot be empty."
            )