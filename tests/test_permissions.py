import pytest

from app.tools.permissions import (
    ToolPermissionError,
    ToolPermissionManager,
)


def test_allowed_tool():
    manager = ToolPermissionManager(
        allowed_tools={"calculator", "rag"}
    )

    assert manager.is_allowed("calculator") is True


def test_unregistered_tool_is_denied():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"}
    )

    assert manager.is_allowed("web") is False


def test_check_allowed_tool():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"}
    )

    manager.check("calculator")


def test_check_denied_tool():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"}
    )

    with pytest.raises(ToolPermissionError):
        manager.check("web")


def test_tool_names_are_normalized():
    manager = ToolPermissionManager(
        allowed_tools={" Calculator "}
    )

    assert manager.is_allowed("calculator") is True


def test_user_permission():
    manager = ToolPermissionManager(
        allowed_tools={"calculator", "web"},
        user_permissions={
            "user1": {"calculator"},
        },
    )

    assert manager.is_allowed(
        "calculator",
        user_id="user1",
    ) is True

    assert manager.is_allowed(
        "web",
        user_id="user1",
    ) is False


def test_unknown_user_uses_global_permission():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"},
        user_permissions={
            "user1": {"calculator"},
        },
    )

    assert manager.is_allowed(
        "calculator",
        user_id="user2",
    ) is True


def test_tenant_permission():
    manager = ToolPermissionManager(
        allowed_tools={"calculator", "sql"},
        tenant_permissions={
            "tenant1": {"calculator"},
        },
    )

    assert manager.is_allowed(
        "calculator",
        tenant_id="tenant1",
    ) is True

    assert manager.is_allowed(
        "sql",
        tenant_id="tenant1",
    ) is False


def test_user_and_tenant_permissions():
    manager = ToolPermissionManager(
        allowed_tools={"calculator", "web", "sql"},
        user_permissions={
            "user1": {"calculator", "web"},
        },
        tenant_permissions={
            "tenant1": {"calculator", "web"},
        },
    )

    assert manager.is_allowed(
        "web",
        user_id="user1",
        tenant_id="tenant1",
    ) is True

    assert manager.is_allowed(
        "sql",
        user_id="user1",
        tenant_id="tenant1",
    ) is False


def test_tenant_can_deny_user_permission():
    manager = ToolPermissionManager(
        allowed_tools={"calculator", "web"},
        user_permissions={
            "user1": {"web"},
        },
        tenant_permissions={
            "tenant1": {"calculator"},
        },
    )

    assert manager.is_allowed(
        "web",
        user_id="user1",
        tenant_id="tenant1",
    ) is False


def test_grant_tool():
    manager = ToolPermissionManager()

    manager.grant_tool("calculator")

    assert manager.is_allowed("calculator") is True


def test_revoke_tool():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"}
    )

    manager.revoke_tool("calculator")

    assert manager.is_allowed("calculator") is False


def test_grant_user_tool():
    manager = ToolPermissionManager(
        allowed_tools={"web"}
    )

    manager.grant_user_tool(
        "user1",
        "web",
    )

    assert manager.is_allowed(
        "web",
        user_id="user1",
    ) is True


def test_revoke_user_tool():
    manager = ToolPermissionManager(
        allowed_tools={"web"},
        user_permissions={
            "user1": {"web"},
        },
    )

    manager.revoke_user_tool(
        "user1",
        "web",
    )

    assert manager.is_allowed(
        "web",
        user_id="user1",
    ) is False


def test_grant_tenant_tool():
    manager = ToolPermissionManager(
        allowed_tools={"sql"}
    )

    manager.grant_tenant_tool(
        "tenant1",
        "sql",
    )

    assert manager.is_allowed(
        "sql",
        tenant_id="tenant1",
    ) is True


def test_revoke_tenant_tool():
    manager = ToolPermissionManager(
        allowed_tools={"sql"},
        tenant_permissions={
            "tenant1": {"sql"},
        },
    )

    manager.revoke_tenant_tool(
        "tenant1",
        "sql",
    )

    assert manager.is_allowed(
        "sql",
        tenant_id="tenant1",
    ) is False


def test_invalid_tool_name():
    manager = ToolPermissionManager()

    with pytest.raises(ToolPermissionError):
        manager.is_allowed("")


def test_invalid_user_id():
    manager = ToolPermissionManager(
        allowed_tools={"calculator"}
    )

    with pytest.raises(ToolPermissionError):
        manager.grant_user_tool(
            "",
            "calculator",
        )


def test_invalid_tenant_id():
    manager = ToolPermissionManager(
        allowed_tools={"sql"}
    )

    with pytest.raises(ToolPermissionError):
        manager.grant_tenant_tool(
            "",
            "sql",
        )