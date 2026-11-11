"""Fleet delegation contracts checked through the real final tool selection."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
from model_tools import _select_tool_names
from tools import delegate_tool
from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS, _expand_parent_toolsets, _resolve_child_toolsets
from tools.registry import registry
from toolsets import TOOLSET_SESSION_PLATFORMS, resolve_toolset, session_disabled_toolsets


@pytest.fixture
def mcp_toolset(monkeypatch: pytest.MonkeyPatch) -> str:
    # Use the real registry/alias paths; per-file isolation contains registration.
    monkeypatch.setattr(registry, "_toolset_aliases", dict(registry._toolset_aliases))
    registry.register(
        name="fork_contract_mcp", toolset="mcp-fork-contract",
        schema={"name": "fork_contract_mcp", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "{}",
    )
    registry.register_toolset_alias("fork-contract-alias", "mcp-fork-contract")
    return "mcp-fork-contract"


def child_selection(parent: SimpleNamespace, requested: list[str] | None, role: str = "leaf") -> tuple[list[str], list[str], set[str]]:
    enabled, disabled = _resolve_child_toolsets(parent, requested, role)
    with delegated_child_context():
        tools = _select_tool_names(enabled, disabled, True)
    return enabled, disabled, tools


@pytest.mark.parametrize("wildcard", ["all", "*"])
def test_wildcard_parent_allows_narrowing_without_kanban(
    monkeypatch: pytest.MonkeyPatch, wildcard: str,
) -> None:
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": False})
    original = [wildcard, "kanban"]
    parent = SimpleNamespace(enabled_toolsets=original, disabled_toolsets=[])
    enabled, disabled, tools = child_selection(parent, ["terminal", "file", "web"])
    assert set(enabled) == {"terminal", "file", "web"}
    assert {"terminal", "read_file", "web_search"} <= tools
    assert not tools.intersection(DELEGATE_BLOCKED_TOOLS)
    assert not any(name.startswith("kanban_") for name in tools)
    assert "kanban" in disabled
    assert original == [wildcard, "kanban"]


@pytest.mark.parametrize("requested", [None, []], ids=["omitted", "empty"])
@pytest.mark.parametrize("mcp_name", ["mcp-fork-contract", "fork-contract-alias"])
def test_omitted_toolsets_strip_canonical_and_alias_mcp(
    monkeypatch: pytest.MonkeyPatch, mcp_toolset: str, requested: list[str] | None, mcp_name: str,
) -> None:
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": False})
    original = ["terminal", mcp_name]
    enabled, _, tools = child_selection(SimpleNamespace(enabled_toolsets=original), requested)
    assert enabled == ["terminal"]
    assert "terminal" in tools and "fork_contract_mcp" not in tools
    assert original == ["terminal", mcp_name]


@pytest.mark.parametrize("mcp_name", ["mcp-fork-contract", "fork-contract-alias"])
def test_explicit_parent_authorized_mcp_survives_false_setting(
    monkeypatch: pytest.MonkeyPatch, mcp_toolset: str, mcp_name: str,
) -> None:
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": False})
    enabled, _, tools = child_selection(SimpleNamespace(enabled_toolsets=["terminal", mcp_name]), [mcp_name])
    assert enabled == [mcp_name]
    assert "fork_contract_mcp" in tools


@pytest.mark.parametrize("wildcard", ["all", "*"])
@pytest.mark.parametrize("inherit", [False, True])
def test_wildcard_inheritance_respects_mcp_setting_and_parent_denies(
    monkeypatch: pytest.MonkeyPatch, mcp_toolset: str, wildcard: str, inherit: bool,
) -> None:
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": inherit})
    parent = SimpleNamespace(enabled_toolsets=[wildcard, "kanban"], disabled_toolsets=["web"])
    _, disabled, tools = child_selection(parent, None)
    assert ("fork_contract_mcp" in tools) is inherit
    assert not tools.intersection(resolve_toolset("web"))
    assert not tools.intersection(DELEGATE_BLOCKED_TOOLS)
    assert not any(name.startswith("kanban_") for name in tools)
    assert "web" in disabled


@pytest.mark.parametrize("wildcard", ["all", "*"])
def test_wildcard_excludes_platform_gated_sets_unless_explicit(wildcard: str) -> None:
    gated = set(TOOLSET_SESSION_PLATFORMS)
    assert gated
    expanded = _expand_parent_toolsets({wildcard})
    assert not expanded.intersection(gated)
    _, _, tools = child_selection(SimpleNamespace(enabled_toolsets=[wildcard]), None)
    gated_tools = {tool for name in gated for tool in resolve_toolset(name)}
    assert not tools.intersection(gated_tools)
    for name in gated:
        assert name in _expand_parent_toolsets({wildcard, name})
        parent = SimpleNamespace(
            enabled_toolsets=[wildcard, name], disabled_toolsets=session_disabled_toolsets([], "cli"),
        )
        _, disabled, selected = child_selection(parent, [name])
        assert name in disabled
        assert not selected.intersection(resolve_toolset(name))


def test_orchestrator_role_retains_delegation_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"inherit_mcp_toolsets": False})
    parent = SimpleNamespace(enabled_toolsets=["terminal", "kanban"], disabled_toolsets=["delegation"])
    enabled, disabled, tools = child_selection(parent, None, "orchestrator")
    assert "delegation" in enabled and "delegation" not in disabled
    assert "delegate_task" in tools
    assert not tools.intersection(DELEGATE_BLOCKED_TOOLS - {"delegate_task"})
    assert not any(name.startswith("kanban_") for name in tools)
