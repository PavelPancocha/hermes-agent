"""Bundled REST Mem0 discovery and owning-profile behavior contracts."""
from __future__ import annotations

import builtins
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import httpx
import pytest

from agent.memory_manager import MemoryManager
from agent.secret_scope import (
    load_env_file,
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory import load_memory_provider


@contextmanager
def _scope(home: Path, secrets: dict[str, str] | None = None) -> Iterator[None]:
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(load_env_file(home / ".env") if secrets is None else secrets, profile_home=str(home))
    multiplex_token = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(multiplex_token)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root"))
    for name, value in {"MEM0_HOST": "https://decoy.invalid/api", "MEM0_API_KEY": "synthetic-root-decoy", "MEM0_USER_ID": "root-decoy", "MEM0_AGENT_ID": "root-decoy"}.items():
        monkeypatch.setenv(name, value)
    homes = (tmp_path / "profiles" / "alpha", tmp_path / "profiles" / "beta")
    for home in homes:
        home.mkdir(parents=True)
        (home / "mem0.json").write_text(json.dumps({"mode": "self_hosted_http", "host": f"https://{home.name}.invalid/api", "user_id": f"user:{home.name}", "agent_id": f"agent:{home.name}", "unknown": "keep"}) + "\n")
        (home / ".env").write_text(f"MEM0_API_KEY=synthetic-{home.name}-key\n")
        (home / "config.yaml").write_text("memory:\n  provider: mem0\n")
    real_import = builtins.__import__

    def deny_sdk(name: str, *args, **kwargs):
        if name == "mem0" or name.startswith("mem0."):
            pytest.fail("REST provider must never import the Mem0 SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_sdk)
    return homes


def test_bundled_mem0_loader_returns_distinct_instances() -> None:
    first = load_memory_provider("mem0")
    second = load_memory_provider("mem0")
    assert first is not None, "bundled REST mem0 provider must be discoverable"
    assert second is not None
    assert first is not second


def test_two_managers_keep_own_http_identity_across_tools_and_background(homes, monkeypatch) -> None:
    records: list[tuple[httpx.Request, dict]] = []
    record_lock = threading.Lock()

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        with record_lock:
            records.append((request, body))
        payload = {"results": [{"id": "synthetic-memory", "memory": f"recall:{request.url.host}"}]} if request.url.path.endswith("/search") else {"event_id": "synthetic-event"}
        return httpx.Response(200, json=payload)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(respond)}))
    managers: list[MemoryManager] = []
    before = {home: ((home / "mem0.json").read_bytes(), (home / ".env").read_bytes()) for home in homes}
    try:
        for home in homes:
            with _scope(home):
                provider = load_memory_provider("mem0")
                assert provider is not None
                manager = MemoryManager()
                manager.add_provider(provider)
                manager.initialize_all(session_id=f"session:{home.name}", hermes_home=str(home), platform="api_server", user_id="gateway-decoy")
                assert provider._backend is not None
                managers.append(manager)
        for home, manager in zip(homes, managers):
            with _scope(home):
                assert "error" not in json.loads(manager.handle_tool_call("mem0_add", {"content": f"remember:{home.name}", "user_id": "attacker", "agent_id": "attacker", "filters": {"user_id": "attacker"}}, hermes_home="/attacker", user_id="attacker", agent_id="attacker"))
                assert "recall:" in manager.prefetch_all(f"question:{home.name}")
                manager.sync_all(f"user turn:{home.name}", f"assistant turn:{home.name}")
                assert manager.flush_pending(timeout=10)
        for home, manager in zip(homes, managers):
            with _scope(home):
                provider = manager.get_provider("mem0")
                provider.shutdown()
                assert before[home] == ((home / "mem0.json").read_bytes(), (home / ".env").read_bytes())
        assert len(records) == 6
        for request, body in records:
            name = request.url.host.split(".")[0]
            assert name in {"alpha", "beta"}
            assert request.url.path.startswith("/api/")
            assert request.headers["X-API-Key"] == f"synthetic-{name}-key"
            if request.url.path.endswith("/search"):
                assert body["filters"] == {"user_id": f"user:{name}"}
            else:
                assert body["user_id"] == f"user:{name}"
                assert body["agent_id"] == f"agent:{name}"
                assert body["metadata"] == {"channel": "api_server"}
        assert sum(body.get("infer") is True for _, body in records) == 2
    finally:
        for home, manager in zip(homes, managers):
            with _scope(home):
                manager.shutdown_all()


@pytest.mark.parametrize("missing", ["api_key", "host"])
def test_secondary_profile_cannot_borrow_root_credentials(homes, monkeypatch, missing) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    home = homes[1]
    cfg = json.loads((home / "mem0.json").read_text())
    cfg.pop(missing, None)
    (home / "mem0.json").write_text(json.dumps(cfg))
    secrets = {} if missing == "api_key" else load_env_file(home / ".env")
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: pytest.fail("invalid profile must not construct a client"))
    with _scope(home, secrets):
        provider = Mem0MemoryProvider()
        assert not provider.is_available()
        with pytest.raises(ValueError, match=missing):
            provider.initialize("synthetic", hermes_home=str(home))


def test_initialization_refuses_wrong_scope_and_instance_reuse(homes, monkeypatch) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: pytest.fail("mismatched scope must not construct client"))
    provider = Mem0MemoryProvider()
    with _scope(homes[0]):
        with pytest.raises(ValueError, match="owning profile scope"):
            provider.initialize("bad-home", hermes_home=str(homes[1]))
    home_token = set_hermes_home_override(homes[0])
    secret_token = set_secret_scope(load_env_file(homes[1] / ".env"), profile_home=str(homes[1]))
    try:
        with pytest.raises(ValueError, match="owning profile scope"):
            provider.initialize("bad-secrets", hermes_home=str(homes[0]))
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: None)
    with _scope(homes[0]):
        provider.initialize("first", hermes_home=str(homes[0]))
        with pytest.raises(ValueError, match="already initialized"):
            provider.initialize("second", hermes_home=str(homes[0]))


def test_tool_kwargs_do_not_rebind_provider(homes, monkeypatch) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    calls: list[dict] = []

    class Backend:
        def add(self, messages, **kwargs):
            calls.append(kwargs)
            return {}

    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: Backend())
    home = homes[0]
    with _scope(home):
        provider = Mem0MemoryProvider()
        provider.initialize("synthetic", hermes_home=str(home))
    result = provider.handle_tool_call("mem0_add", {"content": "synthetic fact", "user_id": "attacker", "agent_id": "attacker"}, hermes_home=str(homes[1]), user_id="attacker", agent_id="attacker")
    assert "error" not in json.loads(result)
    assert calls == [{"user_id": "user:alpha", "agent_id": "agent:alpha", "infer": False, "metadata": {"channel": "cli"}}]
    assert provider._hermes_home == home.resolve()
    provider._backend = None


def test_bundled_lookup_prevents_catalog_install_for_every_home(homes, monkeypatch) -> None:
    from hermes_cli import memory_provider_migration as migration
    from plugins.memory import find_provider_dir
    monkeypatch.setattr(migration, "catalog_source", lambda *a, **kw: pytest.fail("bundled provider must not query catalog"))
    for home in homes:
        with _scope(home):
            assert find_provider_dir("mem0").resolve() == Path(__file__).resolve().parents[3] / "plugins" / "memory" / "mem0"
            assert migration.provider_present("mem0", home)
            assert migration.migrate_home(home, install=lambda *a, **kw: pytest.fail("must not install")) is None


def test_disabled_memory_never_constructs_or_invokes_mem0(homes, monkeypatch) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    from run_agent import AIAgent
    cfg = {"memory": {"provider": ""}, "agent": {}}
    monkeypatch.setattr(Mem0MemoryProvider, "__init__", lambda self: pytest.fail("disabled memory must not construct Mem0"))
    monkeypatch.setattr(Mem0MemoryProvider, "post_setup", lambda *a, **kw: pytest.fail("disabled memory must not run setup"))
    with (
        _scope(homes[0]),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save,
        patch("plugins.memory.load_memory_provider") as load,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="synthetic-api-key", base_url="https://model.invalid/v1", quiet_mode=True, skip_context_files=True, skip_memory=False)
        assert agent._memory_manager is None
        load.assert_not_called()
        save.assert_not_called()
