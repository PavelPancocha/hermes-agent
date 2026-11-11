"""Fleet REST contract, exercised through discovery and real HTTPX requests."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from plugins.memory import load_memory_provider


def test_bundled_rest_provider_is_loadable() -> None:
    provider = load_memory_provider("mem0", register_skills=False)
    assert provider is not None, "Fleet REST provider must be bundled and loadable"
    assert {s["name"] for s in provider.get_tool_schemas()} == {
        "mem0_list", "mem0_search", "mem0_add", "mem0_update", "mem0_delete",
    }


@pytest.fixture
def rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple]:
    from plugins.memory.mem0 import Mem0MemoryProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"mode": "self_hosted_http", "host": "https://memory.invalid/api",
           "api_key": "synthetic-rest-key", "user_id": "hermes-user", "agent_id": "fleet-agent"}
    (tmp_path / "mem0.json").write_text(json.dumps(cfg))
    token = set_secret_scope({}, profile_home=str(tmp_path))
    requests: list[httpx.Request] = []
    reply: dict = {"body": {"results": []}, "status": 200}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(reply["status"], json=reply["body"])

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_: httpx.MockTransport(handle))
    provider = Mem0MemoryProvider()
    provider.initialize("session", hermes_home=tmp_path, user_id="gateway-user", platform="telegram")
    try:
        yield provider, requests, reply
    finally:
        provider.shutdown()
        reset_secret_scope(token)


def call(provider, tool: str, args: dict, **kwargs) -> dict:
    return json.loads(provider.handle_tool_call("mem0_" + tool, args, **kwargs))


def test_exact_add_search_update_delete_requests(rest: tuple) -> None:
    provider, requests, reply = rest
    reply["body"] = {"results": [], "event_id": "event"}
    assert call(provider, "add", {"content": "Remember a synthetic fact"}) == {
        "result": "Fact stored.", "event_id": "event"}
    assert call(provider, "search", {"query": "fact", "top_k": 100}) == {
        "result": "No relevant memories found."}
    reply["body"] = {"id": "m1", "user_id": "hermes-user", "agent_id": "other-agent"}
    assert "error" not in call(provider, "update", {"memory_id": "m1", "text": "updated"})
    assert "error" not in call(provider, "delete", {"memory_id": "m1"})
    assert [(r.method, r.url.path) for r in requests] == [
        ("POST", "/api/memories"), ("POST", "/api/search"), ("GET", "/api/memories/m1"),
        ("PUT", "/api/memories/m1"), ("GET", "/api/memories/m1"), ("DELETE", "/api/memories/m1")]
    assert all(r.headers["X-API-Key"] == "synthetic-rest-key" for r in requests)
    assert json.loads(requests[0].content) == {
        "messages": [{"role": "user", "content": "Remember a synthetic fact"}],
        "user_id": "hermes-user", "agent_id": "fleet-agent", "infer": False,
        "metadata": {"channel": "telegram"}}
    assert json.loads(requests[1].content) == {
        "query": "fact", "top_k": 50, "filters": {"user_id": "hermes-user"}}
    assert json.loads(requests[3].content) == {"text": "updated"}
    assert all(not r.content for r in (requests[2], requests[4], requests[5]))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("page,size,top_k,start,end", [(3, 100, 300, 200, 300), (6, 200, 1000, 1000, 1000)])
def test_bounded_list(rest: tuple, wrapped: bool, page: int, size: int, top_k: int, start: int, end: int) -> None:
    provider, requests, reply = rest
    records = [{"id": str(i), "memory": f"fact {i}"} for i in range(top_k)]
    reply["body"] = {"results": records, "count": top_k} if wrapped else records
    result = call(provider, "list", {"page": page, "page_size": size})
    assert dict(requests[-1].url.params) == {"user_id": "hermes-user", "top_k": str(top_k)}
    assert requests[-1].method == "GET" and requests[-1].url.path == "/api/memories"
    if start == end:
        assert result == {"result": "No memories stored yet."}
    else:
        assert result == {"results": records[start:end], "count": top_k, "page": page, "page_size": size}


@pytest.mark.parametrize("tool,args", [("search", {"query": "safe"}), ("add", {"content": "safe"}), ("list", {})])
def test_tool_identity_overrides_are_ignored(rest: tuple, tool: str, args: dict) -> None:
    provider, requests, _ = rest
    malicious = {"user_id": "foreign", "agent_id": "foreign", "filters": {"user_id": "foreign"},
                 "hermes_home": "/foreign", "run_id": "foreign"}
    assert "error" not in call(provider, tool, {**args, **malicious}, **malicious)
    request = requests[-1]
    payload = json.loads(request.content) if request.content else dict(request.url.params)
    assert "foreign" not in json.dumps(payload)
    assert provider._user_id == "hermes-user"
    assert provider._agent_id == "fleet-agent"


@pytest.mark.parametrize("owner", [None, "", "foreign", 1, [], {}])
@pytest.mark.parametrize("tool", ["update", "delete"])
def test_ownership_fails_closed(rest: tuple, owner, tool: str) -> None:
    provider, requests, reply = rest
    reply["body"] = {"id": "m1", **({"user_id": owner} if owner is not None else {})}
    assert "error" in call(provider, tool, {"memory_id": "m1", "text": "safe"})
    assert [r.method for r in requests] == ["GET"]
    assert provider._consecutive_failures == 0


@pytest.mark.parametrize("body", [{}, None])
def test_nonexistent_memory_does_not_mutate(rest: tuple, body) -> None:
    provider, requests, reply = rest
    reply["body"] = body
    assert "error" in call(provider, "delete", {"memory_id": "m1"})
    assert [r.method for r in requests] == ["GET"]


@pytest.mark.parametrize("tool,args", [("search", {"query": "safe", "top_k": "bad"}),
    ("list", {"page": "bad"}), ("list", {"page_size": None}), ("list", {"page": float("inf")})])
def test_invalid_numbers_do_not_request_or_trip_breaker(rest: tuple, tool: str, args: dict) -> None:
    provider, requests, _ = rest
    assert "error" in call(provider, tool, args)
    assert requests == [] and provider._consecutive_failures == 0


@pytest.mark.parametrize("tool,args", [("list", {}), ("search", {"query": "safe"}), ("add", {"content": "safe"})])
@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_failures_are_tool_errors(rest: tuple, tool: str, args: dict, status: int) -> None:
    provider, requests, reply = rest
    reply["status"] = status
    assert "error" in call(provider, tool, args)
    assert len(requests) == 1 and provider._consecutive_failures == 1


@pytest.mark.parametrize("tool", ["list", "search"])
@pytest.mark.parametrize("body", [[], {"results": []}])
def test_empty_responses(rest: tuple, tool: str, body) -> None:
    provider, _, reply = rest
    reply["body"] = body
    assert "result" in call(provider, tool, {"query": "safe"})


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"mode": "oss"}, {"mode": "cloud"}, {"mode": "platform", "host": ""},
    *({key: ""} for key in ("api_key", "user_id", "agent_id", "host")),
    *({"host": host} for host in ("ftp://memory.invalid", "https://u:p@memory.invalid", "https://memory.invalid?q=x",
                                  "https://memory.invalid/#f", "https://memory.invalid:bad", "https:///missing")),
])
def test_invalid_config_never_constructs_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"mode": "rest", "host": "http://memory.invalid/api", "api_key": "synthetic-key", "user_id": "u", "agent_id": "a", **changes}
    (tmp_path / "mem0.json").write_text(json.dumps(cfg))
    token = set_secret_scope({}, profile_home=str(tmp_path))
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("invalid config constructed a client"))
    try:
        provider = Mem0MemoryProvider()
        assert not provider.is_available()
        assert provider.unavailable_reason()
        with pytest.raises(ValueError, match="Mem0"):
            provider.initialize("s", hermes_home=tmp_path)
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize("mode", ["selfhosted", "self_hosted", "self_hosted_http", "rest", "http", "platform", " SELF-HOSTED-HTTP "])
def test_supported_modes_preserve_volume_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg_path = tmp_path / "mem0.json"
    cfg_path.write_text(json.dumps({"mode": mode, "host": "https://memory.invalid/api/", "api_key": "synthetic-key", "user_id": "hermes-user", "agent_id": "agent"}))
    before = cfg_path.read_bytes()
    token = set_secret_scope({}, profile_home=str(tmp_path))
    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_: httpx.MockTransport(lambda _: httpx.Response(200, json=[])))
    provider = Mem0MemoryProvider()
    try:
        assert provider.is_available()
        provider.initialize("s", hermes_home=tmp_path, user_id="gateway")
        assert provider._user_id == "hermes-user" and provider._host == "https://memory.invalid/api/"
        assert cfg_path.read_bytes() == before
    finally:
        provider.shutdown()
        reset_secret_scope(token)


@pytest.mark.parametrize("field", ["host", "api_key", "user_id", "agent_id"])
def test_missing_config_field_names_setup_command(field: str) -> None:
    from plugins.memory.mem0 import _rest_config_error

    cfg = {"mode": "rest", "host": "https://memory.invalid/api", "api_key": "synthetic-key", "user_id": "u", "agent_id": "a"}
    del cfg[field]
    reason = _rest_config_error(cfg)
    assert field in reason
    assert "hermes memory setup mem0" in reason


def test_shutdown_bounds_both_blocked_threads_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider

    class BlockedThread:
        def __init__(self, name: str) -> None:
            self.name = name
            self.timeouts: list[float] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            assert timeout is not None and 0 < timeout <= 5.0, "unbounded shutdown join"
            self.timeouts.append(timeout)

    provider = Mem0MemoryProvider()
    backend = provider._backend = Mock()
    prefetch = provider._prefetch_thread = BlockedThread("mem0-prefetch")
    sync = provider._sync_thread = BlockedThread("mem0-sync")
    provider.shutdown()
    assert prefetch.timeouts == sync.timeouts == [5.0]
    assert "mem0-prefetch" in caplog.text and "mem0-sync" in caplog.text
    backend.close.assert_called_once_with()
    assert provider._backend is None
