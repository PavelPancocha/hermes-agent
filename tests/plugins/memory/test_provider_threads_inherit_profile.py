"""Multiplex invariant: every memory provider's background thread runs under the spawner's profile.

Profile isolation is a ContextVar-scoped HERMES_HOME override; a plain ``threading.Thread`` starts
with an EMPTY context, so a provider's prefetch/sync/writer thread would silently resolve the DEFAULT
profile's home (and fail closed on scoped secrets). Each case drives the provider's real spawn path
with a fake backend and asserts the thread saw the parent's home.
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def _probe_home(seen: dict, key: str = "home"):
    def _record(*_args, **_kwargs):
        seen[key] = get_hermes_home()
    return _record


def _retaindb(seen, tmp_path):
    from plugins.memory import retaindb

    p = retaindb.RetainDBMemoryProvider()
    p._client = MagicMock()
    p._context_overlay = lambda query: {"context": seen.setdefault("home", get_hermes_home()) and "ctx"}
    p._client.ask_user.return_value = {"answer": ""}
    p._client.get_agent_model.return_value = {}
    p.queue_prefetch("what do you know")
    return list(p._prefetch_threads)


def _byterover(seen, tmp_path):
    from plugins.memory import byterover

    p = byterover.ByteRoverMemoryProvider()
    p._curate = _probe_home(seen)
    return [p._curate_in_background("content", name="brv-test", what="test")]


_PROVIDERS = {
    "retaindb": _retaindb, "byterover": _byterover,
}


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_provider_background_thread_sees_spawner_profile_home(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    profile_home = tmp_path / "profiles" / "b"
    profile_home.mkdir(parents=True)
    seen: dict = {}
    token = set_hermes_home_override(profile_home)
    try:
        threads = _PROVIDERS[name](seen, tmp_path)
    finally:
        reset_hermes_home_override(token)
    for t in threads:
        if t is not None:
            t.join(timeout=10)
    assert seen.get("home") == profile_home, f"{name}: background thread resolved {seen.get('home')}"


def _mem0_scope_spawn(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.secret_scope import (
        current_secret_scope_home, get_secret, reset_secret_scope, set_secret_scope,
    )
    from plugins.memory.mem0 import Mem0MemoryProvider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    home = tmp_path / "profiles" / "mem0-owner"
    home.mkdir(parents=True)
    seen: dict = {}
    finished = threading.Event()

    def record(*args, **kwargs):
        seen.update(home=get_hermes_home(), secret_home=current_secret_scope_home(), key=get_secret("MEM0_API_KEY"))
        finished.set()
        return [] if kind == "prefetch" else {}

    provider = Mem0MemoryProvider()
    provider._backend = MagicMock()
    provider._backend.search.side_effect = record
    provider._backend.add.side_effect = record
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"MEM0_API_KEY": "synthetic-thread-key"}, profile_home=str(home))
    try:
        if kind == "prefetch":
            provider.on_turn_start(1, "synthetic question")
        else:
            provider.sync_turn("synthetic user", "synthetic assistant")
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    assert finished.wait(timeout=10)
    provider.shutdown()
    assert seen == {"home": home, "secret_home": str(home), "key": "synthetic-thread-key"}


def test_mem0_prefetch_inherits_owning_home_and_secret_scope(tmp_path, monkeypatch) -> None:
    _mem0_scope_spawn("prefetch", tmp_path, monkeypatch)


def test_mem0_sync_inherits_owning_home_and_secret_scope(tmp_path, monkeypatch) -> None:
    _mem0_scope_spawn("sync", tmp_path, monkeypatch)
