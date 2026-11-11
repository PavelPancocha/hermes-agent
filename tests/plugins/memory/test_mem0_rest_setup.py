"""REST-only Mem0 setup through the real CLI and dependency preparation path."""
from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path
from typing import Iterator

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli import memory_setup
from hermes_cli.subcommands.memory import build_memory_parser

_KEY = "synthetic-new-key"
_HOST = "https://mem0.invalid/api/"
_USER, _AGENT = "literal user:shared", "literal agent:one"


@pytest.fixture
def setup_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "SOUL.md").write_text("Synthetic existing soul.\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("memory:\n  provider: kept-existing\n", encoding="utf-8")
    (tmp_path / ".env").write_text("MEM0_API_KEY=synthetic-old-env-key\nKEEP=value\n", encoding="utf-8")
    token = set_secret_scope({"MEM0_API_KEY": "synthetic-old-env-key"}, profile_home=str(tmp_path))
    real_import = builtins.__import__

    def deny_sdk(name: str, *args, **kwargs):
        if name == "mem0" or name.startswith("mem0."):
            pytest.fail("Mem0 SDK import is forbidden in the REST variant")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_sdk)
    monkeypatch.setattr("pm.sync_venv", lambda *a, **kw: pytest.fail("REST setup must not invoke PM"))
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("setup must not construct an HTTP backend"))
    try:
        yield tmp_path
    finally:
        reset_secret_scope(token)


def _flags(mode: str = "self_hosted_http", *, equals: bool = False, host_flag: str = "--host") -> list[str]:
    pairs = [("--mode", mode), (host_flag, _HOST), ("--api-key", _KEY),
             ("--user-id", _USER), ("--agent-id", _AGENT)]
    return [f"{key}={value}" for key, value in pairs] if equals else [item for pair in pairs for item in pair]


def _run(flags: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["hermes", "memory", "setup", "mem0", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    parser = argparse.ArgumentParser(prog="hermes")
    build_memory_parser(parser.add_subparsers(), cmd_memory=memory_setup.memory_command)
    args = parser.parse_args(argv[1:])
    args.func(args)


def _bytes(home: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in home.iterdir() if p.is_file()}


@pytest.mark.parametrize("mode", ["selfhosted", "self_hosted", "self_hosted_http", "rest", "http", "self-hosted", "self-hosted-http", "platform"])
@pytest.mark.parametrize("equals,host_flag", [(False, "--host"), (True, "--api-url")])
def test_deployed_setup_command_preserves_ids_and_writes_only_env_key(setup_home, monkeypatch, capsys, mode, equals, host_flag) -> None:
    from plugins.memory.mem0 import _setup
    (setup_home / "mem0.json").write_text(json.dumps({"api_key": "synthetic-inline-old", "unknown": {"keep": True}}))
    probes: list[tuple[str, str, int]] = []
    monkeypatch.setattr(_setup, "_prompt", lambda *a, **kw: pytest.fail("complete command must not prompt"))
    monkeypatch.setattr(_setup, "_http_get", lambda host, path, timeout: probes.append((host, path, timeout)))
    _run(_flags(mode, equals=equals, host_flag=host_flag), monkeypatch)
    cfg = json.loads((setup_home / "mem0.json").read_text())
    assert cfg == {"api_key": "", "unknown": {"keep": True}, "mode": "self_hosted_http", "rerank": "false", "host": _HOST, "user_id": _USER, "agent_id": _AGENT}
    from agent.secret_scope import load_env_file
    assert load_env_file(setup_home / ".env") == {"MEM0_API_KEY": _KEY, "KEEP": "value"}
    import hermes_yaml
    assert hermes_yaml.safe_load((setup_home / "config.yaml").read_text())["memory"]["provider"] == "mem0"
    assert probes == [(_HOST, "/docs", 5)]
    output = capsys.readouterr()
    assert _KEY not in output.out + output.err
    assert "synthetic-inline-old" not in output.out + output.err


def test_dry_run_preserves_every_file_and_skips_probe_and_dependencies(setup_home, monkeypatch, capsys) -> None:
    from plugins.memory.mem0 import _setup
    (setup_home / "mem0.json").write_text('{"unknown": "keep"}\n')
    before = _bytes(setup_home)
    monkeypatch.setattr(_setup, "_prompt", lambda *a, **kw: pytest.fail("complete command must not prompt"))
    monkeypatch.setattr(_setup, "_http_get", lambda *a, **kw: pytest.fail("dry-run must not probe"))
    _run([*_flags(), "--dry-run"], monkeypatch)
    assert _bytes(setup_home) == before
    output = capsys.readouterr()
    assert _KEY not in output.out + output.err


@pytest.mark.parametrize("mode", ["oss", "cloud", "unknown", "platform"])
def test_unsupported_mode_exits_before_prompt_or_mutation(setup_home, monkeypatch, mode) -> None:
    from plugins.memory.mem0 import _setup
    before = _bytes(setup_home)
    monkeypatch.setattr(_setup, "_prompt", lambda *a, **kw: pytest.fail("unsupported mode must not prompt"))
    monkeypatch.setattr(_setup, "_http_get", lambda *a, **kw: pytest.fail("unsupported mode must not probe"))
    with pytest.raises(SystemExit):
        _run(["--mode", mode], monkeypatch)
    assert _bytes(setup_home) == before


@pytest.mark.parametrize("bad_flags", [["--wat=do-not-echo"], ["--api-key"], ["--host", "--api-key", "do-not-echo"], ["--dry-run=do-not-echo"]])
def test_invalid_flags_fail_without_prompt_or_supplied_value(setup_home, monkeypatch, capsys, bad_flags) -> None:
    from plugins.memory.mem0 import _setup
    before = _bytes(setup_home)
    monkeypatch.setattr(_setup, "_prompt", lambda *a, **kw: pytest.fail("invalid flags must not prompt"))
    with pytest.raises(SystemExit) as exc:
        _run(bad_flags, monkeypatch)
    assert "do-not-echo" not in str(exc.value)
    assert "do-not-echo" not in capsys.readouterr().out
    assert _bytes(setup_home) == before


@pytest.mark.parametrize("host", ["https://user:password@mem0.invalid/api", "https://mem0.invalid/api?key=secret", "https://mem0.invalid/#secret", "file:///tmp/mem0", "https://mem0.invalid:bad/api"])
def test_invalid_candidate_cannot_write_or_probe(setup_home, monkeypatch, host) -> None:
    from plugins.memory.mem0 import _setup
    before = _bytes(setup_home)
    monkeypatch.setattr(_setup, "_http_get", lambda *a, **kw: pytest.fail("invalid host must not probe"))
    flags = _flags()
    flags[flags.index(_HOST)] = host
    with pytest.raises(SystemExit):
        _run(flags, monkeypatch)
    assert _bytes(setup_home) == before


@pytest.mark.parametrize("failure", ["env", "json"])
def test_failed_write_keeps_old_activation_and_reports_no_secret(setup_home, monkeypatch, capsys, failure) -> None:
    from plugins.memory.mem0 import Mem0MemoryProvider, _setup
    old_activation = (setup_home / "config.yaml").read_bytes()
    old_env = (setup_home / ".env").read_bytes()

    def denied(*args, **kwargs):
        raise PermissionError("synthetic denied write")

    if failure == "env":
        monkeypatch.setattr(_setup, "_write_env", denied)
    else:
        monkeypatch.setattr(Mem0MemoryProvider, "save_config", denied)
    monkeypatch.setattr(_setup, "_http_get", lambda *a, **kw: pytest.fail("failed setup must not probe"))
    with pytest.raises(PermissionError):
        _run(_flags(), monkeypatch)
    assert (setup_home / "config.yaml").read_bytes() == old_activation
    assert not (setup_home / "mem0.json").exists()
    assert ((setup_home / ".env").read_bytes() == old_env) is (failure == "env")
    output = capsys.readouterr()
    assert _KEY not in output.out + output.err


def test_real_declaration_and_dependency_preparation_require_no_packages(setup_home) -> None:
    from plugins.memory import find_provider_dir
    from pm.plugin_declarations import read_python_declaration
    declaration = read_python_declaration(find_provider_dir("mem0"))
    assert declaration.requirements == ()
    assert not declaration.installs_packages
    assert declaration.pyproject is None
    meta, inputs = memory_setup.memory_provider_dependency_inputs("mem0")
    assert inputs == {}
    assert meta["pip_dependencies"] == []
    assert "extra" not in meta
    assert memory_setup.prepare_memory_provider_dependencies("mem0") == (meta, None)


def test_api_key_prompt_shows_configured_without_key_suffix(setup_home, monkeypatch) -> None:
    from plugins.memory.mem0 import _setup
    labels: list[str] = []
    monkeypatch.setattr(_setup, "_prompt", lambda label, **kw: labels.append(label) or "")
    assert _setup._api_key_writes({}, "Server API key") == {}
    assert labels and "configured" in labels[0]
    assert "synthetic-old-env-key" not in labels[0]
    assert "-key" not in labels[0]
