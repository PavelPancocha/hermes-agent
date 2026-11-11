"""Tests for `hermes memory setup [provider]` routing.

The `memory setup` subcommand accepts an optional positional ``provider`` so a
fresh install can configure a specific provider directly (e.g.
``hermes memory setup honcho``) without the interactive picker — which matters
because the per-provider ``hermes <provider>`` subcommand is only registered
once that provider is active.
"""

import argparse

import pytest

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import memory_setup


class TestMemorySetupProviderRouting:
    def test_setup_with_provider_arg_skips_picker(self):
        """`memory setup honcho` routes straight to cmd_setup_provider."""
        args = SimpleNamespace(memory_command="setup", provider="honcho")
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        direct.assert_called_once_with("honcho")
        picker.assert_not_called()


    def test_unknown_provider_reports_and_returns_early(self, capsys):
        """An unknown provider name surfaces a helpful message and returns
        before any config load/save (the not-found guard precedes those imports)."""
        memory_setup.cmd_setup_provider("notaprovider")
        out = capsys.readouterr().out
        assert "not found" in out
        assert "hermes memory setup" in out


def test_setup_parser_preserves_complete_provider_arguments() -> None:
    from hermes_cli.subcommands.memory import build_memory_parser

    parser = argparse.ArgumentParser()
    build_memory_parser(parser.add_subparsers(dest="command"), cmd_memory=memory_setup.memory_command)
    provider_args = [
        "--mode", "self_hosted_http", "--host", "https://memory.invalid/api",
        "--api-key", "synthetic-key", "--user-id", "exact-user", "--agent-id", "exact-agent",
    ]
    args = parser.parse_args(["memory", "setup", "mem0", *provider_args])
    assert args.provider == "mem0"
    assert args.provider_args == provider_args
    assert args.func is memory_setup.memory_command


@pytest.mark.parametrize("provider", ["retaindb", "mem0"])
@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_setup_help_after_provider_exits_without_provider_work(
    provider, help_flag, tmp_path, monkeypatch, capsys,
) -> None:
    from hermes_cli.subcommands.memory import build_memory_parser

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  provider: kept-existing\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("MEM0_API_KEY=synthetic-existing-key\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    parser = argparse.ArgumentParser(prog="hermes")
    build_memory_parser(parser.add_subparsers(dest="command"), cmd_memory=memory_setup.memory_command)
    with (
        patch.object(memory_setup, "cmd_setup_provider") as direct,
        patch.object(memory_setup, "cmd_setup") as picker,
        patch("hermes_cli.config.load_config") as load,
        patch("hermes_cli.config.save_config") as save,
    ):
        with pytest.raises(SystemExit) as exit_info:
            args = parser.parse_args(["memory", "setup", provider, help_flag])
            args.func(args)
        assert exit_info.value.code == 0
        direct.assert_not_called()
        picker.assert_not_called()
        load.assert_not_called()
        save.assert_not_called()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    output = capsys.readouterr()
    assert "usage: hermes memory setup" in output.out
    assert "PROVIDER_ARG" in output.out
    assert not output.err
