"""Flag-based or interactive setup for the fleet's self-hosted Mem0 REST provider."""
from __future__ import annotations

import getpass
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict

from agent.secret_scope import get_secret
from utils import read_json_or_empty

from . import _REST_MODES, _load_config, _rest_config_error


class SetupFlags(TypedDict):
    mode: str
    api_key: str
    host: str
    user_id: str
    agent_id: str
    dry_run: bool


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt for a value with optional default and secret masking."""
    sys.stdout.write(f"  {label}{f' [{default}]' if default else ''}: ")
    sys.stdout.flush()
    val = getpass.getpass(prompt="") if secret and sys.stdin.isatty() else sys.stdin.readline().strip()
    return val or (default or "")


def _http_get(url: str, path: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(f"{url.rstrip('/')}{path}", method="GET"), timeout=timeout)


def _api_key_writes(flags: SetupFlags, label: str, *, url: str | None = None, fresh_label: str | None = None) -> dict[str, str]:
    """MEM0_API_KEY for .env: from --api-key, else prompt (keeping any key already in the profile scope)."""
    if flags.get("api_key"):
        return {"MEM0_API_KEY": flags["api_key"]}
    existing = get_secret("MEM0_API_KEY", "")
    if url and not existing:
        print(f"  Get yours at {url}")
    val = _prompt(f"{label} (current: configured, blank to keep)" if existing else fresh_label or label, secret=True)
    return {"MEM0_API_KEY": val} if val else {}


def _print_dry_run(summary: str, env_writes: dict) -> None:
    print(f"\n  [dry-run] Would save config: {summary}")
    if env_writes:
        print("  [dry-run] Would write API key to .env")
    print("  [dry-run] No files written.\n")


_FLAG_KEYS = ("mode", "api_key", "host", "user_id", "agent_id")


def parse_flags(argv: list[str] | None = None) -> SetupFlags:
    args = argv if argv is not None else sys.argv[1:]
    flags: SetupFlags = {"mode": "", "api_key": "", "host": "", "user_id": "", "agent_id": "", "dry_run": False}
    flag_map = {"--" + key.replace("_", "-"): key for key in _FLAG_KEYS}
    flag_map["--api-url"] = "host"
    i = 0
    while i < len(args):
        token, equals, value = args[i].partition("=")
        if token == "--dry-run" and not equals:
            flags["dry_run"] = True
        elif token in flag_map:
            if not equals:
                i += 1
                if i >= len(args) or args[i].startswith("-"):
                    raise SystemExit("Invalid Mem0 setup arguments: missing flag value.")
                value = args[i]
            if not value:
                raise SystemExit("Invalid Mem0 setup arguments: missing flag value.")
            flags[flag_map[token]] = value
        elif token.startswith("-"):
            raise SystemExit("Invalid Mem0 setup arguments: unrecognized flag.")
        i += 1
    return flags


def _write_env(env_path: Path, env_writes: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig like the canonical .env readers: a BOM'd first line would miss the key match and get duplicated.
    existing_lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    keys = [line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None for line in existing_lines]
    new_lines = [f"{k}={env_writes[k]}" if k in env_writes else line for k, line in zip(keys, existing_lines)]
    new_lines += [f"{k}={v}" for k, v in env_writes.items() if k not in keys]
    from utils import atomic_write_text

    atomic_write_text(env_path, "\n".join(new_lines) + "\n", mode=0o600)


def _activate_provider(config: dict) -> None:
    """Point config.yaml's memory.provider at mem0."""
    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)


def _persist_provider_config(hermes_home: str, config: dict, provider_config: dict, env_writes: dict[str, str], label: str, key_line: str, server: str | None = None) -> None:
    """Write credentials and JSON before activating; earlier successful writes are not rolled back."""
    from . import Mem0MemoryProvider

    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    if "MEM0_API_KEY" in env_writes:
        provider_config["api_key"] = ""  # new .env key wins over legacy inline key
    Mem0MemoryProvider().save_config(provider_config, hermes_home)
    _activate_provider(config)
    if server:
        _check_selfhosted_server(server)
    print("\n".join(["", f"  Memory provider: {label}", *([f"  Server: {server}"] if server else []), "  Activation saved to config.yaml", "  Provider config saved",
                     *([f"  {key_line}"] if env_writes else []), "", "  Start a new session to activate.", ""]))


def _check_selfhosted_server(host: str) -> None:
    """Best-effort reachability check for a self-hosted Mem0 server (non-fatal)."""
    try:
        _http_get(host, "/docs", 5)
        print(f"  ✓ Mem0 server reachable at {host}")
    except urllib.error.HTTPError:
        # Any HTTP response (401/403/404) still means something is listening.
        print(f"  ✓ Mem0 server responding at {host}")
    except Exception:  # health: allow BLE001 -- vendor reachability probe is best-effort; report only a sanitized warning
        print(f"  ⚠ Could not reach {host} — check the URL and that the server is running.")


def _setup_selfhosted(hermes_home: str, config: dict, flags: SetupFlags) -> None:
    provider_config = read_json_or_empty(Path(hermes_home) / "mem0.json")
    resolved = _load_config(hermes_home)
    print("\n  Configuring mem0 (self-hosted server):\n")
    host = flags.get("host") or _prompt("Mem0 server URL (e.g. http://localhost:8888)", default=resolved.get("host") or None)
    env_writes = _api_key_writes(flags, "Server API key")
    user_id = flags.get("user_id") or _prompt("User identifier", default=resolved.get("user_id") or "hermes-user")
    agent_id = flags.get("agent_id") or _prompt("Agent identifier", default=resolved.get("agent_id") or "hermes")
    provider_config.update(mode="self_hosted_http", rerank="false", host=host, user_id=user_id, agent_id=agent_id)
    candidate = {**resolved, **provider_config,
                 "api_key": env_writes.get("MEM0_API_KEY") or resolved.get("api_key", "")}
    if error := _rest_config_error(candidate):
        raise SystemExit(error)
    if flags.get("dry_run"):
        _print_dry_run(f"host={host}, user_id={user_id}, agent_id={agent_id}", env_writes)
        return
    _persist_provider_config(hermes_home, config, provider_config, env_writes,
                             "mem0 (self-hosted)", "API key saved to .env", server=host)


def post_setup(hermes_home: str, config: dict) -> None:
    flags = parse_flags(sys.argv[1:])
    mode = (flags["mode"] or "self_hosted_http").strip().lower().replace("-", "_")
    if mode not in _REST_MODES and not (mode == "platform" and flags.get("host")):
        raise SystemExit("This Mem0 setup supports self-hosted REST only.")
    _setup_selfhosted(hermes_home, config, flags)
