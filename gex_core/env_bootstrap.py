"""Load optional env files before the app reads ``UW_API_KEY``.

Cloud agents and local/docker deployments often inject secrets via the process
environment, but operators may also keep keys in ``.env`` or ``config/spx.env``.
Files are parsed in order; existing environment variables are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENV_FILES = (
    _REPO_ROOT / ".env",
    _REPO_ROOT / "config" / "spx.env",
)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def bootstrap_env(paths: tuple[Path, ...] | None = None) -> list[str]:
    """Sync process secrets to disk, then load env files into ``os.environ``."""
    targets = paths or _DEFAULT_ENV_FILES
    for path in targets:
        sync_env_files_from_process(path)
    return load_env_files(paths)


def _env_value_missing(key: str) -> bool:
    value = os.environ.get(key)
    return value is None or not str(value).strip()


def _file_has_key(path: Path, env_key: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed and parsed[0] == env_key and parsed[1].strip():
            return True
    return False


def _upsert_env_file(path: Path, updates: dict[str, str]) -> bool:
    """Merge key=value pairs into an env file without clobbering unrelated lines."""
    if not updates:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    if path.is_file():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    present = set()
    merged: list[str] = []
    for line in existing_lines:
        parsed = _parse_env_line(line)
        if parsed and parsed[0] in updates:
            key = parsed[0]
            if key not in present:
                merged.append(f"{key}={updates[key]}")
                present.add(key)
            continue
        merged.append(line)
    for key, value in updates.items():
        if key not in present:
            merged.append(f"{key}={value}")
            present.add(key)
    path.write_text("\n".join(merged).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o600)
    return True


_SYNC_ENV_KEYS = (
    "UW_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEX_HERMES_PROVIDER",
    "GEX_AGENT_MODEL",
)


def sync_env_files_from_process(target: Path | None = None) -> list[str]:
    """Persist injected secrets from the process env into local env files when missing."""
    updates = {
        key: os.environ.get(key, "").strip()
        for key in _SYNC_ENV_KEYS
        if not _env_value_missing(key)
    }
    if not updates:
        return []
    targets = (target,) if target is not None else _DEFAULT_ENV_FILES
    written: list[str] = []
    for path in targets:
        missing = {k: v for k, v in updates.items() if not _file_has_key(path, k)}
        if not missing:
            continue
        if _upsert_env_file(path, missing):
            written.append(str(path))
    return written


def load_env_files(paths: tuple[Path, ...] | None = None) -> list[str]:
    """Populate ``os.environ`` from env files.

    Existing non-blank environment variables win. Blank placeholders like
    ``UW_API_KEY=`` from docker compose are treated as unset so file values load.
    """
    loaded: list[str] = []
    for path in paths or _DEFAULT_ENV_FILES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if _env_value_missing(key):
                os.environ[key] = value
        loaded.append(str(path))
    return loaded


def uw_api_key() -> str | None:
    """Return the trimmed UW API key when configured."""
    value = os.environ.get("UW_API_KEY", "").strip()
    return value or None


def uw_api_configured() -> bool:
    return uw_api_key() is not None


def llm_api_key_diagnostics() -> dict:
    """Non-secret diagnostics for Hermes / OpenAI provider configuration."""
    openai_present = not _env_value_missing("OPENAI_API_KEY")
    openrouter_present = not _env_value_missing("OPENROUTER_API_KEY")
    provider = os.environ.get("GEX_HERMES_PROVIDER", "").strip().lower() or None
    configured = openai_present or openrouter_present
    if openai_present and (not provider or provider == "openai"):
        active = "openai"
    elif openrouter_present and (not provider or provider == "openrouter"):
        active = "openrouter"
    elif provider:
        active = provider
    else:
        active = None
    return {
        "llm_configured": configured,
        "llm_provider": active,
        "openai_api_key_present": openai_present,
        "openrouter_api_key_present": openrouter_present,
        "gex_hermes_provider": provider,
        "gex_agent_model": os.environ.get("GEX_AGENT_MODEL", "").strip() or None,
    }


def uw_api_key_diagnostics() -> dict:
    """Non-secret diagnostics for why live UW data is or is not available."""
    env_present = "UW_API_KEY" in os.environ
    env_blank = env_present and _env_value_missing("UW_API_KEY")
    file_hits = [str(path) for path in _DEFAULT_ENV_FILES if path.is_file()]
    source = "none"
    if uw_api_configured():
        if env_present and not env_blank:
            source = "environment"
        elif file_hits:
            source = "file"
        else:
            source = "unknown"
    return {
        "uw_api_configured": uw_api_configured(),
        "uw_api_key_source": source,
        "uw_api_env_present": env_present,
        "uw_api_env_blank": env_blank,
        "uw_api_env_files_present": file_hits,
    }
