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
    sync_env_files_from_process()
    return load_env_files(paths)


def sync_env_files_from_process(target: Path | None = None) -> str | None:
    """Persist ``UW_API_KEY`` from the process env into ``.env`` when missing on disk."""
    key = uw_api_key()
    if not key:
        return None
    path = target or (_REPO_ROOT / ".env")
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        for line in existing.splitlines():
            parsed = _parse_env_line(line)
            if parsed and parsed[0] == "UW_API_KEY" and parsed[1].strip():
                return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"UW_API_KEY={key}\n", encoding="utf-8")
    path.chmod(0o600)
    return str(path)


def load_env_files(paths: tuple[Path, ...] | None = None) -> list[str]:
    """Populate ``os.environ`` from env files without clobbering existing keys."""
    loaded: list[str] = []
    for path in paths or _DEFAULT_ENV_FILES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value
        loaded.append(str(path))
    return loaded


def uw_api_key() -> str | None:
    """Return the trimmed UW API key when configured."""
    value = os.environ.get("UW_API_KEY", "").strip()
    return value or None


def uw_api_configured() -> bool:
    return uw_api_key() is not None
