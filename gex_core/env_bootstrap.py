"""Load optional env files before the app reads ``UW_API_KEY``.

Cloud agents and local/docker deployments often inject secrets via the process
environment, but operators may also keep keys in ``.env`` or ``config/spx.env``.
Files are parsed in order; existing environment variables are never overwritten.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _file_has_uw_key(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed and parsed[0] == "UW_API_KEY" and parsed[1].strip():
            return True
    return False


def sync_env_files_from_process(target: Path | None = None) -> list[str]:
    """Persist ``UW_API_KEY`` from the process env into local env files when missing."""
    key = uw_api_key()
    if not key:
        return []
    targets = (target,) if target is not None else _DEFAULT_ENV_FILES
    written: list[str] = []
    for path in targets:
        if _file_has_uw_key(path):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"UW_API_KEY={key}\n", encoding="utf-8")
        path.chmod(0o600)
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


def parse_env_minutes(name: str, default: float = 10.0) -> float:
    """Parse an env var as minutes; accepts floats like ``0.5`` or ``.5``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %.4g minutes", name, raw, default)
        return float(default)
    if value <= 0:
        logger.warning("Non-positive %s=%r; using default %.4g minutes", name, raw, default)
        return float(default)
    return value


def uw_api_key() -> str | None:
    """Return the trimmed UW API key when configured."""
    value = os.environ.get("UW_API_KEY", "").strip()
    return value or None


def uw_api_configured() -> bool:
    return uw_api_key() is not None


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
