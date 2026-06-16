"""Resolve writable data paths for exports and SQLite state."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIGURED = False


def repo_root() -> Path:
    return _REPO_ROOT


def _path_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".gex_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("GEX_DATA_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(_REPO_ROOT / "data")
    roots.append(Path.home() / ".gex-data")
    roots.append(Path(tempfile.gettempdir()) / "gex-data")
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def resolve_data_root() -> Path:
    for root in _candidate_roots():
        if _path_writable(root):
            return root
    fallback = Path(tempfile.gettempdir()) / "gex-data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def configure_data_paths() -> Path:
    """Pick a writable data root and export env defaults for sqlite/export paths."""
    global _CONFIGURED
    if _CONFIGURED:
        return resolve_data_root()

    preferred = _REPO_ROOT / "data"
    root = resolve_data_root()
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GEX_DATA_DIR", str(root))
    os.environ.setdefault("GEX_EXPORT_DIR", str(exports))
    os.environ.setdefault("GEX_TRADING_DB", str(root / "trading_journal.db"))
    os.environ.setdefault("GEX_INDEX_DB", str(root / "gex_index.db"))
    webull_token_dir = root / "webull"
    if not os.environ.get("GEX_PROCESSOR_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        webull_token_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WEBULL_OPENAPI_TOKEN_DIR", str(webull_token_dir))

    if root != preferred and not _path_writable(preferred):
        logger.warning(
            "Data directory %s is not writable for uid=%s; using %s instead. "
            "Mount your Railway volume at /app/data and ensure the container can chown it, "
            "or set GEX_DATA_DIR to a writable path.",
            preferred,
            os.getuid(),
            root,
        )

    try:
        from gex_core.exports import refresh_export_dir

        refresh_export_dir()
    except ImportError:
        pass

    _CONFIGURED = True
    return root
