"""Load and validate api_spec.yaml against the UW loader implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_SPEC_PATH = Path(__file__).resolve().parent.parent / "api_spec.yaml"


@lru_cache(maxsize=1)
def load_api_spec() -> dict:
    if not _SPEC_PATH.is_file():
        raise FileNotFoundError(f"Missing API spec: {_SPEC_PATH}")
    with _SPEC_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def uw_endpoint_paths() -> list[str]:
    """OpenAPI paths from api_spec.yaml (e.g. ``/api/stock/{ticker}/greek-exposure/strike``)."""
    spec = load_api_spec()
    return sorted(spec.get("paths", {}).keys())


def gex_scaling(endpoint_family: str) -> dict:
    """Return scaling metadata for an endpoint family key from x-gex-scaling."""
    spec = load_api_spec()
    scaling = spec.get("x-gex-scaling", {})
    if endpoint_family not in scaling:
        raise KeyError(f"Unknown scaling family: {endpoint_family!r}")
    return dict(scaling[endpoint_family])


def implementation_rules() -> dict:
    return dict(load_api_spec().get("x-gex-implementation", {}))
