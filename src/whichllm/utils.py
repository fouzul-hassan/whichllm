from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError


def _current_version() -> str:
    """Return installed package version."""
    try:
        return version("whichllm")
    except PackageNotFoundError:
        return "unknown"
