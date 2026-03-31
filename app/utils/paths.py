from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


def resolve_workspace_root() -> Path:
    return get_settings().resolved_workspace_root


def resolve_path_from_workspace(path_str: str | None) -> Path | None:
    return get_settings().resolve_workspace_path(path_str)

