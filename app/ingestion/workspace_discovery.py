from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(slots=True)
class WorkspaceFiles:
    workspace_root: Path
    pdf_files: list[Path]
    diagram_files: list[Path]
    searched_locations: list[Path]


def _is_diagram_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return path.suffix.lower() in IMAGE_EXTENSIONS and ("er" in lower_name or "diagram" in lower_name)


def discover_workspace_files(pdf_dir: str | Path | None = None) -> WorkspaceFiles:
    settings = get_settings()
    workspace_root = settings.resolved_workspace_root
    searched_locations: list[Path] = []

    candidate_roots: list[Path] = []
    configured_dir = settings.resolve_workspace_path(pdf_dir) if pdf_dir else settings.resolve_workspace_path(settings.pdf_data_dir)
    if configured_dir:
        candidate_roots.append(configured_dir)
    for common_name in ("data", "cases"):
        candidate_roots.append(workspace_root / common_name)

    pdf_files: list[Path] = []
    for candidate in candidate_roots:
        searched_locations.append(candidate)
        if candidate.exists() and candidate.is_dir():
            pdf_files = sorted({path.resolve() for path in candidate.rglob("*") if path.suffix.lower() in PDF_EXTENSIONS})
            if pdf_files:
                logger.info("Discovered %s PDFs under %s", len(pdf_files), candidate)
                break

    if not pdf_files:
        searched_locations.append(workspace_root)
        pdf_files = sorted({path.resolve() for path in workspace_root.rglob("*") if path.suffix.lower() in PDF_EXTENSIONS})
        logger.info("Fallback recursive scan discovered %s PDFs under %s", len(pdf_files), workspace_root)

    diagram_files = sorted(
        {
            path.resolve()
            for path in workspace_root.rglob("*")
            if path.is_file() and _is_diagram_file(path)
        }
    )
    logger.info("Discovered %s ER diagram/image files", len(diagram_files))

    return WorkspaceFiles(
        workspace_root=workspace_root,
        pdf_files=pdf_files,
        diagram_files=diagram_files,
        searched_locations=searched_locations,
    )

