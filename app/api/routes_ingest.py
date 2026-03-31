from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.ingestion.workspace_discovery import discover_workspace_files
from app.services.ingest_service import IngestService

router = APIRouter(tags=["ingestion"])


class IngestWorkspaceRequest(BaseModel):
    pdf_dir: str | None = None


@router.post("/ingest/workspace")
def ingest_workspace(payload: IngestWorkspaceRequest, db: Session = Depends(get_db_session)) -> dict:
    service = IngestService(db)
    return service.ingest_workspace(pdf_dir=payload.pdf_dir)


@router.get("/workspace/files")
def list_workspace_files(db: Session = Depends(get_db_session)) -> dict:
    discovered = discover_workspace_files()
    return {
        "workspace_root": str(discovered.workspace_root),
        "searched_locations": [str(path) for path in discovered.searched_locations],
        "pdf_files": [str(path) for path in discovered.pdf_files],
        "diagram_files": [str(path) for path in discovered.diagram_files],
    }
