from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.db.models import Case, CaseChunk
from app.db.session import get_db_session
from app.ingestion.workspace_discovery import discover_workspace_files

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db_session)) -> dict:
    workspace = discover_workspace_files()

    case_count = db.scalar(select(func.count()).select_from(Case)) or 0
    chunk_count = db.scalar(select(func.count()).select_from(CaseChunk)) or 0
    recent_cases = list(
        db.execute(
            select(Case.title, Case.status, Case.judgment_date, Case.source_file)
            .order_by(Case.created_at.desc())
            .limit(8)
        )
    )

    return {
        "workspace_root": str(workspace.workspace_root),
        "discovered_pdf_count": len(workspace.pdf_files),
        "diagram_count": len(workspace.diagram_files),
        "ingested_case_count": case_count,
        "stored_chunk_count": chunk_count,
        "recent_cases": [
            {
                "title": row.title,
                "status": row.status,
                "judgment_date": row.judgment_date.isoformat() if row.judgment_date else None,
                "source_file": row.source_file,
            }
            for row in recent_cases
        ],
    }
