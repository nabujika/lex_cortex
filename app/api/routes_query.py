from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


@router.post("/query")
def query(payload: QueryRequest, db: Session = Depends(get_db_session)) -> dict:
    return QueryService(db).ask(payload.question)

