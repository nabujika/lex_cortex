from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db import export_schema_summary, init_db
from ingest import ingest_directory
from rag_graph import run_rag_query


app = FastAPI(title="The Legal Brain", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language legal question.")
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    cases_used: List[Dict[str, Any]]
    filters: Dict[str, Any]


class IngestRequest(BaseModel):
    directory: str = Field(default=".", description="Directory containing judgment/statute PDFs.")


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "service": "The Legal Brain"}


@app.get("/schema")
def schema_summary() -> Dict[str, Any]:
    return export_schema_summary()


@app.post("/admin/init-db")
def initialize_database() -> Dict[str, str]:
    init_db()
    return {"status": "initialized"}


@app.post("/admin/ingest")
def ingest_documents(request: IngestRequest) -> Dict[str, Any]:
    directory = Path(request.directory).resolve()
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    results = ingest_directory(directory)
    return {"directory": str(directory), "documents_processed": len(results), "results": results}


@app.post("/query", response_model=QueryResponse)
def query_legal_brain(request: QueryRequest) -> QueryResponse:
    return QueryResponse(**run_rag_query(request.query, top_k=request.top_k))
