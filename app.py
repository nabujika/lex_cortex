from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from db import export_schema_summary, fetch_table_rows, init_db, list_public_tables
from ingest import ingest_directory
from rag_graph import run_rag_query


app = FastAPI(title="The Legal Brain", version="1.0.0")
FRONTEND_PATH = Path(__file__).with_name("frontend.html")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language legal question.")
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    cases_used: List[Dict[str, Any]]
    filters: Dict[str, Any]


class IngestRequest(BaseModel):
    directory: str = Field(default=".", description="Directory containing judgment/statute PDFs.")


@app.get("/", response_class=FileResponse)
def frontend() -> FileResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "service": "The Legal Brain"}


@app.get("/schema")
def schema_summary() -> Dict[str, Any]:
    return export_schema_summary()


@app.get("/ui/tables")
def get_tables() -> Dict[str, Any]:
    return {"tables": list_public_tables()}


@app.get("/ui/tables/{table_name}")
def get_table_rows(table_name: str, limit: int = 50) -> Dict[str, Any]:
    try:
        return fetch_table_rows(table_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
