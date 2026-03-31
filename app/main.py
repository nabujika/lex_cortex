from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    yield


settings = get_settings()
static_dir = Path(__file__).resolve().parent / "static"
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(dashboard_router)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}

