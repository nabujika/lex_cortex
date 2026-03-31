import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from db import (
    delete_user,
    export_schema_summary,
    fetch_table_rows,
    get_user_by_username,
    init_db,
    list_public_tables,
    list_users,
    create_user,
    seed_default_users,
    verify_password,
)
from ingest import ingest_directory
from rag_graph import run_rag_query

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

app = FastAPI(title="The Legal Brain", version="1.0.0")
FRONTEND_PATH = Path(__file__).with_name("frontend.html")


@app.on_event("startup")
def on_startup() -> None:
    """Ensure the app_user table exists and default accounts are seeded."""
    try:
        from db import db_cursor

        with db_cursor() as (_, cur):
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_user (
                    user_id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    hashed_password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        seed_default_users()
    except Exception as exc:
        print(f"[startup] Warning: could not seed users: {exc}")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if not user:
        raise credentials_exception
    return user


def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required."
        )
    return current_user


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserOut(BaseModel):
    user_id: int
    username: str
    role: str
    created_at: Any


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language legal question.")
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    cases_used: List[Dict[str, Any]]
    filters: Dict[str, Any]


class IngestRequest(BaseModel):
    directory: str = Field(
        default=".", description="Directory containing judgment/statute PDFs."
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", response_class=FileResponse)
def frontend() -> FileResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(FRONTEND_PATH)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
    )


@app.get("/auth/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return {"username": current_user["username"], "role": current_user["role"]}


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------


@app.post("/admin/users", response_model=UserOut)
def create_new_user(body: UserCreate, _: Dict = Depends(require_admin)) -> UserOut:
    existing = get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")
    row = create_user(body.username, body.password, body.role)
    return UserOut(**row)


@app.get("/admin/users")
def get_users(_: Dict = Depends(require_admin)) -> List[Dict[str, Any]]:
    return list_users()


@app.delete("/admin/users/{username}")
def remove_user(
    username: str, current_user: Dict = Depends(require_admin)
) -> Dict[str, str]:
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")
    if not delete_user(username):
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "deleted", "username": username}


# ---------------------------------------------------------------------------
# Public / read-only endpoints (any authenticated user)
# ---------------------------------------------------------------------------


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "service": "The Legal Brain"}


@app.get("/schema")
def schema_summary(_: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    return export_schema_summary()


@app.get("/ui/tables")
def get_tables(_: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    return {"tables": list_public_tables()}


@app.get("/ui/tables/{table_name}")
def get_table_rows(
    table_name: str, limit: int = 50, _: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        return fetch_table_rows(table_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query_legal_brain(
    request: QueryRequest, _: Dict = Depends(get_current_user)
) -> QueryResponse:
    return QueryResponse(**run_rag_query(request.query, top_k=request.top_k))


# ---------------------------------------------------------------------------
# Admin-only endpoints
# ---------------------------------------------------------------------------


@app.post("/admin/init-db")
def initialize_database(_: Dict = Depends(require_admin)) -> Dict[str, str]:
    init_db()
    seed_default_users()
    return {"status": "initialized"}


@app.post("/admin/ingest")
def ingest_documents(
    request: IngestRequest, _: Dict = Depends(require_admin)
) -> Dict[str, Any]:
    directory = Path(request.directory).resolve()
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    results = ingest_directory(directory)
    return {
        "directory": str(directory),
        "documents_processed": len(results),
        "results": results,
    }
