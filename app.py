import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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
    update_case_notes,
    update_user_permissions,
    verify_password,
)
from ingest import ingest_directory
from rag_graph import run_rag_query

VALID_PERMISSIONS = {"read", "write", "create_table", "update_records"}
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.local", override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours
LIVEKIT_ADMIN_AGENT_NAME = os.getenv(
    "LIVEKIT_ADMIN_AGENT_NAME", "legal-brain-admin-agent"
)
IS_AZURE_APP_SERVICE = bool(os.getenv("WEBSITE_HOSTNAME"))
DEPLOY_MARKER = os.getenv("DEPLOY_MARKER", "legal-brain-backend-2026-04-02-v2")


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


RUN_STARTUP_DB_BOOTSTRAP = env_flag(
    "RUN_STARTUP_DB_BOOTSTRAP", default=not IS_AZURE_APP_SERVICE
)
LIVEKIT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://legal-brain-admin-agent.vercel.app",
]

app = FastAPI(title="The Legal Brain", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=LIVEKIT_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
FRONTEND_PATH = Path(__file__).with_name("frontend.html")


@app.on_event("startup")
def on_startup() -> None:
    """Ensure the app_user table exists and default accounts are seeded."""
    if not RUN_STARTUP_DB_BOOTSTRAP:
        print("[startup] Skipping DB bootstrap on startup.")
        return

    try:
        from db import db_cursor, ensure_app_user_schema

        with db_cursor() as (_, cur):
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_user (
                    user_id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    hashed_password TEXT NOT NULL,
                    permissions TEXT[] NOT NULL DEFAULT '{read}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        ensure_app_user_schema()
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


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        env_file = Path(__file__).with_name(".env")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Missing required environment variable: {name}. "
                f"Add it to {env_file} and restart the backend."
            ),
        )
    return value


def get_admin_agent_room_name(user_id: int) -> str:
    session_suffix = uuid4().hex[:10]
    return f"admin-room-{user_id}-{session_suffix}"


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


def require_permission(permission: str):
    """Dependency factory: ensures the current user holds a specific permission."""
    def _checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_perms = current_user.get("permissions") or []
        if permission not in user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission}",
            )
        return current_user
    return Depends(_checker)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    permissions: List[str]
    username: str


class AgentTokenResponse(BaseModel):
    server_url: str
    participant_token: str
    room_name: str


class UserCreate(BaseModel):
    username: str
    password: str
    permissions: List[str] = Field(default=["read"])


class UserUpdate(BaseModel):
    permissions: List[str]


class UserOut(BaseModel):
    user_id: int
    username: str
    permissions: List[str]
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


class CaseNoteUpdate(BaseModel):
    notes: str = Field("", description="Free-text note for a case record.")


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
    permissions = user.get("permissions") or ["read"]
    token = create_access_token(
        data={
            "sub": user["username"],
            "user_id": int(user["user_id"]),
            "permissions": permissions,
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(
        access_token=token,
        token_type="bearer",
        user_id=int(user["user_id"]),
        permissions=permissions,
        username=user["username"],
    )


@app.get("/auth/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "permissions": current_user.get("permissions") or ["read"],
    }


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------


@app.get("/admin/agent-token", response_model=AgentTokenResponse)
def get_admin_agent_token(
    current_user: Dict[str, Any] = require_permission("create_table"),
) -> AgentTokenResponse:
    try:
        from livekit.api import (
            AccessToken,
            RoomAgentDispatch,
            RoomConfiguration,
            VideoGrants,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LiveKit dependencies are not installed on the FastAPI server.",
        ) from exc

    room_name = get_admin_agent_room_name(int(current_user["user_id"]))
    participant_identity = f"admin-{current_user['user_id']}-{uuid4().hex[:10]}"
    permissions = current_user.get("permissions") or ["read"]
    metadata = json.dumps(
        {
            "user_id": str(current_user["user_id"]),
            "username": current_user["username"],
            "permissions": permissions,
            "room_name": room_name,
        }
    )

    try:
        token = (
            AccessToken(
                api_key=get_required_env("LIVEKIT_API_KEY"),
                api_secret=get_required_env("LIVEKIT_API_SECRET"),
            )
            .with_identity(participant_identity)
            .with_name(current_user["username"])
            .with_metadata(metadata)
            .with_attributes(
                {
                    "user_id": str(current_user["user_id"]),
                    "username": current_user["username"],
                    "permissions": ",".join(permissions),
                }
            )
            .with_ttl(timedelta(hours=1))
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_publish_data=True,
                    can_subscribe=True,
                )
            )
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name=LIVEKIT_ADMIN_AGENT_NAME,
                            metadata=metadata,
                        )
                    ]
                )
            )
        )
        participant_token = token.to_jwt()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate LiveKit agent token: {exc}",
        ) from exc

    return AgentTokenResponse(
        server_url=get_required_env("LIVEKIT_URL"),
        participant_token=participant_token,
        room_name=room_name,
    )


@app.post("/admin/users", response_model=UserOut)
def create_new_user(body: UserCreate, _: Dict = require_permission("create_table")) -> UserOut:
    invalid = set(body.permissions) - VALID_PERMISSIONS
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid permissions: {invalid}")
    existing = get_user_by_username(body.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")
    row = create_user(body.username, body.password, permissions=body.permissions)
    return UserOut(**row)


@app.get("/admin/users")
def get_users(_: Dict = require_permission("create_table")) -> List[Dict[str, Any]]:
    return list_users()


@app.put("/admin/users/{user_id}")
def update_user(
    user_id: int, body: UserUpdate, _: Dict = require_permission("create_table")
) -> Dict[str, Any]:
    invalid = set(body.permissions) - VALID_PERMISSIONS
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid permissions: {invalid}")
    if not body.permissions:
        raise HTTPException(status_code=400, detail="Must have at least one permission.")
    updated = update_user_permissions(user_id, body.permissions)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return updated


@app.delete("/admin/users/{username}")
def remove_user(
    username: str, current_user: Dict = require_permission("create_table")
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
def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "The Legal Brain",
        "deploy_marker": DEPLOY_MARKER,
        "startup_db_bootstrap": RUN_STARTUP_DB_BOOTSTRAP,
    }


@app.get("/schema")
def schema_summary(_: Dict = require_permission("read")) -> Dict[str, Any]:
    return export_schema_summary()


@app.get("/ui/tables")
def get_tables(_: Dict = require_permission("read")) -> Dict[str, Any]:
    return {"tables": list_public_tables()}


@app.get("/ui/tables/{table_name}")
def get_table_rows(
    table_name: str, limit: int = 50, _: Dict = require_permission("read")
) -> Dict[str, Any]:
    try:
        return fetch_table_rows(table_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query_legal_brain(
    request: QueryRequest, _: Dict = require_permission("read")
) -> QueryResponse:
    try:
        return QueryResponse(**run_rag_query(request.query, top_k=request.top_k))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Legal query failed. Check the Azure OpenAI chat and embedding "
                "deployments for the main Legal Brain app."
            ),
        ) from exc


@app.put("/ui/tables/case/{case_id}/notes")
def update_case_note(
    case_id: int, body: CaseNoteUpdate, _: Dict = require_permission("update_records")
) -> Dict[str, Any]:
    updated = update_case_notes(case_id, body.notes)
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found.")
    return updated


# ---------------------------------------------------------------------------
# Admin-only endpoints
# ---------------------------------------------------------------------------


@app.post("/admin/init-db")
def initialize_database(_: Dict = require_permission("create_table")) -> Dict[str, str]:
    init_db()
    seed_default_users()
    return {"status": "initialized"}


@app.post("/admin/ingest")
def ingest_documents(
    request: IngestRequest, _: Dict = require_permission("write")
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
