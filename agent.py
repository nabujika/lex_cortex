import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, cli, function_tool
from livekit.plugins import openai, silero

from db import (
    create_user,
    delete_user,
    get_user_by_username,
    list_users,
    update_user_permissions,
)
from rag_graph import run_rag_query

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.local", override=True)

logger = logging.getLogger("legal-brain-admin-agent")
logger.setLevel(logging.INFO)

VALID_PERMISSIONS = {"read", "write", "create_table", "update_records"}
AGENT_NAME = os.getenv("LIVEKIT_ADMIN_AGENT_NAME", "legal-brain-admin-agent")


def get_optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


AZURE_API_VERSION = get_optional_env(
    "AZURE_VOICE_OPENAI_API_VERSION",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_VERSION",
) or "2024-10-21"
DEFAULT_AZURE_STT_DEPLOYMENT = get_optional_env(
    "AZURE_VOICE_OPENAI_STT_DEPLOYMENT",
    "AZURE_OPENAI_STT_DEPLOYMENT",
) or "gpt-4o-mini-transcribe"
DEFAULT_AZURE_TTS_DEPLOYMENT = get_optional_env(
    "AZURE_VOICE_OPENAI_TTS_DEPLOYMENT",
    "AZURE_OPENAI_TTS_DEPLOYMENT",
) or "gpt-4o-mini-tts"

SYSTEM_PROMPT = """
You are The Legal Brain Admin Voice Agent.

Your job is to help an authenticated admin manage application users over voice.

Primary tasks:
- Create new users in the database.
- List existing users and their permissions.
- Update a user's permissions.
- Delete a user when the admin explicitly confirms it.
- Answer grounded legal research questions by using the Legal Brain retrieval pipeline.

Rules:
- Be concise, operational, and calm.
- Before calling create_database_user, make sure you have the username, password, and permissions.
- Before calling update_database_user_permissions, make sure you have the username and final permissions.
- Before calling delete_database_user, make sure you have the username.
- When the admin asks a legal research question, call run_legal_research instead of answering from memory.
- For create, update, and delete actions, read back the action and ask for explicit confirmation before calling the tool.
- Never speak or repeat the raw password back aloud. Confirm it as "password received".
- Never invent permissions. The only valid permissions are read, write, create_table, and update_records.
- Ensure read permission is present. If the admin does not mention it, treat it as included automatically.
- Never mention hashed passwords or any secret values.
- When listing users, summarize usernames and permissions only.
- Never delete the built-in admin account.
- If the tool returns success=false, explain the tool message clearly and ask whether they want to try again.
- Keep responses short unless clarification is required.
""".strip()


@dataclass
class AdminSessionState:
    room_name: str


def normalize_permissions(permissions: list[str] | None) -> tuple[list[str], list[str]]:
    normalized_permissions: list[str] = []
    for permission in permissions or []:
        value = str(permission).strip()
        if value and value not in normalized_permissions:
            normalized_permissions.append(value)

    if "read" not in normalized_permissions:
        normalized_permissions.insert(0, "read")

    invalid_permissions = [
        permission
        for permission in normalized_permissions
        if permission not in VALID_PERMISSIONS
    ]
    return normalized_permissions, invalid_permissions


def serialize_user_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "permissions": list(row.get("permissions") or []),
        "created_at": str(row["created_at"]),
    }


def get_required_env(*names: str) -> str:
    value = get_optional_env(*names)
    if not value:
        raise RuntimeError(
            "Missing required environment variable. Expected one of: "
            + ", ".join(names)
        )
    return value


def build_llm() -> openai.LLM:
    return openai.LLM.with_azure(
        model=get_optional_env(
            "AZURE_VOICE_OPENAI_LLM_MODEL",
            "AZURE_OPENAI_LLM_MODEL",
        )
        or "gpt-4o-mini",
        azure_deployment=get_required_env(
            "AZURE_VOICE_OPENAI_LLM_DEPLOYMENT",
            "AZURE_OPENAI_LLM_DEPLOYMENT",
        ),
        azure_endpoint=get_required_env(
            "AZURE_VOICE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_ENDPOINT",
        ),
        api_key=get_required_env(
            "AZURE_VOICE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ),
        api_version=AZURE_API_VERSION,
    )


def build_stt() -> Any:
    azure_endpoint = get_optional_env(
        "AZURE_VOICE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
    )
    azure_api_key = get_optional_env(
        "AZURE_VOICE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )
    if azure_endpoint and azure_api_key:
        return openai.STT.with_azure(
            model=get_optional_env(
                "AZURE_VOICE_OPENAI_STT_MODEL",
                "AZURE_OPENAI_STT_MODEL",
            )
            or "gpt-4o-mini-transcribe",
            azure_deployment=DEFAULT_AZURE_STT_DEPLOYMENT,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=AZURE_API_VERSION,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return openai.STT()

    raise RuntimeError(
        "Speech-to-text is not configured. Set AZURE_VOICE_OPENAI_API_KEY and "
        "AZURE_VOICE_OPENAI_ENDPOINT for agent speech, or set OPENAI_API_KEY."
    )


def build_tts() -> Any:
    azure_endpoint = get_optional_env(
        "AZURE_VOICE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
    )
    azure_api_key = get_optional_env(
        "AZURE_VOICE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )
    if azure_endpoint and azure_api_key:
        return openai.TTS.with_azure(
            model=get_optional_env(
                "AZURE_VOICE_OPENAI_TTS_MODEL",
                "AZURE_OPENAI_TTS_MODEL",
            )
            or "gpt-4o-mini-tts",
            voice=get_optional_env(
                "AZURE_VOICE_OPENAI_TTS_VOICE",
                "AZURE_OPENAI_TTS_VOICE",
            )
            or "alloy",
            azure_deployment=DEFAULT_AZURE_TTS_DEPLOYMENT,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=AZURE_API_VERSION,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return openai.TTS()

    raise RuntimeError(
        "Text-to-speech is not configured. Set AZURE_VOICE_OPENAI_API_KEY and "
        "AZURE_VOICE_OPENAI_ENDPOINT for agent speech, or set OPENAI_API_KEY."
    )


class LegalBrainAdminAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def create_database_user(
        self,
        context: RunContext[AdminSessionState],
        username: str,
        password: str,
        permissions: list[str],
    ) -> dict[str, Any]:
        """
        Create a new application user in PostgreSQL.

        Args:
            username: New username to create.
            password: Plain-text password for the new user. Never repeat or log this value.
            permissions: Permission names to assign. Valid values are read, write, create_table, update_records.
        """
        clean_username = username.strip()
        if not clean_username:
            return {"success": False, "message": "Username is required."}

        if not password or not password.strip():
            return {"success": False, "message": "Password is required."}

        normalized_permissions, invalid_permissions = normalize_permissions(permissions)
        if invalid_permissions:
            return {
                "success": False,
                "message": (
                    "Invalid permissions requested: "
                    f"{', '.join(invalid_permissions)}. "
                    "Valid permissions are read, write, create_table, and update_records."
                ),
            }

        existing_user = await asyncio.to_thread(get_user_by_username, clean_username)
        if existing_user:
            return {
                "success": False,
                "message": f"User {clean_username} already exists.",
            }

        try:
            row = await asyncio.to_thread(
                create_user,
                clean_username,
                password,
                normalized_permissions,
            )
        except Exception:
            logger.exception(
                "Database user creation failed for username=%s permissions=%s room=%s",
                clean_username,
                normalized_permissions,
                context.userdata.room_name,
            )
            return {
                "success": False,
                "message": "I could not create that user because the database write failed.",
            }

        return {
            "success": True,
            "message": (
                f"User {row['username']} was created successfully with permissions "
                f"{', '.join(row['permissions'])}."
            ),
            "user_id": row["user_id"],
            "username": row["username"],
            "permissions": row["permissions"],
            "created_at": str(row["created_at"]),
        }

    @function_tool
    async def list_database_users(
        self,
        context: RunContext[AdminSessionState],
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        List application users with their permissions.

        Args:
            limit: Maximum number of users to return, between 1 and 50.
        """
        safe_limit = max(1, min(limit, 50))

        try:
            rows = await asyncio.to_thread(list_users)
        except Exception:
            logger.exception(
                "Listing users failed in room=%s",
                context.userdata.room_name,
            )
            return {
                "success": False,
                "message": "I could not load the user list because the database read failed.",
            }

        serialized_users = [serialize_user_row(row) for row in rows[:safe_limit]]
        total_users = len(rows)
        if not serialized_users:
            return {
                "success": True,
                "message": "There are no application users yet.",
                "count": 0,
                "users": [],
            }

        preview = ", ".join(
            f"{row['username']} ({', '.join(row['permissions'])})"
            for row in serialized_users[:5]
        )
        if total_users > 5:
            preview += ", and more."

        return {
            "success": True,
            "message": f"I found {total_users} users. {preview}",
            "count": total_users,
            "users": serialized_users,
        }

    @function_tool
    async def update_database_user_permissions(
        self,
        context: RunContext[AdminSessionState],
        username: str,
        permissions: list[str],
    ) -> dict[str, Any]:
        """
        Update an application's user permissions by username.

        Args:
            username: Username to update.
            permissions: Final permission set to apply.
        """
        clean_username = username.strip()
        if not clean_username:
            return {"success": False, "message": "Username is required."}

        normalized_permissions, invalid_permissions = normalize_permissions(permissions)
        if invalid_permissions:
            return {
                "success": False,
                "message": (
                    "Invalid permissions requested: "
                    f"{', '.join(invalid_permissions)}. "
                    "Valid permissions are read, write, create_table, and update_records."
                ),
            }

        existing_user = await asyncio.to_thread(get_user_by_username, clean_username)
        if not existing_user:
            return {
                "success": False,
                "message": f"User {clean_username} does not exist.",
            }

        current_permissions = list(existing_user.get("permissions") or [])
        if current_permissions == normalized_permissions:
            return {
                "success": True,
                "message": (
                    f"User {clean_username} already has permissions "
                    f"{', '.join(normalized_permissions)}."
                ),
                "user": serialize_user_row(existing_user),
            }

        try:
            row = await asyncio.to_thread(
                update_user_permissions,
                int(existing_user["user_id"]),
                normalized_permissions,
            )
        except Exception:
            logger.exception(
                "Updating permissions failed for username=%s permissions=%s room=%s",
                clean_username,
                normalized_permissions,
                context.userdata.room_name,
            )
            return {
                "success": False,
                "message": "I could not update that user's permissions because the database write failed.",
            }

        if not row:
            return {
                "success": False,
                "message": f"User {clean_username} could not be updated.",
            }

        serialized = serialize_user_row(row)
        return {
            "success": True,
            "message": (
                f"User {serialized['username']} now has permissions "
                f"{', '.join(serialized['permissions'])}."
            ),
            "user": serialized,
        }

    @function_tool
    async def delete_database_user(
        self,
        context: RunContext[AdminSessionState],
        username: str,
    ) -> dict[str, Any]:
        """
        Delete an application user by username.

        Args:
            username: Username to delete. Never use this for the built-in admin account.
        """
        clean_username = username.strip()
        if not clean_username:
            return {"success": False, "message": "Username is required."}

        if clean_username.lower() == "admin":
            return {
                "success": False,
                "message": "The built-in admin account cannot be deleted.",
            }

        existing_user = await asyncio.to_thread(get_user_by_username, clean_username)
        if not existing_user:
            return {
                "success": False,
                "message": f"User {clean_username} does not exist.",
            }

        try:
            deleted = await asyncio.to_thread(delete_user, clean_username)
        except Exception:
            logger.exception(
                "Deleting user failed for username=%s room=%s",
                clean_username,
                context.userdata.room_name,
            )
            return {
                "success": False,
                "message": "I could not delete that user because the database write failed.",
            }

        if not deleted:
            return {
                "success": False,
                "message": f"User {clean_username} could not be deleted.",
            }

        return {
            "success": True,
            "message": f"User {clean_username} was deleted successfully.",
            "username": clean_username,
        }

    @function_tool
    async def run_legal_research(
        self,
        context: RunContext[AdminSessionState],
        query: str,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """
        Run the Legal Brain RAG pipeline for grounded legal research.

        Args:
            query: The legal research question to answer.
            top_k: Number of semantic matches to retrieve, between 3 and 12.
        """
        clean_query = query.strip()
        if not clean_query:
            return {"success": False, "message": "A legal research question is required."}

        try:
            safe_top_k = max(3, min(int(top_k), 12))
        except (TypeError, ValueError):
            safe_top_k = 8
        logger.info(
            "Running legal research in room=%s top_k=%s query=%s",
            context.userdata.room_name,
            safe_top_k,
            clean_query[:160],
        )

        try:
            result = await asyncio.to_thread(run_rag_query, clean_query, safe_top_k)
        except Exception:
            logger.exception(
                "Legal research failed in room=%s",
                context.userdata.room_name,
            )
            return {
                "success": False,
                "message": "I could not complete that legal research request right now.",
            }

        cases_used = [
            {
                "title": str(case.get("title") or "").strip(),
                "court": str(case.get("court") or "").strip(),
                "judge": str(case.get("judge") or "").strip(),
            }
            for case in result.get("cases_used", [])
            if str(case.get("title") or "").strip()
        ]

        return {
            "success": True,
            "message": (
                f"Legal research completed with {len(cases_used)} supporting "
                f"{'case' if len(cases_used) == 1 else 'cases'}."
            ),
            "answer": result.get("answer", ""),
            "filters": result.get("filters", {}),
            "cases_used": cases_used[:8],
        }


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def run_session(ctx: JobContext) -> None:
    logger.info("Admin session started for room=%s", ctx.room.name)

    session = AgentSession(
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
        vad=silero.VAD.load(),
        userdata=AdminSessionState(room_name=ctx.room.name),
        max_tool_steps=6,
    )
    greeting_handle: Any | None = None
    greeting_task: asyncio.Task[None] | None = None

    @session.on("user_speech_committed")
    def on_user_speech(transcript: str) -> None:
        logger.info("USER: %s", transcript)

    @session.on("agent_speech_committed")
    def on_agent_speech(transcript: str) -> None:
        logger.info("AGENT: %s", transcript)

    @session.on("function_calls_collected")
    def on_tool_calls(calls: list[Any]) -> None:
        for call in calls:
            function_call = getattr(call, "function_call", None)
            name = getattr(function_call, "name", "unknown-tool")
            logger.info("TOOL CALLED: %s", name)

    @session.on("close")
    def on_session_close(event: Any) -> None:
        nonlocal greeting_handle, greeting_task
        if greeting_task is not None and not greeting_task.done():
            greeting_task.cancel()
        if greeting_handle is not None and not greeting_handle.done():
            greeting_handle.interrupt(force=True)
        logger.info(
            "Admin session closed for room=%s reason=%s",
            ctx.room.name,
            getattr(getattr(event, "reason", None), "value", "unknown"),
        )

    await session.start(
        agent=LegalBrainAdminAgent(),
        room=ctx.room,
    )

    async def deliver_greeting() -> None:
        nonlocal greeting_handle
        await asyncio.sleep(0.35)
        greeting_handle = session.say(
            (
                "Hello, you're connected to The Legal Brain admin agent. "
                "I can create a new user for you, list users, update permissions, and delete users. "
                "Tell me the username, password, and permissions when you're ready."
            ),
            allow_interruptions=False,
        )

    greeting_task = asyncio.create_task(deliver_greeting(), name="admin-agent-greeting")

    logger.info("Admin voice agent ready in room=%s", ctx.room.name)


if __name__ == "__main__":
    cli.run_app(server)
