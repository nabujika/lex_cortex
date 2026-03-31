from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "legal-rag-azure"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    workspace_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    pdf_data_dir: str | None = None

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "legal_rag"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_sslmode: str | None = None
    postgres_sslrootcert: str | None = None

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""
    azure_openai_embedding_batch_size: int = 16
    azure_openai_max_retries: int = 6
    azure_openai_retry_base_seconds: float = 5.0

    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "legal-case-chunks"
    azure_search_vector_dimensions: int = 3072

    chunk_size: int = 1400
    chunk_overlap: int = 250
    retrieval_k: int = 8
    retrieval_min_score: float = 0.15
    graded_retrieval_min_relevance: float = 0.45
    rewrite_min_results: int = 2

    @property
    def sqlalchemy_database_uri(self) -> str:
        query: dict[str, str] = {}
        if self.postgres_sslmode:
            query["sslmode"] = self.postgres_sslmode
        if self.postgres_sslrootcert:
            query["sslrootcert"] = self.postgres_sslrootcert

        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
            query=query,
        ).render_as_string(hide_password=False)

    @property
    def resolved_workspace_root(self) -> Path:
        return self.workspace_root.resolve()

    def resolve_workspace_path(self, candidate: str | Path | None) -> Path | None:
        if candidate is None:
            return None
        path = Path(candidate)
        if path.is_absolute():
            return path
        return (self.resolved_workspace_root / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
