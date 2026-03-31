import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from dotenv import load_dotenv
from openai import AzureOpenAI
from psycopg.rows import dict_row

load_dotenv()


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_embedding_dimensions() -> int:
    return int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))


SCHEMA_SQL_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS court (
    court_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    jurisdiction TEXT,
    state TEXT,
    UNIQUE (name, jurisdiction, state)
);

CREATE TABLE IF NOT EXISTS judge (
    judge_id BIGSERIAL PRIMARY KEY,
    full_name TEXT NOT NULL UNIQUE,
    appointed_date DATE,
    specialization TEXT
);

CREATE TABLE IF NOT EXISTS "case" (
    case_id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    filing_date DATE,
    judgment_date DATE,
    status TEXT,
    court_id BIGINT REFERENCES court(court_id),
    presiding_judge_id BIGINT REFERENCES judge(judge_id),
    UNIQUE (title, judgment_date)
);

CREATE TABLE IF NOT EXISTS statute (
    statute_id BIGSERIAL PRIMARY KEY,
    short_title TEXT NOT NULL,
    act_number TEXT,
    UNIQUE (short_title, act_number)
);

CREATE TABLE IF NOT EXISTS case_chunk (
    chunk_id BIGSERIAL PRIMARY KEY,
    case_id BIGINT NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_type TEXT,
    page_number INTEGER
);

CREATE TABLE IF NOT EXISTS statute_version (
    version_id BIGSERIAL PRIMARY KEY,
    statute_id BIGINT NOT NULL REFERENCES statute(statute_id) ON DELETE CASCADE,
    full_text TEXT NOT NULL,
    amendment_date DATE,
    valid_from DATE,
    valid_to DATE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS case_precedent (
    case_id BIGINT NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    precedent_case_id BIGINT NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    PRIMARY KEY (case_id, precedent_case_id),
    CHECK (case_id <> precedent_case_id)
);

CREATE TABLE IF NOT EXISTS case_statute (
    case_id BIGINT NOT NULL REFERENCES "case"(case_id) ON DELETE CASCADE,
    statute_id BIGINT NOT NULL REFERENCES statute(statute_id) ON DELETE CASCADE,
    PRIMARY KEY (case_id, statute_id)
);

CREATE TABLE IF NOT EXISTS embedding_store (
    chunk_id BIGINT PRIMARY KEY REFERENCES case_chunk(chunk_id) ON DELETE CASCADE,
    embedding VECTOR(%(embedding_dimensions)s) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_case_court_id ON "case"(court_id);
CREATE INDEX IF NOT EXISTS idx_case_presiding_judge_id ON "case"(presiding_judge_id);
CREATE INDEX IF NOT EXISTS idx_case_dates ON "case"(filing_date, judgment_date);
CREATE INDEX IF NOT EXISTS idx_case_chunk_case_id ON case_chunk(case_id);
CREATE INDEX IF NOT EXISTS idx_statute_version_statute_id ON statute_version(statute_id);
CREATE INDEX IF NOT EXISTS idx_case_precedent_case_id ON case_precedent(case_id);
CREATE INDEX IF NOT EXISTS idx_case_precedent_precedent_case_id ON case_precedent(precedent_case_id);
CREATE INDEX IF NOT EXISTS idx_case_statute_case_id ON case_statute(case_id);
CREATE INDEX IF NOT EXISTS idx_case_statute_statute_id ON case_statute(statute_id);
CREATE INDEX IF NOT EXISTS idx_embedding_store_hnsw
ON embedding_store
USING hnsw (embedding vector_cosine_ops);
"""


def get_schema_sql() -> str:
    return SCHEMA_SQL_TEMPLATE % {"embedding_dimensions": get_embedding_dimensions()}


def get_db_connection() -> psycopg.Connection:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg.connect(database_url, row_factory=dict_row)

    return psycopg.connect(
        host=get_env("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=get_env("POSTGRES_DB"),
        user=get_env("POSTGRES_USER"),
        password=get_env("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        row_factory=dict_row,
    )


@contextmanager
def db_cursor():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            yield conn, cur
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    # Applies the full relational + vector schema in one place for local runs and cloud deployments.
    with db_cursor() as (_, cur):
        cur.execute(get_schema_sql())


def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=get_env("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
    )


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def upsert_court(cur: psycopg.Cursor, name: str, jurisdiction: Optional[str] = None, state: Optional[str] = None) -> int:
    cur.execute(
        """
        SELECT court_id
        FROM court
        WHERE name = %s
          AND jurisdiction IS NOT DISTINCT FROM %s
          AND state IS NOT DISTINCT FROM %s
        """,
        (name, jurisdiction, state),
    )
    existing = cur.fetchone()
    if existing:
        return int(existing["court_id"])

    cur.execute(
        """
        INSERT INTO court (name, jurisdiction, state)
        VALUES (%s, %s, %s)
        RETURNING court_id
        """,
        (name, jurisdiction, state),
    )
    return int(cur.fetchone()["court_id"])


def upsert_judge(
    cur: psycopg.Cursor,
    full_name: str,
    appointed_date: Optional[str] = None,
    specialization: Optional[str] = None,
) -> int:
    cur.execute(
        """
        INSERT INTO judge (full_name, appointed_date, specialization)
        VALUES (%s, %s, %s)
        ON CONFLICT (full_name)
        DO UPDATE SET
            appointed_date = COALESCE(judge.appointed_date, EXCLUDED.appointed_date),
            specialization = COALESCE(judge.specialization, EXCLUDED.specialization)
        RETURNING judge_id
        """,
        (full_name, appointed_date, specialization),
    )
    return int(cur.fetchone()["judge_id"])


def upsert_case(
    cur: psycopg.Cursor,
    title: str,
    filing_date: Optional[str],
    judgment_date: Optional[str],
    status: Optional[str],
    court_id: Optional[int],
    presiding_judge_id: Optional[int],
) -> int:
    cur.execute(
        """
        SELECT case_id
        FROM "case"
        WHERE title = %s
          AND judgment_date IS NOT DISTINCT FROM %s
        """,
        (title, judgment_date),
    )
    existing = cur.fetchone()
    if existing:
        cur.execute(
            """
            UPDATE "case"
            SET
                filing_date = COALESCE("case".filing_date, %s),
                status = COALESCE(%s, "case".status),
                court_id = COALESCE(%s, "case".court_id),
                presiding_judge_id = COALESCE(%s, "case".presiding_judge_id)
            WHERE case_id = %s
            RETURNING case_id
            """,
            (filing_date, status, court_id, presiding_judge_id, existing["case_id"]),
        )
        return int(cur.fetchone()["case_id"])

    cur.execute(
        """
        INSERT INTO "case" (title, filing_date, judgment_date, status, court_id, presiding_judge_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING case_id
        """,
        (title, filing_date, judgment_date, status, court_id, presiding_judge_id),
    )
    return int(cur.fetchone()["case_id"])


def replace_case_chunks(cur: psycopg.Cursor, case_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
    cur.execute("DELETE FROM case_chunk WHERE case_id = %s", (case_id,))
    chunk_ids: List[int] = []
    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO case_chunk (case_id, chunk_text, chunk_type, page_number)
            VALUES (%s, %s, %s, %s)
            RETURNING chunk_id
            """,
            (case_id, chunk["chunk_text"], chunk.get("chunk_type"), chunk.get("page_number")),
        )
        chunk_ids.append(int(cur.fetchone()["chunk_id"]))
    return chunk_ids


def upsert_chunk_embeddings(cur: psycopg.Cursor, chunk_ids: Sequence[int], embeddings: Sequence[Sequence[float]]) -> None:
    for chunk_id, embedding in zip(chunk_ids, embeddings):
        cur.execute(
            """
            INSERT INTO embedding_store (chunk_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (chunk_id)
            DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (chunk_id, vector_literal(embedding)),
        )


def upsert_statute(cur: psycopg.Cursor, short_title: str, act_number: Optional[str]) -> int:
    cur.execute(
        """
        SELECT statute_id
        FROM statute
        WHERE short_title = %s
          AND act_number IS NOT DISTINCT FROM %s
        """,
        (short_title, act_number),
    )
    existing = cur.fetchone()
    if existing:
        return int(existing["statute_id"])

    cur.execute(
        """
        INSERT INTO statute (short_title, act_number)
        VALUES (%s, %s)
        RETURNING statute_id
        """,
        (short_title, act_number),
    )
    return int(cur.fetchone()["statute_id"])


def replace_statute_versions(
    cur: psycopg.Cursor,
    statute_id: int,
    full_text: str,
    amendment_date: Optional[str],
    valid_from: Optional[str],
    valid_to: Optional[str],
    is_active: bool,
) -> int:
    cur.execute("DELETE FROM statute_version WHERE statute_id = %s", (statute_id,))
    cur.execute(
        """
        INSERT INTO statute_version (statute_id, full_text, amendment_date, valid_from, valid_to, is_active)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING version_id
        """,
        (statute_id, full_text, amendment_date, valid_from, valid_to, is_active),
    )
    return int(cur.fetchone()["version_id"])


def link_case_to_statutes(cur: psycopg.Cursor, case_id: int, full_text: str) -> None:
    cur.execute("SELECT statute_id, short_title, COALESCE(act_number, '') AS act_number FROM statute")
    statutes = cur.fetchall()
    normalized_text = full_text.lower()
    for statute in statutes:
        title = statute["short_title"].lower()
        act_number = statute["act_number"].lower()
        if title in normalized_text or (act_number and act_number in normalized_text):
            cur.execute(
                """
                INSERT INTO case_statute (case_id, statute_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (case_id, statute["statute_id"]),
            )


def search_similar_case_chunks(
    query_embedding: Sequence[float],
    limit: int = 8,
    judge: Optional[str] = None,
    court: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Hybrid retrieval starts here: semantic match first, relational enrichment happens later.
    where_clauses = []
    params: List[Any] = [vector_literal(query_embedding), vector_literal(query_embedding)]

    if judge:
        where_clauses.append("j.full_name ILIKE %s")
        params.append(f"%{judge}%")
    if court:
        where_clauses.append("co.name ILIKE %s")
        params.append(f"%{court}%")
    if date_from:
        where_clauses.append('c.judgment_date >= %s')
        params.append(date_from)
    if date_to:
        where_clauses.append('c.judgment_date <= %s')
        params.append(date_to)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)

    query = f"""
        SELECT
            cc.chunk_id,
            cc.case_id,
            cc.chunk_text,
            cc.chunk_type,
            cc.page_number,
            c.title,
            c.judgment_date,
            j.full_name AS judge_name,
            co.name AS court_name,
            1 - (es.embedding <=> %s::vector) AS similarity
        FROM embedding_store es
        JOIN case_chunk cc ON cc.chunk_id = es.chunk_id
        JOIN "case" c ON c.case_id = cc.case_id
        LEFT JOIN judge j ON j.judge_id = c.presiding_judge_id
        LEFT JOIN court co ON co.court_id = c.court_id
        {where_sql}
        ORDER BY es.embedding <=> %s::vector
        LIMIT %s
    """

    with db_cursor() as (_, cur):
        cur.execute(query, params)
        return list(cur.fetchall())


def fetch_case_details(case_ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not case_ids:
        return []

    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT
                c.case_id,
                c.title,
                c.filing_date,
                c.judgment_date,
                c.status,
                j.full_name AS judge_name,
                co.name AS court_name,
                co.jurisdiction,
                co.state,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT s.short_title), NULL) AS statutes
            FROM "case" c
            LEFT JOIN judge j ON j.judge_id = c.presiding_judge_id
            LEFT JOIN court co ON co.court_id = c.court_id
            LEFT JOIN case_statute cs ON cs.case_id = c.case_id
            LEFT JOIN statute s ON s.statute_id = cs.statute_id
            WHERE c.case_id = ANY(%s)
            GROUP BY c.case_id, j.full_name, co.name, co.jurisdiction, co.state
            ORDER BY c.judgment_date DESC NULLS LAST
            """,
            (list(case_ids),),
        )
        return list(cur.fetchall())


def fetch_relevant_statutes(statute_name: Optional[str] = None, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
    conditions = []
    params: List[Any] = []
    if statute_name:
        conditions.append("s.short_title ILIKE %s")
        params.append(f"%{statute_name}%")
    if as_of_date:
        conditions.append("(sv.valid_from IS NULL OR sv.valid_from <= %s) AND (sv.valid_to IS NULL OR sv.valid_to >= %s)")
        params.extend([as_of_date, as_of_date])
    else:
        conditions.append("sv.is_active = TRUE")

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with db_cursor() as (_, cur):
        cur.execute(
            f"""
            SELECT
                sv.version_id,
                s.statute_id,
                s.short_title,
                s.act_number,
                sv.amendment_date,
                sv.valid_from,
                sv.valid_to,
                sv.is_active,
                LEFT(sv.full_text, 4000) AS full_text_excerpt
            FROM statute_version sv
            JOIN statute s ON s.statute_id = sv.statute_id
            {where_sql}
            ORDER BY sv.valid_from DESC NULLS LAST, sv.amendment_date DESC NULLS LAST
            LIMIT 5
            """,
            params,
        )
        return list(cur.fetchall())


def fetch_precedents(case_ids: Sequence[int]) -> Dict[int, List[Dict[str, Any]]]:
    if not case_ids:
        return {}

    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT
                cp.case_id,
                p.case_id AS precedent_case_id,
                p.title,
                p.judgment_date
            FROM case_precedent cp
            JOIN "case" p ON p.case_id = cp.precedent_case_id
            WHERE cp.case_id = ANY(%s)
            ORDER BY p.judgment_date DESC NULLS LAST
            """,
            (list(case_ids),),
        )
        result: Dict[int, List[Dict[str, Any]]] = {}
        for row in cur.fetchall():
            result.setdefault(int(row["case_id"]), []).append(
                {
                    "case_id": int(row["precedent_case_id"]),
                    "title": row["title"],
                    "judgment_date": row["judgment_date"],
                }
            )
        return result


def export_schema_summary() -> Dict[str, Any]:
    return {"embedding_dimensions": get_embedding_dimensions(), "schema_sql": get_schema_sql()}


def dump_json(data: Any) -> str:
    return json.dumps(data, default=str, indent=2)
