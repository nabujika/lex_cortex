from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "courts",
        sa.Column("court_id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("jurisdiction", sa.String(length=128), nullable=True),
        sa.Column("state", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_courts_name", "courts", ["name"])

    op.create_table(
        "judges",
        sa.Column("judge_id", sa.String(length=64), primary_key=True),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("appointed_date", sa.Date(), nullable=True),
        sa.Column("specialization", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_judges_full_name", "judges", ["full_name"])

    op.create_table(
        "statutes",
        sa.Column("statute_id", sa.String(length=64), primary_key=True),
        sa.Column("short_title", sa.String(length=255), nullable=False),
        sa.Column("act_number", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_statutes_short_title", "statutes", ["short_title"])

    op.create_table(
        "statute_versions",
        sa.Column("version_id", sa.String(length=64), primary_key=True),
        sa.Column("statute_id", sa.String(length=64), sa.ForeignKey("statutes.statute_id", ondelete="CASCADE"), nullable=False),
        sa.Column("full_text", sa.Text(), nullable=False),
        sa.Column("amendment_date", sa.Date(), nullable=True),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_statute_versions_statute_id_active", "statute_versions", ["statute_id", "is_active"])

    op.create_table(
        "cases",
        sa.Column("case_id", sa.String(length=64), primary_key=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=True),
        sa.Column("judgment_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=128), nullable=True),
        sa.Column("source_file", sa.String(length=1024), nullable=True, unique=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("court_id", sa.String(length=64), sa.ForeignKey("courts.court_id", ondelete="SET NULL"), nullable=True),
        sa.Column("presiding_judge_id", sa.String(length=64), sa.ForeignKey("judges.judge_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cases_title", "cases", ["title"])
    op.create_index("ix_cases_status", "cases", ["status"])
    op.create_index("ix_cases_title_judgment_date", "cases", ["title", "judgment_date"])

    op.create_table(
        "case_chunks",
        sa.Column("chunk_id", sa.String(length=64), primary_key=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding_vector", sa.JSON(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_case_chunks_case_id_page_number", "case_chunks", ["case_id", "page_number"])
    op.create_index("ix_case_chunks_chunk_type", "case_chunks", ["chunk_type"])

    op.create_table(
        "case_precedents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="SET NULL"), nullable=True),
        sa.Column("citation_context_chunk_id", sa.String(length=64), sa.ForeignKey("case_chunks.chunk_id", ondelete="SET NULL"), nullable=True),
        sa.Column("citation_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_case_id", "target_case_id", "citation_context_chunk_id", name="uq_case_precedent_ref"),
    )

    op.create_table(
        "case_statute_references",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("statute_id", sa.String(length=64), sa.ForeignKey("statutes.statute_id", ondelete="SET NULL"), nullable=True),
        sa.Column("statute_version_id", sa.String(length=64), sa.ForeignKey("statute_versions.version_id", ondelete="SET NULL"), nullable=True),
        sa.Column("context_chunk_id", sa.String(length=64), sa.ForeignKey("case_chunks.chunk_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_case_statute_reference_case_statute", "case_statute_references", ["case_id", "statute_id"])


def downgrade() -> None:
    op.drop_index("ix_case_statute_reference_case_statute", table_name="case_statute_references")
    op.drop_table("case_statute_references")
    op.drop_table("case_precedents")
    op.drop_index("ix_case_chunks_chunk_type", table_name="case_chunks")
    op.drop_index("ix_case_chunks_case_id_page_number", table_name="case_chunks")
    op.drop_table("case_chunks")
    op.drop_index("ix_cases_title_judgment_date", table_name="cases")
    op.drop_index("ix_cases_status", table_name="cases")
    op.drop_index("ix_cases_title", table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_statute_versions_statute_id_active", table_name="statute_versions")
    op.drop_table("statute_versions")
    op.drop_index("ix_statutes_short_title", table_name="statutes")
    op.drop_table("statutes")
    op.drop_index("ix_judges_full_name", table_name="judges")
    op.drop_table("judges")
    op.drop_index("ix_courts_name", table_name="courts")
    op.drop_table("courts")

