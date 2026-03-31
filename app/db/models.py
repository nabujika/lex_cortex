from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Court(TimestampMixin, Base):
    __tablename__ = "courts"

    court_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    jurisdiction: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[str | None] = mapped_column(String(128))

    cases: Mapped[list["Case"]] = relationship(back_populates="court")


class Judge(TimestampMixin, Base):
    __tablename__ = "judges"

    judge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    appointed_date: Mapped[date | None] = mapped_column(Date)
    specialization: Mapped[str | None] = mapped_column(String(255))

    cases: Mapped[list["Case"]] = relationship(back_populates="presiding_judge")


class Statute(TimestampMixin, Base):
    __tablename__ = "statutes"

    statute_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    short_title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    act_number: Mapped[str | None] = mapped_column(String(128))

    versions: Mapped[list["StatuteVersion"]] = relationship(back_populates="statute", cascade="all, delete-orphan")
    case_references: Mapped[list["CaseStatuteReference"]] = relationship(back_populates="statute")


class StatuteVersion(TimestampMixin, Base):
    __tablename__ = "statute_versions"
    __table_args__ = (
        Index("ix_statute_versions_statute_id_active", "statute_id", "is_active"),
    )

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    statute_id: Mapped[str] = mapped_column(ForeignKey("statutes.statute_id", ondelete="CASCADE"), nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    amendment_date: Mapped[date | None] = mapped_column(Date)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    statute: Mapped["Statute"] = relationship(back_populates="versions")
    case_references: Mapped[list["CaseStatuteReference"]] = relationship(back_populates="statute_version")


class Case(TimestampMixin, Base):
    __tablename__ = "cases"
    __table_args__ = (
        Index("ix_cases_title_judgment_date", "title", "judgment_date"),
    )

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    filing_date: Mapped[date | None] = mapped_column(Date)
    judgment_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str | None] = mapped_column(String(128), index=True)
    source_file: Mapped[str | None] = mapped_column(String(1024), unique=True)
    raw_text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    court_id: Mapped[str | None] = mapped_column(ForeignKey("courts.court_id", ondelete="SET NULL"))
    presiding_judge_id: Mapped[str | None] = mapped_column(ForeignKey("judges.judge_id", ondelete="SET NULL"))

    court: Mapped["Court | None"] = relationship(back_populates="cases")
    presiding_judge: Mapped["Judge | None"] = relationship(back_populates="cases")
    chunks: Mapped[list["CaseChunk"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    outgoing_precedents: Mapped[list["CasePrecedent"]] = relationship(
        back_populates="source_case",
        foreign_keys="CasePrecedent.source_case_id",
        cascade="all, delete-orphan",
    )
    incoming_precedents: Mapped[list["CasePrecedent"]] = relationship(
        back_populates="target_case",
        foreign_keys="CasePrecedent.target_case_id",
    )
    statute_references: Mapped[list["CaseStatuteReference"]] = relationship(back_populates="case")


class CaseChunk(TimestampMixin, Base):
    __tablename__ = "case_chunks"
    __table_args__ = (
        Index("ix_case_chunks_case_id_page_number", "case_id", "page_number"),
        Index("ix_case_chunks_chunk_type", "chunk_type"),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vector: Mapped[list[float] | None] = mapped_column(JSON)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    metadata_json: Mapped[dict | None] = mapped_column(JSON)

    case: Mapped["Case"] = relationship(back_populates="chunks")
    precedent_contexts: Mapped[list["CasePrecedent"]] = relationship(back_populates="citation_context_chunk")
    statute_contexts: Mapped[list["CaseStatuteReference"]] = relationship(back_populates="context_chunk")


class CasePrecedent(TimestampMixin, Base):
    __tablename__ = "case_precedents"
    __table_args__ = (
        UniqueConstraint("source_case_id", "target_case_id", "citation_context_chunk_id", name="uq_case_precedent_ref"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False)
    target_case_id: Mapped[str | None] = mapped_column(ForeignKey("cases.case_id", ondelete="SET NULL"))
    citation_context_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("case_chunks.chunk_id", ondelete="SET NULL"))
    citation_text: Mapped[str | None] = mapped_column(Text)

    source_case: Mapped["Case"] = relationship(back_populates="outgoing_precedents", foreign_keys=[source_case_id])
    target_case: Mapped["Case | None"] = relationship(back_populates="incoming_precedents", foreign_keys=[target_case_id])
    citation_context_chunk: Mapped["CaseChunk | None"] = relationship(back_populates="precedent_contexts")


class CaseStatuteReference(TimestampMixin, Base):
    __tablename__ = "case_statute_references"
    __table_args__ = (
        Index("ix_case_statute_reference_case_statute", "case_id", "statute_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False)
    statute_id: Mapped[str | None] = mapped_column(ForeignKey("statutes.statute_id", ondelete="SET NULL"))
    statute_version_id: Mapped[str | None] = mapped_column(ForeignKey("statute_versions.version_id", ondelete="SET NULL"))
    context_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("case_chunks.chunk_id", ondelete="SET NULL"))

    case: Mapped["Case"] = relationship(back_populates="statute_references")
    statute: Mapped["Statute | None"] = relationship(back_populates="case_references")
    statute_version: Mapped["StatuteVersion | None"] = relationship(back_populates="case_references")
    context_chunk: Mapped["CaseChunk | None"] = relationship(back_populates="statute_contexts")

