from __future__ import annotations

import re
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Case, CaseChunk, Court, Judge, Statute, StatuteVersion
from app.ingestion.metadata_extractor import infer_court_name_from_text, infer_judge_name_from_text

STOPWORDS = {
    "which",
    "court",
    "heard",
    "case",
    "judge",
    "what",
    "when",
    "the",
    "was",
    "is",
    "of",
    "for",
    "did",
    "in",
    "on",
    "a",
    "an",
    "to",
    "status",
    "filing",
    "judgment",
    "date",
    "show",
}


class SQLLookupTool:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _question_tokens(self, question: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", question.lower())
        return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]

    def _rank_case_rows(self, rows: list[dict], question: str) -> list[dict]:
        tokens = self._question_tokens(question)
        if not tokens:
            return rows[:10]

        query_text = " ".join(tokens)
        ranked: list[dict] = []
        for row in rows:
            title = str(row.get("title") or "").lower()
            token_hits = sum(1 for token in tokens if token in title)
            ratio = SequenceMatcher(None, query_text, title).ratio()
            score = token_hits + ratio
            if token_hits > 0 or ratio >= 0.22:
                ranked.append({**row, "match_score": round(score, 4)})

        ranked.sort(key=lambda item: item.get("match_score", 0), reverse=True)
        return ranked[:10] if ranked else rows[:10]

    def run(self, question: str) -> list[dict]:
        lowered = question.lower()
        if "which court" in lowered or "heard case" in lowered:
            rows = []
            for row in self.db.execute(
                select(Case.case_id, Case.title, Court.name.label("court_name"), Case.raw_text)
                    .join(Court, Case.court_id == Court.court_id, isouter=True)
                    .order_by(Case.judgment_date.desc().nullslast())
                ):
                item = dict(row._mapping)
                raw_text = item.pop("raw_text", None)
                if not item.get("court_name") or "citation" in str(item.get("court_name", "")).lower():
                    item["court_name"] = infer_court_name_from_text(raw_text or "") or item.get("court_name")
                rows.append(item)
            return self._rank_case_rows(rows, question)
        if "judge" in lowered:
            rows = []
            for row in self.db.execute(
                select(Case.case_id, Case.title, Judge.full_name.label("judge_name"), Case.raw_text)
                    .join(Judge, Case.presiding_judge_id == Judge.judge_id, isouter=True)
                    .order_by(Case.judgment_date.desc().nullslast())
                ):
                item = dict(row._mapping)
                raw_text = item.pop("raw_text", None)
                if not item.get("judge_name"):
                    item["judge_name"] = infer_judge_name_from_text(raw_text or "")
                rows.append(item)
            return self._rank_case_rows(rows, question)
        if "filing date" in lowered or "judgment date" in lowered or "status" in lowered:
            rows = [
                dict(row._mapping)
                for row in self.db.execute(
                    select(Case.case_id, Case.title, Case.status, Case.filing_date, Case.judgment_date)
                    .order_by(Case.judgment_date.desc().nullslast())
                )
            ]
            return self._rank_case_rows(rows, question)
        if "statute version" in lowered or "applied on" in lowered:
            stmt = (
                select(
                    Statute.short_title,
                    StatuteVersion.version_id,
                    StatuteVersion.valid_from,
                    StatuteVersion.valid_to,
                    StatuteVersion.is_active,
                )
                .join(StatuteVersion, Statute.statute_id == StatuteVersion.statute_id)
                .limit(10)
            )
        else:
            stmt = (
                select(Case.case_id, Case.title, Case.status, Case.judgment_date)
                .join(Court, Case.court_id == Court.court_id, isouter=True)
                .order_by(Case.judgment_date.desc().nullslast())
                .limit(10)
            )
        result = self.db.execute(stmt)
        return [dict(row._mapping) for row in result]

    def local_chunk_search(self, question: str, top_k: int, case_id: str | None = None) -> list[dict]:
        stmt = (
            select(
                CaseChunk.chunk_id,
                CaseChunk.case_id,
                Case.title,
                Case.status,
                Case.filing_date,
                Case.judgment_date,
                CaseChunk.page_number,
                CaseChunk.chunk_type,
                CaseChunk.chunk_text,
            )
            .join(Case, CaseChunk.case_id == Case.case_id)
        )
        if case_id:
            stmt = stmt.where(CaseChunk.case_id == case_id)

        rows = [dict(row._mapping) for row in self.db.execute(stmt)]
        tokens = self._question_tokens(question)
        query_text = " ".join(tokens)

        scored: list[dict] = []
        for row in rows:
            haystack = f"{row.get('title', '')} {row.get('chunk_text', '')}".lower()
            token_hits = sum(1 for token in tokens if token in haystack)
            title_ratio = SequenceMatcher(None, query_text, str(row.get("title", "")).lower()).ratio() if query_text else 0.0
            chunk_bonus = 0.35 if row.get("chunk_type") in {"ruling", "facts", "arguments", "dissent"} else 0.0
            score = token_hits + title_ratio + chunk_bonus
            if token_hits > 0 or title_ratio >= 0.18:
                scored.append({**row, "score": round(score, 4), "reranker_score": round(score, 4)})

        scored.sort(key=lambda item: item.get("score", 0), reverse=True)
        return scored[:top_k]
