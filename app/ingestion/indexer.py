from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Case, CaseChunk, Court, Judge
from app.retrievers.azure_search import AzureSearchIndexer
from app.utils.hashing import stable_hash

logger = get_logger(__name__)


def _to_jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


class IngestionIndexer:
    def __init__(self, db: Session, search_indexer: AzureSearchIndexer) -> None:
        self.db = db
        self.search_indexer = search_indexer

    def _upsert_court(self, court_name: str | None) -> Court | None:
        if not court_name:
            return None
        court_id = stable_hash("court", court_name)
        court = self.db.get(Court, court_id)
        if court is None:
            court = Court(court_id=court_id, name=court_name)
            self.db.add(court)
        return court

    def _upsert_judge(self, judge_name: str | None) -> Judge | None:
        if not judge_name:
            return None
        judge_id = stable_hash("judge", judge_name)
        judge = self.db.get(Judge, judge_id)
        if judge is None:
            judge = Judge(judge_id=judge_id, full_name=judge_name)
            self.db.add(judge)
        return judge

    def persist_case(
        self,
        case_id: str,
        title: str,
        source_file: str,
        raw_text: str,
        metadata: dict,
        chunks: Iterable[dict],
    ) -> Case:
        court = self._upsert_court(metadata.get("court_name"))
        judge = self._upsert_judge(metadata.get("judge_name"))

        case = self.db.get(Case, case_id)
        if case is None:
            case = Case(case_id=case_id, title=title)
            self.db.add(case)

        case.title = title
        case.source_file = source_file
        case.raw_text = raw_text
        case.judgment_date = metadata.get("judgment_date")
        case.filing_date = metadata.get("filing_date")
        case.status = metadata.get("status")
        case.metadata_json = _to_jsonable(metadata)
        case.court = court
        case.presiding_judge = judge

        self.db.flush()
        self.db.query(CaseChunk).filter(CaseChunk.case_id == case_id).delete()

        chunk_models: list[CaseChunk] = []
        search_docs: list[dict] = []
        for chunk in chunks:
            chunk_model = CaseChunk(
                chunk_id=chunk["chunk_id"],
                case_id=case.case_id,
                chunk_text=chunk["chunk_text"],
                embedding_vector=chunk.get("embedding_vector"),
                page_number=chunk["page_number"],
                chunk_type=chunk["chunk_type"],
                metadata_json=_to_jsonable(chunk.get("metadata")),
            )
            chunk_models.append(chunk_model)
            search_docs.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "case_id": case.case_id,
                    "title": case.title,
                    "court_name": court.name if court else None,
                    "judge_name": judge.full_name if judge else None,
                    "status": case.status,
                    "filing_date": case.filing_date.isoformat() if case.filing_date else None,
                    "judgment_date": case.judgment_date.isoformat() if case.judgment_date else None,
                    "page_number": chunk["page_number"],
                    "chunk_type": chunk["chunk_type"],
                    "chunk_text": chunk["chunk_text"],
                    "embedding_vector": chunk.get("embedding_vector"),
                }
            )

        self.db.add_all(chunk_models)
        self.db.commit()
        self.search_indexer.upload_documents(search_docs)
        logger.info("Persisted case %s with %s chunks", case.title, len(search_docs))
        return case
