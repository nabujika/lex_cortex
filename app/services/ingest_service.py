from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.ingestion.chunker import chunk_case_text
from app.ingestion.embedder import AzureOpenAIClient
from app.ingestion.indexer import IngestionIndexer
from app.ingestion.parser import parse_pdf_document
from app.ingestion.workspace_discovery import WorkspaceFiles, discover_workspace_files
from app.retrievers.azure_search import AzureSearchIndexer
from app.utils.hashing import stable_hash

logger = get_logger(__name__)


class IngestService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self._llm: AzureOpenAIClient | None = None
        self._search_indexer: AzureSearchIndexer | None = None
        self._indexer: IngestionIndexer | None = None

    @property
    def llm(self) -> AzureOpenAIClient:
        if self._llm is None:
            self._llm = AzureOpenAIClient()
        return self._llm

    @property
    def search_indexer(self) -> AzureSearchIndexer:
        if self._search_indexer is None:
            self._search_indexer = AzureSearchIndexer()
        return self._search_indexer

    @property
    def indexer(self) -> IngestionIndexer:
        if self._indexer is None:
            self._indexer = IngestionIndexer(db=self.db, search_indexer=self.search_indexer)
        return self._indexer

    def discover_files(self, pdf_dir: str | None = None) -> WorkspaceFiles:
        return discover_workspace_files(pdf_dir=pdf_dir)

    def ingest_workspace(self, pdf_dir: str | None = None) -> dict:
        workspace_files = self.discover_files(pdf_dir=pdf_dir)
        ingested_cases: list[dict] = []
        for pdf_path in workspace_files.pdf_files:
            ingested_cases.append(self.ingest_pdf(pdf_path))
        return {
            "workspace_root": str(workspace_files.workspace_root),
            "pdf_count": len(workspace_files.pdf_files),
            "diagram_count": len(workspace_files.diagram_files),
            "ingested_cases": ingested_cases,
        }

    def ingest_pdf(self, pdf_path: Path) -> dict:
        logger.info("Parsing %s", pdf_path)
        parsed = parse_pdf_document(pdf_path)
        case_id = stable_hash("case", parsed.metadata.title, pdf_path.resolve())
        page_payload = [{"page_number": page.page_number, "text": page.text} for page in parsed.pages]
        chunks = chunk_case_text(case_id=case_id, pages=page_payload)
        vectors = self.llm.embed_texts([chunk.chunk_text for chunk in chunks]) if chunks else []

        chunk_payloads = []
        for chunk, vector in zip(chunks, vectors):
            chunk_payloads.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                    "metadata": chunk.metadata,
                    "embedding_vector": vector,
                }
            )

        metadata = {
            "court_name": parsed.metadata.court_name,
            "judge_name": parsed.metadata.judge_name,
            "judgment_date": parsed.metadata.judgment_date,
            "filing_date": parsed.metadata.filing_date,
            "status": parsed.metadata.status,
            **parsed.metadata.raw_fields,
        }
        self.indexer.persist_case(
            case_id=case_id,
            title=parsed.metadata.title,
            source_file=str(pdf_path.resolve()),
            raw_text=parsed.full_text,
            metadata=metadata,
            chunks=chunk_payloads,
        )
        return {
            "case_id": case_id,
            "title": parsed.metadata.title,
            "source_file": str(pdf_path.resolve()),
            "chunk_count": len(chunk_payloads),
        }
