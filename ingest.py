import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tiktoken
from dateutil import parser as date_parser
from pypdf import PdfReader

from db import (
    db_cursor,
    get_azure_openai_client,
    get_embedding_dimensions,
    link_case_to_statutes,
    replace_case_chunks,
    replace_statute_versions,
    upsert_case,
    upsert_chunk_embeddings,
    upsert_court,
    upsert_judge,
    upsert_statute,
)


@dataclass
class DocumentMetadata:
    title: str
    document_type: str
    filing_date: Optional[str] = None
    judgment_date: Optional[str] = None
    status: Optional[str] = None
    court_name: Optional[str] = None
    jurisdiction: Optional[str] = None
    state: Optional[str] = None
    judge_name: Optional[str] = None
    short_title: Optional[str] = None
    act_number: Optional[str] = None
    amendment_date: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    is_active: bool = True


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_date_from_text(value: str) -> Optional[str]:
    try:
        return date_parser.parse(value, dayfirst=True, fuzzy=True).date().isoformat()
    except Exception:
        return None


def infer_document_type(filename: str) -> str:
    lowered = filename.lower()
    statute_markers = ["constitution", "article_", "section_", "penal_code", "act_", "act "]
    return "statute" if any(marker in lowered for marker in statute_markers) else "case"


def extract_pdf_text(pdf_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = normalize_whitespace(raw_text)
        if cleaned:
            pages.append((page_number, cleaned))
    full_text = "\n".join(page_text for _, page_text in pages)
    return full_text, pages


def get_token_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.encoding_for_model("gpt-4o-mini")


def chunk_text(page_texts: Sequence[Tuple[int, str]], chunk_size: int = 450, overlap: int = 50) -> List[Dict[str, object]]:
    # Keeps chunks in the requested 400-500 token range with overlap for better recall.
    encoding = get_token_encoder()
    chunks: List[Dict[str, object]] = []
    for page_number, page_text in page_texts:
        tokens = encoding.encode(page_text)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text_value = encoding.decode(chunk_tokens).strip()
            if chunk_text_value:
                chunks.append(
                    {
                        "chunk_text": chunk_text_value,
                        "chunk_type": "judgment_excerpt",
                        "page_number": page_number,
                    }
                )
            if end >= len(tokens):
                break
            start = max(end - overlap, 0)
    return chunks


def parse_case_title_from_filename(filename: str) -> Tuple[str, Optional[str]]:
    stem = Path(filename).stem.replace("_", " ")
    parts = re.split(r"\s+on\s+", stem, maxsplit=1, flags=re.IGNORECASE)
    title = parts[0].strip()
    judgment_date = parse_date_from_text(parts[1]) if len(parts) > 1 else None
    return title, judgment_date


def parse_statute_metadata_from_filename(filename: str) -> Tuple[str, Optional[str]]:
    stem = Path(filename).stem.replace("_", " ")
    act_match = re.search(r"(\d{4})", stem)
    act_number = act_match.group(1) if act_match else None
    return stem.strip(), act_number


def extract_case_metadata_from_text(title: str, full_text: str) -> DocumentMetadata:
    preview = full_text[:4000]
    court_match = re.search(r"IN THE\s+(.{0,120}?COURT.{0,80}?)(?:\n|$)", preview, flags=re.IGNORECASE)
    judge_match = re.search(
        r"HON'?BLE\s+(?:MR\.?|MS\.?|MRS\.?)?\s*JUSTICE\s+([A-Z][A-Z\s\.\-]{3,80})",
        preview.upper(),
        flags=re.IGNORECASE,
    )
    court_name = normalize_whitespace(court_match.group(1).title()) if court_match else "Unknown Court"
    judge_name = normalize_whitespace(judge_match.group(1).title()) if judge_match else "Unknown Judge"
    state = None
    state_match = re.search(r"\b(?:OF|AT)\s+([A-Z][A-Z\s]+)$", court_name.upper())
    if state_match:
        state = state_match.group(1).title()
    return DocumentMetadata(
        title=title,
        document_type="case",
        status="judgment",
        court_name=court_name,
        jurisdiction=None,
        state=state,
        judge_name=judge_name,
    )


def extract_statute_metadata_from_text(filename: str) -> DocumentMetadata:
    short_title, act_number = parse_statute_metadata_from_filename(filename)
    return DocumentMetadata(
        title=short_title,
        document_type="statute",
        short_title=short_title,
        act_number=act_number,
        is_active=True,
    )


def build_case_metadata(pdf_path: Path, full_text: str) -> DocumentMetadata:
    title, judgment_date = parse_case_title_from_filename(pdf_path.name)
    metadata = extract_case_metadata_from_text(title, full_text)
    metadata.judgment_date = judgment_date
    return metadata


def build_statute_metadata(pdf_path: Path) -> DocumentMetadata:
    return extract_statute_metadata_from_text(pdf_path.name)


def create_embeddings(texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
    client = get_azure_openai_client()
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required for ingestion.")
    dimensions = get_embedding_dimensions()

    all_embeddings: List[List[float]] = []
    for index in range(0, len(texts), batch_size):
        batch = list(texts[index:index + batch_size])
        response = client.embeddings.create(model=deployment, input=batch, dimensions=dimensions)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def ingest_case_pdf(pdf_path: Path, full_text: str, page_texts: Sequence[Tuple[int, str]]) -> Dict[str, object]:
    # Cases are chunked and embedded because CASE_CHUNK + embedding_store back the vector search path.
    metadata = build_case_metadata(pdf_path, full_text)
    chunks = chunk_text(page_texts)
    embeddings = create_embeddings([chunk["chunk_text"] for chunk in chunks]) if chunks else []

    with db_cursor() as (_, cur):
        court_id = upsert_court(cur, metadata.court_name or "Unknown Court", metadata.jurisdiction, metadata.state)
        judge_id = upsert_judge(cur, metadata.judge_name or "Unknown Judge")
        case_id = upsert_case(
            cur,
            title=metadata.title,
            filing_date=metadata.filing_date,
            judgment_date=metadata.judgment_date,
            status=metadata.status,
            court_id=court_id,
            presiding_judge_id=judge_id,
        )
        chunk_ids = replace_case_chunks(cur, case_id, chunks)
        upsert_chunk_embeddings(cur, chunk_ids, embeddings)
        link_case_to_statutes(cur, case_id, full_text)

    return {
        "document_type": "case",
        "path": str(pdf_path),
        "title": metadata.title,
        "case_id": case_id,
        "chunks_indexed": len(chunks),
    }


def ingest_statute_pdf(pdf_path: Path, full_text: str) -> Dict[str, object]:
    # Statutes are stored in versioned form to support temporal retrieval by validity windows.
    metadata = build_statute_metadata(pdf_path)
    with db_cursor() as (_, cur):
        statute_id = upsert_statute(cur, metadata.short_title or metadata.title, metadata.act_number)
        version_id = replace_statute_versions(
            cur,
            statute_id=statute_id,
            full_text=full_text,
            amendment_date=metadata.amendment_date,
            valid_from=metadata.valid_from,
            valid_to=metadata.valid_to,
            is_active=metadata.is_active,
        )
    return {
        "document_type": "statute",
        "path": str(pdf_path),
        "title": metadata.short_title or metadata.title,
        "statute_id": statute_id,
        "version_id": version_id,
    }


def ingest_pdf(pdf_path: Path) -> Dict[str, object]:
    full_text, page_texts = extract_pdf_text(pdf_path)
    if not full_text:
        return {"path": str(pdf_path), "status": "skipped", "reason": "No extractable text found in PDF."}
    document_type = infer_document_type(pdf_path.name)
    if document_type == "statute":
        return ingest_statute_pdf(pdf_path, full_text)
    return ingest_case_pdf(pdf_path, full_text, page_texts)


def ingest_directory(directory: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    pdfs = sorted(directory.glob("*.PDF")) + sorted(directory.glob("*.pdf"))
    ordered_pdfs = sorted(pdfs, key=lambda path: 0 if infer_document_type(path.name) == "statute" else 1)
    for pdf_path in ordered_pdfs:
        results.append(ingest_pdf(pdf_path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into PostgreSQL + pgvector.")
    parser.add_argument("--directory", default=".", help="Directory containing legal PDFs.")
    args = parser.parse_args()
    results = ingest_directory(Path(args.directory).resolve())
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
