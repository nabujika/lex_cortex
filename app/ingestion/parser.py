from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.metadata_extractor import ExtractedMetadata, extract_metadata_from_text
from app.ingestion.pdf_loader import PDFPage, load_pdf_pages


@dataclass(slots=True)
class ParsedCaseDocument:
    pdf_path: Path
    pages: list[PDFPage]
    full_text: str
    metadata: ExtractedMetadata


def parse_pdf_document(pdf_path: Path) -> ParsedCaseDocument:
    pages = load_pdf_pages(pdf_path)
    full_text = "\n\n".join(page.text for page in pages if page.text)
    first_page_text = pages[0].text if pages else ""
    metadata = extract_metadata_from_text(pdf_path, first_page_text, full_text)
    return ParsedCaseDocument(pdf_path=pdf_path, pages=pages, full_text=full_text, metadata=metadata)

