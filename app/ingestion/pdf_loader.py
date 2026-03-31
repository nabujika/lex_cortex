from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass(slots=True)
class PDFPage:
    page_number: int
    text: str


def load_pdf_pages(pdf_path: Path) -> list[PDFPage]:
    reader = pypdf.PdfReader(str(pdf_path))
    pages: list[PDFPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(PDFPage(page_number=index, text=text))
    return pages

