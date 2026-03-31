from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.utils.hashing import stable_hash

SECTION_HINTS = {
    "facts": ("facts", "background", "brief facts"),
    "arguments": ("argument", "contention", "submission"),
    "ruling": ("held", "decision", "judgment", "order", "conclusion"),
    "dissent": ("dissent", "minority", "disagree"),
}


@dataclass(slots=True)
class ChunkedDocument:
    chunk_id: str
    case_id: str
    chunk_text: str
    page_number: int
    chunk_type: str
    metadata: dict


def infer_chunk_type(text: str) -> str:
    lowered = text.lower()
    for chunk_type, hints in SECTION_HINTS.items():
        if any(hint in lowered for hint in hints):
            return chunk_type
    return "general"


def _split_section_aware(text: str) -> list[str]:
    sections = re.split(r"\n(?=(?:facts?|background|arguments?|submissions?|judgment|order|conclusion|dissent)[\s:])", text, flags=re.IGNORECASE)
    return [section.strip() for section in sections if section.strip()]


def chunk_case_text(case_id: str, pages: list[dict]) -> list[ChunkedDocument]:
    settings = get_settings()
    chunks: list[ChunkedDocument] = []
    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"].strip()
        if not page_text:
            continue

        sections = _split_section_aware(page_text)
        for section in sections:
            start = 0
            while start < len(section):
                end = min(start + settings.chunk_size, len(section))
                chunk_text = section[start:end].strip()
                if not chunk_text:
                    break
                chunk_type = infer_chunk_type(chunk_text)
                chunk_id = stable_hash(case_id, page_number, start, chunk_text)
                chunks.append(
                    ChunkedDocument(
                        chunk_id=chunk_id,
                        case_id=case_id,
                        chunk_text=chunk_text,
                        page_number=page_number,
                        chunk_type=chunk_type,
                        metadata={"offset_start": start, "offset_end": end},
                    )
                )
                if end >= len(section):
                    break
                start = max(0, end - settings.chunk_overlap)
    return chunks

