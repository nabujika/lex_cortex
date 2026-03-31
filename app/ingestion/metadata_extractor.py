from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b"),
    re.compile(r"\b([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b"),
]
CITATION_PREFIXES = ("equivalent citations", "citation", "citations")


@dataclass(slots=True)
class ExtractedMetadata:
    title: str
    court_name: str | None = None
    judgment_date: date | None = None
    filing_date: date | None = None
    status: str | None = None
    judge_name: str | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)


def _parse_date(value: str) -> date | None:
    for fmt in ("%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _clean_title_from_filename(pdf_path: Path) -> str:
    stem = pdf_path.stem.replace("_", " ").strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def infer_court_name_from_text(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bench_candidate = None
    for line in lines[:60]:
        lower = line.lower()
        if lower.startswith(CITATION_PREFIXES):
            continue
        if "in the" in lower and "court" in lower:
            normalized = re.sub(r"\s+", " ", line)
            return normalized[:255]
        if "court of" in lower or "high court" in lower or "supreme court" in lower:
            normalized = re.sub(r"\s+", " ", line)
            return normalized[:255]
        if "tribunal" in lower:
            normalized = re.sub(r"\s+", " ", line)
            return normalized[:255]
        if "bench" in lower and bench_candidate is None:
            bench_candidate = re.sub(r"\s+", " ", line)[:255]
    return bench_candidate


def infer_judge_name_from_text(text: str) -> str | None:
    bench_match = re.search(r"Bench:\s*([^\n]+)", text, re.IGNORECASE)
    if bench_match:
        return bench_match.group(1).strip()[:255]
    author_match = re.search(r"Author:\s*([^\n]+)", text, re.IGNORECASE)
    if author_match:
        return author_match.group(1).strip()[:255]
    judge_match = re.search(r"(Hon'?ble.*?Justice[^\n]+|Justice[^\n]+)", text, re.IGNORECASE)
    if judge_match:
        return judge_match.group(1).strip()[:255]
    return None


def extract_metadata_from_text(pdf_path: Path, first_page_text: str, full_text: str) -> ExtractedMetadata:
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    title = lines[0] if lines else _clean_title_from_filename(pdf_path)
    if len(title) < 8:
        title = _clean_title_from_filename(pdf_path)

    court_name = infer_court_name_from_text(first_page_text)
    judge_name = infer_judge_name_from_text(first_page_text)

    status = None
    lowered = full_text.lower()
    if "dismissed" in lowered:
        status = "dismissed"
    elif "allowed" in lowered:
        status = "allowed"
    elif "disposed of" in lowered:
        status = "disposed"
    elif "pending" in lowered:
        status = "pending"

    judgment_date = None
    for pattern in DATE_PATTERNS:
        match = pattern.search(first_page_text)
        if match:
            judgment_date = _parse_date(match.group(1))
            if judgment_date:
                break

    if judgment_date is None:
        logger.warning("Could not infer judgment date for %s", pdf_path.name)
    if court_name is None:
        logger.warning("Could not infer court name for %s", pdf_path.name)
    if judge_name is None:
        logger.warning("Could not infer judge name for %s", pdf_path.name)

    return ExtractedMetadata(
        title=title[:500],
        court_name=court_name,
        judgment_date=judgment_date,
        filing_date=None,
        status=status,
        judge_name=judge_name,
        raw_fields={"source_file": str(pdf_path)},
    )
