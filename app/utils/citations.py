from __future__ import annotations

from typing import Any


def format_citation(chunk: dict[str, Any]) -> str:
    title = chunk.get("title") or "Unknown Case"
    page_number = chunk.get("page_number")
    chunk_id = chunk.get("chunk_id") or "unknown-chunk"
    return f"{title} | page {page_number} | chunk {chunk_id}"


def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        citations.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "case_id": chunk.get("case_id"),
                "title": chunk.get("title"),
                "page_number": chunk.get("page_number"),
                "chunk_type": chunk.get("chunk_type"),
                "citation": format_citation(chunk),
            }
        )
    return citations
