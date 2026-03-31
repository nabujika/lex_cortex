from __future__ import annotations

from sqlalchemy.orm import Session

from app.graph.workflow import build_query_graph


class QueryService:
    def __init__(self, db: Session) -> None:
        self.graph = build_query_graph(db)

    def ask(self, question: str) -> dict:
        try:
            result = self.graph.invoke({"question": question})
        except Exception as exc:
            return {
                "answer": f"Query processing failed: {exc}",
                "citations": [],
                "retrieved_chunks_summary": [],
                "route_used": "error",
            }
        return {
            "answer": result.get("answer"),
            "citations": result.get("citations", []),
            "retrieved_chunks_summary": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "title": chunk.get("title"),
                    "page_number": chunk.get("page_number"),
                    "chunk_type": chunk.get("chunk_type"),
                    "score": chunk.get("score"),
                }
                for chunk in result.get("filtered_chunks", [])
            ],
            "route_used": result.get("route"),
        }
