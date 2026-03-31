from __future__ import annotations

from typing import Literal, TypedDict


RouteType = Literal["sql", "retriever", "both"]


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: str
    route: RouteType
    sql_results: list[dict]
    retrieved_chunks: list[dict]
    filtered_chunks: list[dict]
    citations: list[dict]
    answer: str
    error: str
