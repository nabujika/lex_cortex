from __future__ import annotations

from datetime import date
import re

from app.core.config import get_settings
from app.graph.state import GraphState
from app.ingestion.embedder import AzureOpenAIClient
from app.retrievers.azure_search import AzureSearchIndexer
from app.retrievers.sql_lookup import SQLLookupTool
from app.utils.citations import build_citations


def determine_route_heuristic(question: str) -> str:
    lowered = question.lower()
    heuristic_route = "retriever"
    if any(term in lowered for term in ("which court", "judge", "filing date", "judgment date", "status", "statute version")):
        heuristic_route = "sql"
    if any(term in lowered for term in ("summarize", "reasoning", "similar", "dissent", "find cases")):
        heuristic_route = "retriever"
    if any(term in lowered for term in ("and also", "along with", "together with", "plus")):
        heuristic_route = "both"
    return heuristic_route


def classify_intent(state: GraphState, llm: AzureOpenAIClient) -> GraphState:
    return {"route": determine_route_heuristic(state["question"])}


def _normalized_query(question: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", question.lower())
    cleaned = re.sub(
        r"\b(which|what|show|find|summarize|summary|reasoning|explain|please|case|cases|did|does|the|a|an)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or question


def maybe_rewrite_query(state: GraphState, llm: AzureOpenAIClient) -> GraphState:
    question = state["question"]
    retrieved = state.get("retrieved_chunks", [])
    if len(retrieved) >= get_settings().rewrite_min_results:
        return {"rewritten_question": question}
    return {"rewritten_question": _normalized_query(question)}


def sql_lookup_node(state: GraphState, sql_tool: SQLLookupTool) -> GraphState:
    return {"sql_results": sql_tool.run(state["question"])}


def _search_filter_from_sql_results(sql_results: list[dict]) -> str | None:
    if not sql_results:
        return None
    top_case_id = sql_results[0].get("case_id")
    if top_case_id:
        return f"case_id eq '{top_case_id}'"
    return None


def retrieve_chunks_node(
    state: GraphState,
    llm: AzureOpenAIClient,
    search: AzureSearchIndexer,
    sql_tool: SQLLookupTool,
) -> GraphState:
    query = state.get("rewritten_question") or state["question"]
    filter_expression = _search_filter_from_sql_results(state.get("sql_results", []))
    case_id = None
    if filter_expression and "case_id eq '" in filter_expression:
        case_id = filter_expression.split("case_id eq '", 1)[1].rstrip("'")
    try:
        chunks = search.text_search(query=query, top_k=get_settings().retrieval_k, filter_expression=filter_expression)
    except Exception:
        chunks = sql_tool.local_chunk_search(question=query, top_k=get_settings().retrieval_k, case_id=case_id)
    return {"retrieved_chunks": chunks}


def grade_retrieved_docs(state: GraphState, llm: AzureOpenAIClient) -> GraphState:
    if state.get("route") == "sql":
        return {"filtered_chunks": []}
    filtered: list[dict] = []
    query_tokens = set(_normalized_query(state["question"]).split())
    for chunk in state.get("retrieved_chunks", []):
        score = float(chunk.get("reranker_score") or chunk.get("score") or 0.0)
        haystack = f"{chunk.get('title', '')} {chunk.get('chunk_text', '')}".lower()
        token_hits = sum(1 for token in query_tokens if token and token in haystack)
        if score > 0.5 or token_hits > 0:
            filtered.append({**chunk, "token_hits": token_hits})
    if not filtered:
        filtered = state.get("retrieved_chunks", [])[: min(4, len(state.get("retrieved_chunks", [])))]
    filtered = sorted(
        filtered,
        key=lambda item: (item.get("token_hits", 0), item.get("reranker_score") or item.get("score") or 0.0),
        reverse=True,
    )[:6]
    return {"filtered_chunks": filtered}


def _format_scalar(value) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _build_sql_answer(question: str, sql_rows: list[dict]) -> str:
    if not sql_rows:
        return "I could not find matching structured metadata in PostgreSQL for that question."

    best = sql_rows[0]
    title = best.get("title") or best.get("short_title") or "The matched record"
    lowered = question.lower()

    if "court" in lowered:
        court_name = best.get("court_name") or "an unidentified court in the stored metadata"
        return f"{title} was heard by {court_name}."
    if "judge" in lowered:
        judge_name = best.get("judge_name") or "an unidentified judge in the stored metadata"
        return f"{title} lists {judge_name} as the presiding judge."
    if "filing date" in lowered:
        filing_date = best.get("filing_date")
        return f"{title} has filing date {_format_scalar(filing_date)}." if filing_date else f"I found {title}, but its filing date is not populated in the stored metadata."
    if "judgment date" in lowered:
        judgment_date = best.get("judgment_date")
        return f"{title} has judgment date {_format_scalar(judgment_date)}." if judgment_date else f"I found {title}, but its judgment date is not populated in the stored metadata."
    if "status" in lowered:
        status = best.get("status")
        return f"{title} is marked as {status}." if status else f"I found {title}, but its status is not populated in the stored metadata."
    if "statute version" in lowered or "applied on" in lowered:
        valid_from = best.get("valid_from")
        valid_to = best.get("valid_to")
        active_text = "active" if best.get("is_active") else "inactive"
        return (
            f"{best.get('short_title', 'The statute')} version {best.get('version_id')} "
            f"is {active_text}, valid from {_format_scalar(valid_from)} to {_format_scalar(valid_to)}."
        )

    preview = ", ".join(
        f"{key}={_format_scalar(value)}"
        for key, value in best.items()
        if key != "match_score" and value is not None
    )
    return f"Best structured match: {preview}."


def _chunk_excerpt(text: str, max_chars: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    cut = normalized[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}..."


def _build_retrieval_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I could not find enough indexed evidence to answer that question confidently."

    lowered = question.lower()
    unique_titles: list[str] = []
    for chunk in chunks:
        title = chunk.get("title")
        if title and title not in unique_titles:
            unique_titles.append(title)

    if "similar" in lowered:
        lines = ["I found these closely related results in the indexed corpus:"]
        for chunk in chunks[:5]:
            lines.append(
                f"- {chunk.get('title')} (page {chunk.get('page_number')}): {_chunk_excerpt(chunk.get('chunk_text', ''))}"
            )
        return "\n".join(lines)

    if "dissent" in lowered:
        dissent_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "dissent"]
        if not dissent_chunks:
            return "I did not find any retrieved chunks explicitly labeled as dissent in the current indexed corpus."
        lines = ["I found dissent-related chunks in these materials:"]
        for chunk in dissent_chunks[:5]:
            lines.append(
                f"- {chunk.get('title')} (page {chunk.get('page_number')}): {_chunk_excerpt(chunk.get('chunk_text', ''))}"
            )
        return "\n".join(lines)

    if "summarize" in lowered or "reasoning" in lowered or "summary" in lowered:
        lead_title = unique_titles[0] if unique_titles else "the retrieved case material"
        excerpts = " ".join(_chunk_excerpt(chunk.get("chunk_text", ""), 220) for chunk in chunks[:3])
        return f"Based on the top retrieved chunks from {lead_title}, the reasoning centers on: {excerpts}"

    lines = ["Top grounded evidence from the corpus:"]
    for chunk in chunks[:4]:
        lines.append(f"- {chunk.get('title')} (page {chunk.get('page_number')}): {_chunk_excerpt(chunk.get('chunk_text', ''))}")
    return "\n".join(lines)


def generate_answer_node(state: GraphState, llm: AzureOpenAIClient) -> GraphState:
    sql_blob = state.get("sql_results", [])
    evidence = state.get("filtered_chunks", [])
    if state.get("route") == "sql":
        return {"answer": _build_sql_answer(state["question"], sql_blob)}
    return {"answer": _build_retrieval_answer(state["question"], evidence)}


def format_citations_node(state: GraphState) -> GraphState:
    return {"citations": build_citations(state.get("filtered_chunks", []))}


def should_retry_retrieval(state: GraphState) -> str:
    rewritten = (state.get("rewritten_question") or "").strip()
    original = state["question"].strip()
    retrieved = state.get("retrieved_chunks", [])
    if len(retrieved) < get_settings().rewrite_min_results and rewritten and rewritten != original:
        return "retry"
    return "grade"


def route_after_sql(state: GraphState) -> str:
    if state.get("route") == "both":
        return "retrieve"
    return "answer"
