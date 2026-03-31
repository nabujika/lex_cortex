import json
import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from db import (
    fetch_case_details,
    fetch_precedents,
    fetch_relevant_statutes,
    get_azure_openai_client,
    get_embedding_dimensions,
    search_similar_case_chunks,
)


class RAGState(TypedDict, total=False):
    query: str
    top_k: int
    filters: Dict[str, Any]
    vector_hits: List[Dict[str, Any]]
    case_details: List[Dict[str, Any]]
    precedents: Dict[int, List[Dict[str, Any]]]
    statute_versions: List[Dict[str, Any]]
    merged_context: str
    answer: str
    cases_used: List[Dict[str, Any]]


def get_chat_client():
    client = get_azure_openai_client()
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_CHAT_DEPLOYMENT is required for query generation.")
    return client, deployment


def analyze_query(state: RAGState) -> RAGState:
    # The LLM converts free-text questions into structured filters the graph can reuse downstream.
    client, deployment = get_chat_client()
    response = client.chat.completions.create(
        model=deployment,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract legal search filters from the query. "
                    "Return JSON with keys judge, court, statute, date_from, date_to, as_of_date, "
                    "needs_precedents, answer_style. Use null when unknown."
                ),
            },
            {"role": "user", "content": state["query"]},
        ],
    )
    return {"filters": json.loads(response.choices[0].message.content or "{}")}


def vector_search(state: RAGState) -> RAGState:
    client = get_azure_openai_client()
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required for semantic search.")
    embedding = client.embeddings.create(
        model=deployment,
        input=state["query"],
        dimensions=get_embedding_dimensions(),
    ).data[0].embedding
    filters = state.get("filters", {})
    hits = search_similar_case_chunks(
        query_embedding=embedding,
        limit=state.get("top_k", 8),
        judge=filters.get("judge"),
        court=filters.get("court"),
        date_from=filters.get("date_from"),
        date_to=filters.get("date_to"),
    )
    return {"vector_hits": hits}


def sql_enrich(state: RAGState) -> RAGState:
    filters = state.get("filters", {})
    case_ids: List[int] = []
    for hit in state.get("vector_hits", []):
        case_id = int(hit["case_id"])
        if case_id not in case_ids:
            case_ids.append(case_id)
    return {
        "case_details": fetch_case_details(case_ids),
        "precedents": fetch_precedents(case_ids) if filters.get("needs_precedents") else {},
        "statute_versions": fetch_relevant_statutes(
            statute_name=filters.get("statute"),
            as_of_date=filters.get("as_of_date") or filters.get("date_to") or filters.get("date_from"),
        ),
    }


def merge_results(state: RAGState) -> RAGState:
    # Merge semantic hits with relational metadata into a citation-friendly prompt context.
    chunk_lines: List[str] = []
    for hit in state.get("vector_hits", []):
        chunk_lines.append(
            (
                f"Case: {hit['title']} | Judge: {hit.get('judge_name') or 'Unknown Judge'} | "
                f"Court: {hit.get('court_name') or 'Unknown Court'} | Page: {hit.get('page_number') or 'n/a'} | "
                f"Similarity: {round(float(hit.get('similarity') or 0), 4)}\n"
                f"Excerpt: {hit['chunk_text']}"
            )
        )

    cases_used: List[Dict[str, Any]] = []
    case_lines: List[str] = []
    for row in state.get("case_details", []):
        cases_used.append({"title": row["title"], "judge": row.get("judge_name"), "court": row.get("court_name")})
        case_lines.append(
            (
                f"Case Metadata: {row['title']} | Judgment Date: {row.get('judgment_date')} | "
                f"Judge: {row.get('judge_name') or 'Unknown Judge'} | "
                f"Court: {row.get('court_name') or 'Unknown Court'} | "
                f"Statutes: {', '.join(row.get('statutes') or []) or 'None'}"
            )
        )

    precedent_lines: List[str] = []
    for case_id, precedents in state.get("precedents", {}).items():
        for precedent in precedents:
            precedent_lines.append(f"Precedent for case_id={case_id}: {precedent['title']} ({precedent.get('judgment_date')})")

    statute_lines: List[str] = []
    for statute in state.get("statute_versions", []):
        statute_lines.append(
            (
                f"Statute: {statute['short_title']} | Act Number: {statute.get('act_number')} | "
                f"Valid From: {statute.get('valid_from')} | Valid To: {statute.get('valid_to')} | "
                f"Excerpt: {statute.get('full_text_excerpt')}"
            )
        )

    merged_context = "\n\n".join(
        section
        for section in [
            "Vector Retrieval Results:\n" + "\n\n".join(chunk_lines) if chunk_lines else "",
            "SQL Enrichment Results:\n" + "\n".join(case_lines) if case_lines else "",
            "Precedent Links:\n" + "\n".join(precedent_lines) if precedent_lines else "",
            "Relevant Statute Versions:\n" + "\n".join(statute_lines) if statute_lines else "",
        ]
        if section
    )
    return {"merged_context": merged_context, "cases_used": cases_used}


def generate_answer(state: RAGState) -> RAGState:
    # Final answer generation is grounded only on the merged retrieval context.
    client, deployment = get_chat_client()
    answer_style = state.get("filters", {}).get("answer_style") or "concise"
    response = client.chat.completions.create(
        model=deployment,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are The Legal Brain, a legal research assistant. "
                    "Answer only from the provided context. Cite supporting cases inline using their titles. "
                    "If support is incomplete, explicitly say what is missing. "
                    f"Preferred style: {answer_style}."
                ),
            },
            {
                "role": "user",
                "content": f"User query:\n{state['query']}\n\nRetrieved context:\n{state.get('merged_context', 'No context found.')}",
            },
        ],
    )
    return {"answer": response.choices[0].message.content or ""}


def build_rag_graph():
    workflow = StateGraph(RAGState)
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("vector_search", vector_search)
    workflow.add_node("sql_enrich", sql_enrich)
    workflow.add_node("merge_results", merge_results)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "vector_search")
    workflow.add_edge("vector_search", "sql_enrich")
    workflow.add_edge("sql_enrich", "merge_results")
    workflow.add_edge("merge_results", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


rag_app = build_rag_graph()


def run_rag_query(query: str, top_k: int = 8) -> Dict[str, Any]:
    result = rag_app.invoke({"query": query, "top_k": top_k})
    return {
        "answer": result.get("answer", ""),
        "cases_used": result.get("cases_used", []),
        "filters": result.get("filters", {}),
    }
