from __future__ import annotations

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.graph.nodes import (
    classify_intent,
    format_citations_node,
    generate_answer_node,
    grade_retrieved_docs,
    maybe_rewrite_query,
    route_after_sql,
    retrieve_chunks_node,
    should_retry_retrieval,
    sql_lookup_node,
)
from app.graph.state import GraphState
from app.ingestion.embedder import AzureOpenAIClient
from app.retrievers.azure_search import AzureSearchIndexer
from app.retrievers.sql_lookup import SQLLookupTool


def build_query_graph(db: Session):
    llm = AzureOpenAIClient()
    search = AzureSearchIndexer()
    sql_tool = SQLLookupTool(db)

    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", lambda state: classify_intent(state, llm))
    graph.add_node("sql_lookup", lambda state: sql_lookup_node(state, sql_tool))
    graph.add_node("retrieve_chunks", lambda state: retrieve_chunks_node(state, llm, search, sql_tool))
    graph.add_node("maybe_rewrite_query", lambda state: maybe_rewrite_query(state, llm))
    graph.add_node("retrieve_chunks_retry", lambda state: retrieve_chunks_node(state, llm, search, sql_tool))
    graph.add_node("grade_retrieved_docs", lambda state: grade_retrieved_docs(state, llm))
    graph.add_node("generate_answer", lambda state: generate_answer_node(state, llm))
    graph.add_node("format_citations", format_citations_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state["route"],
        {
            "sql": "sql_lookup",
            "retriever": "retrieve_chunks",
            "both": "sql_lookup",
        },
    )
    graph.add_conditional_edges(
        "sql_lookup",
        route_after_sql,
        {
            "retrieve": "retrieve_chunks",
            "answer": "generate_answer",
        },
    )
    graph.add_edge("retrieve_chunks", "maybe_rewrite_query")
    graph.add_conditional_edges(
        "maybe_rewrite_query",
        should_retry_retrieval,
        {
            "retry": "retrieve_chunks_retry",
            "grade": "grade_retrieved_docs",
        },
    )
    graph.add_edge("retrieve_chunks_retry", "grade_retrieved_docs")
    graph.add_edge("grade_retrieved_docs", "generate_answer")
    graph.add_edge("generate_answer", "format_citations")
    graph.add_edge("format_citations", END)
    return graph.compile()
