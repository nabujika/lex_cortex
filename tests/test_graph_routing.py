from app.graph.nodes import _build_sql_answer, determine_route_heuristic, format_citations_node, route_after_sql
from app.utils.citations import format_citation


def test_format_citation():
    citation = format_citation({"title": "Case A", "page_number": 4, "chunk_id": "abc"})
    assert "Case A" in citation
    assert "page 4" in citation


def test_format_citations_node():
    state = {
        "filtered_chunks": [
            {"chunk_id": "a1", "case_id": "c1", "title": "Case A", "page_number": 2, "chunk_type": "facts"}
        ]
    }
    result = format_citations_node(state)
    assert len(result["citations"]) == 1


def test_route_heuristics():
    assert determine_route_heuristic("Which court heard case X?") == "sql"
    assert determine_route_heuristic("Summarize the reasoning in case X") == "retriever"
    assert determine_route_heuristic("Which court heard case X and also summarize it") == "both"


def test_route_after_sql():
    assert route_after_sql({"route": "sql"}) == "answer"
    assert route_after_sql({"route": "both"}) == "retrieve"


def test_build_sql_answer_for_court_question():
    answer = _build_sql_answer(
        "Which court heard Sanjay Chandra vs CBI?",
        [{"title": "Sanjay Chandra vs CBI", "court_name": "Supreme Court of India"}],
    )
    assert "Supreme Court of India" in answer
